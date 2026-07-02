"""Experiment 3b: by-construction fine-tuning of GPT-2 (layer-wise).

The frozen-retrofit experiment (exp_gpt2.py) found that on a *frozen* model a
reconstruction-only SAE is the more faithful channel, and the paper's thesis
predicts why: CANDOR's behavioural and causal objectives only pay off when the
model can reshape its own computation to route through the legible channel.
This experiment tests that prediction directly at the smallest honest scale:
the spliced layer's MLP is unfrozen (layer-wise fine-tuning, 4.7M of 124M
parameters) and trained jointly with the channel, under a language-modelling
loss that anchors capability.

Two matched conditions share identical unfrozen parameters, steps, seeds,
batches and learning rates; the ONLY difference is the channel objective:

  FT-SAE    : LM cross-entropy (full model) + normalised reconstruction
  FT-CANDOR : LM cross-entropy (full model) + normalised reconstruction
              + behavioural faithfulness (KL of the spliced legible model
                against the current full model)
              + causal sufficiency (TV against the leak-swapped splice)

Note the reconstruction target is NOT detached in either condition, so both
conditions let the model reshape its MLP output toward reconstructability;
FT-CANDOR additionally trains the routing to be behaviourally faithful and
causally sufficient.  Reported per condition, on held-out text: certificate
delta, leak-swap effect, unexplained variance, and full-model LM loss versus
the original model (the capability check).  Results -> results/gpt2_ft.json.

Runs on a small GPU (about 1.5 GB is enough for the default batch) and falls
back to CPU automatically if the CUDA probe fails.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

# exp_gpt2 defaults to offline mode; this script permits the one-time weight
# download (about 0.5 GB) in case the GPT-2 weights are not cached locally.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.exp_gpt2 import Splicer, Transcoder, load_corpus, tv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
DATA = os.path.join(ROOT, "data")


def _gpt2_src():
    gpt2_dir = os.path.join(DATA, "gpt2")
    return gpt2_dir if os.path.exists(os.path.join(gpt2_dir, "model.safetensors")) else "gpt2"


def fresh_model(layer, device):
    """A fresh GPT-2 with everything frozen except the spliced layer's MLP,
    and a Splicer hook on that MLP.  Kept in eval mode throughout so dropout
    stays off; eval mode does not block gradients."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(_gpt2_src()).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.transformer.h[layer].mlp.parameters():
        p.requires_grad_(True)
    splicer = Splicer()
    model.transformer.h[layer].mlp.register_forward_hook(splicer)
    return model, splicer


def _lm_ce(logits, b):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           b[:, 1:].reshape(-1))


@torch.no_grad()
def ce_loss(model, splicer, seqs, batch, device, mode, tc=None):
    splicer.mode, splicer.tc = mode, tc
    tot, ntok = 0.0, 0
    for i in range(0, seqs.shape[0], batch):
        b = seqs[i:i + batch].to(device)
        lo = model(b).logits
        tot += float(F.cross_entropy(lo[:, :-1].reshape(-1, lo.shape[-1]),
                                     b[:, 1:].reshape(-1), reduction="sum"))
        ntok += b[:, 1:].numel()
    splicer.mode, splicer.tc = "full", None
    return tot / ntok


@torch.no_grad()
def evaluate(model, splicer, tc, ids_va, batch, device, label=""):
    model.eval()
    deltas, swaps = [], []
    rn = rd = 0.0
    for i in range(0, ids_va.shape[0], batch):
        b = ids_va[i:i + batch].to(device)
        splicer.mode, splicer.tc = "full", None
        full = model(b).logits[:, -1, :]
        m_hat, _ = tc(splicer.mlp_in)
        rn += float(((m_hat - splicer.mlp_out) ** 2).sum())
        rd += float((splicer.mlp_out ** 2).sum())
        splicer.tc = tc
        splicer.mode = "legible"
        leg = model(b).logits[:, -1, :]
        splicer.mode = "leakswap"
        sw = model(b).logits[:, -1, :]
        splicer.mode, splicer.tc = "full", None
        deltas.append(tv(full, leg).cpu())
        swaps.append(tv(full, sw).cpu())
    r = dict(delta=float(torch.cat(deltas).mean()),
             leak_swap=float(torch.cat(swaps).mean()),
             recon_unexplained=rn / max(rd, 1e-9),
             loss_full=ce_loss(model, splicer, ids_va, batch, device, "full"),
             loss_legible=ce_loss(model, splicer, ids_va, batch, device, "legible", tc),
             loss_ablated=ce_loss(model, splicer, ids_va, batch, device, "ablate"))
    print(f"  {label}: delta={r['delta']:.4f} swap={r['leak_swap']:.4f} "
          f"recon_unexpl={r['recon_unexplained']:.3f} loss_full={r['loss_full']:.3f} "
          f"loss_leg={r['loss_legible']:.3f} loss_abl={r['loss_ablated']:.3f}", flush=True)
    return r


def run_condition(candor, ids_tr, ids_va, layer, m, k, steps, batch,
                  lr_tc, lr_mlp, recon_w, faith_w, causal_w, seed, device,
                  log_every=50):
    name = "FT-CANDOR" if candor else "FT-SAE  "
    torch.manual_seed(seed)
    model, splicer = fresh_model(layer, device)
    d = model.config.n_embd
    tc = Transcoder(d, m, k).to(device)
    opt = torch.optim.Adam([
        {"params": tc.parameters(), "lr": lr_tc},
        {"params": model.transformer.h[layer].mlp.parameters(), "lr": lr_mlp},
    ])
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, ids_tr.shape[0], size=batch)
        b = ids_tr[idx].to(device)
        splicer.mode, splicer.tc = "full", None
        logits = model(b).logits
        lm = _lm_ce(logits, b)
        # live (non-detached) reconstruction: trains the channel AND lets the
        # MLP reshape its output toward reconstructability, in BOTH conditions
        m_hat, _ = tc(splicer.mlp_in)
        x_out = splicer.mlp_out
        recon = (((m_hat - x_out) ** 2).sum(-1) / ((x_out ** 2).sum(-1) + 1e-6)).mean()
        loss = lm + recon_w * recon
        faith = causal = None
        if candor:
            tgt = logits[:, -1, :].detach()
            splicer.tc = tc
            splicer.mode = "legible"
            leg = model(b).logits[:, -1, :]
            faith = F.kl_div(F.log_softmax(leg, -1), F.softmax(tgt, -1),
                             reduction="batchmean")
            splicer.mode = "leakswap"
            sw = model(b).logits[:, -1, :]
            causal = tv(tgt, sw).mean()
            loss = loss + faith_w * faith + causal_w * causal
            splicer.mode, splicer.tc = "full", None
        opt.zero_grad()
        loss.backward()
        opt.step()
        tc.normalize()
        if step % log_every == 0 or step == steps - 1:
            msg = (f"  [{name}] step {step:4d}  lm={float(lm.detach()):.3f}  "
                   f"recon={float(recon.detach()):.3f}")
            if candor:
                msg += (f"  faith={float(faith.detach()):.4f}"
                        f"  causal={float(causal.detach()):.4f}")
            print(msg, flush=True)
    train_sec = round(time.time() - t0, 1)
    r = evaluate(model, splicer, tc, ids_va, batch, device, label=name.strip())
    r["train_sec"] = train_sec
    del model, splicer, tc, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return r


def pick_device(pref, layer, batch, seq_len, m, k):
    if pref != "auto":
        return pref
    if not torch.cuda.is_available():
        return "cpu"
    try:
        model, splicer = fresh_model(layer, "cuda")
        tc = Transcoder(model.config.n_embd, m, k).to("cuda")
        b = torch.randint(0, 1000, (batch, seq_len), device="cuda")
        logits = model(b).logits
        lm = _lm_ce(logits, b)
        m_hat, _ = tc(splicer.mlp_in)
        x_out = splicer.mlp_out
        recon = (((m_hat - x_out) ** 2).sum(-1) / ((x_out ** 2).sum(-1) + 1e-6)).mean()
        tgt = logits[:, -1, :].detach()
        splicer.tc = tc
        splicer.mode = "legible"
        leg = model(b).logits[:, -1, :]
        faith = F.kl_div(F.log_softmax(leg, -1), F.softmax(tgt, -1), reduction="batchmean")
        splicer.mode = "leakswap"
        sw = model(b).logits[:, -1, :]
        causal = tv(tgt, sw).mean()
        (lm + recon + faith + causal).backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, splicer, tc, b, logits, m_hat, x_out, tgt, leg, sw
        gc.collect()
        torch.cuda.empty_cache()
        print(f"cuda probe ok (peak {peak:.2f} GB); using the GPU", flush=True)
        return "cuda"
    except Exception as e:
        print(f"cuda probe failed ({type(e).__name__}: {e}); using cpu", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return "cpu"


def main(layer=9, seq_len=24, n_seq=1400, m=2048, k=32, steps=500, batch=12,
         lr_tc=2e-3, lr_mlp=1e-4, recon_w=1.0, faith_w=1.0, causal_w=1.0,
         seed=0, device="auto", out=os.path.join(RESULTS, "gpt2_ft.json")):
    t0 = time.time()
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(_gpt2_src())
    ids = load_corpus(tok, seq_len, n_seq)
    n_tr = int(0.85 * ids.shape[0])
    tr, va = ids[:n_tr], ids[n_tr:]
    dev = pick_device(device, layer, batch, seq_len, m, k)
    print(f"device {dev} | {tr.shape[0]} train / {va.shape[0]} val seqs (len {seq_len}) | "
          f"unfreezing layer {layer} MLP only | {steps} steps, batch {batch}", flush=True)

    # capability reference: the original, un-finetuned model on the same split
    model0, splicer0 = fresh_model(layer, dev)
    orig_loss = ce_loss(model0, splicer0, va, batch, dev, "full")
    print(f"original GPT-2 val loss: {orig_loss:.3f}", flush=True)
    del model0, splicer0
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    common = dict(ids_tr=tr, ids_va=va, layer=layer, m=m, k=k, steps=steps,
                  batch=batch, lr_tc=lr_tc, lr_mlp=lr_mlp, recon_w=recon_w,
                  faith_w=faith_w, causal_w=causal_w, seed=seed, device=dev)
    print("training FT-SAE (LM + reconstruction only) ...", flush=True)
    res_sae = run_condition(candor=False, **common)
    print("training FT-CANDOR (LM + reconstruction + faithfulness + causal) ...", flush=True)
    res_can = run_condition(candor=True, **common)

    frozen = None
    fz = os.path.join(RESULTS, "gpt2.json")
    if os.path.exists(fz):
        fzd = json.load(open(fz))
        frozen = {"sae": fzd["sae"], "candor": fzd["candor"]}

    out_d = {
        "model": "gpt2", "layer": layer, "unfrozen": "mlp-of-spliced-layer",
        "seq_len": seq_len, "n_train": int(tr.shape[0]), "n_val": int(va.shape[0]),
        "m": m, "k": k, "steps": steps, "batch": batch,
        "lr_tc": lr_tc, "lr_mlp": lr_mlp,
        "recon_w": recon_w, "faith_w": faith_w, "causal_w": causal_w,
        "seed": seed, "device": dev,
        "orig_loss_full": orig_loss,
        "ft_sae": res_sae, "ft_candor": res_can,
        "frozen_reference": frozen,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out, "w") as f:
        json.dump(out_d, f, indent=2)
    print(f"wrote {out}  ({out_d['elapsed_sec']}s)", flush=True)
    return out_d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--n-seq", type=int, default=1400)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=os.path.join(RESULTS, "gpt2_ft.json"))
    a = ap.parse_args()
    main(steps=a.steps, n_seq=a.n_seq, batch=a.batch, device=a.device, out=a.out)

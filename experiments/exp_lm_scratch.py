"""Experiment 4: by-construction language-model training (the decisive test).

Every real-LLM result so far in this project is a retrofit: the model's
computation was laid down without the channel (frozen splice, or a single
fine-tuned MLP), and there the reconstruction-only control certified as
tightly or tighter than CANDOR.  The paper's thesis predicts the objectives
only pay when the network is trained *with* its bottlenecks from the first
step.  This experiment runs that test at the smallest honest scale: train a
language model from scratch on real text, with the channel in place, and ask
whether the CANDOR objectives buy behavioural faithfulness that
reconstruction pressure alone does not, and at what capability tax.

Three conditions, matched in architecture, data order, optimiser, steps and
batch, trained per seed:

  OPAQUE     : no bottlenecks; plain next-token cross-entropy.  The
               capability reference, and the substrate for the post-hoc
               pipeline: TopK SAEs (matched m, k) are fitted to its MLP
               hidden states afterwards and spliced back in, both at the
               final site only (as exp_gpt2.py does) and at every site (the
               ablation-matched comparison).
  RECON-ONLY : bottlenecks present, trained with cross-entropy plus leak
               energy only.  The control that separates "a channel exists
               and reconstructs" from "the CANDOR objectives act on
               behaviour".
  CANDOR     : the full conjoined objective (candorkit.candor_lm_loss).

Metrics per condition, on the held-out split: full-model LM loss (the tax is
CANDOR full vs OPAQUE full), legible-model LM loss, mean and p95 certificate
delta (TV between full and leak-ablated next-token distributions over all
positions), leak-swap effect, top-1 agreement, and per-site unexplained
variance.  Results -> results/lm_scratch.json.

Data: TinyShakespeare, GPT-2 tokenisation, sequence length 128, 90/10 split
by sequence.  Needs the [llm] extra (tokeniser download on first run).

    python experiments/exp_lm_scratch.py                          # both seeds
    python experiments/exp_lm_scratch.py --seed 1 --partial DIR   # one seed
    python experiments/exp_lm_scratch.py --aggregate DIR          # merge
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candorkit.losses import (LossWeights, candor_lm_loss, leak_energy,  # noqa: E402
                              lm_cross_entropy)
from candorkit.model import LegibleLMTransformer, OpaqueLMTransformer  # noqa: E402
from candorkit.train import fit_sae  # noqa: E402
from experiments.exp_gpt2 import CORPUS_URL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results"); os.makedirs(RESULTS, exist_ok=True)
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
OUT = os.path.join(RESULTS, "lm_scratch.json")

SEEDS = (0, 1)

# One architecture and schedule for every condition; plain fp32 throughout
# (no mixed precision), plain Adam, no schedule, no weight decay.
ARCH = dict(seq_len=128, d_model=256, n_heads=4, d_mlp=1024, n_layers=4,
            m=1024, k=32)
TRAIN = dict(steps=4000, batch=32, lr=3e-4)
WEIGHTS = LossWeights()          # library defaults, identical across seeds
SAE_FIT = dict(steps=4000, lr=1e-3, batch=1024)   # post-hoc SAE fitting


def load_lm_corpus(seq_len=128, val_frac=0.1, seed=0):
    """TinyShakespeare -> GPT-2 tokens -> non-overlapping length-seq_len
    sequences, shuffled and split 90/10 by sequence (fixed corpus seed)."""
    import urllib.request
    from transformers import GPT2TokenizerFast
    path = os.path.join(DATA, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(CORPUS_URL, path)
    gpt2_dir = os.path.join(DATA, "gpt2")
    src = gpt2_dir if os.path.isdir(gpt2_dir) else "gpt2"
    tok = GPT2TokenizerFast.from_pretrained(src)
    ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    nb = ids.shape[0] // seq_len
    ids = ids[:nb * seq_len].reshape(nb, seq_len)
    perm = np.random.default_rng(seed).permutation(nb)
    n_va = max(1, int(round(val_frac * nb)))
    va, tr = ids[perm[:n_va]], ids[perm[n_va:]]
    return tr, va, int(tok.vocab_size)


def tv_rows(a, b):
    """Total variation between two (N, V) logit blocks, per row."""
    return 0.5 * (F.softmax(a, -1) - F.softmax(b, -1)).abs().sum(-1)


def make_model(condition, vocab, device):
    if condition == "opaque":
        return OpaqueLMTransformer(vocab=vocab, **{k: v for k, v in ARCH.items()
                                                   if k not in ("m", "k")}).to(device)
    return LegibleLMTransformer(vocab=vocab, **ARCH).to(device)


def train_condition(condition, tr, vocab, seed, device, log_every=400):
    """Train one condition; identical seeding and batch schedule across
    conditions, so the data order is matched exactly."""
    torch.manual_seed(seed)
    model = make_model(condition, vocab, device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN["lr"])
    rng = np.random.default_rng(seed)
    tr_dev = tr.to(device)
    t0 = time.time()
    for step in range(TRAIN["steps"]):
        idx = rng.integers(0, tr_dev.shape[0], size=TRAIN["batch"])
        b = tr_dev[idx]
        model.train()
        if condition == "candor":
            lb = candor_lm_loss(model, b, WEIGHTS)
            loss, parts = lb.total, lb.parts
        elif condition == "recon":
            logits = model(b, mode="full")
            l_task = lm_cross_entropy(logits, b)
            l_leak = leak_energy(model.sites)
            loss = l_task + WEIGHTS.leak * l_leak
            parts = {"task": float(l_task.detach()), "leak": float(l_leak.detach())}
        else:  # opaque
            loss = lm_cross_entropy(model(b), b)
            parts = {"task": float(loss.detach())}
        opt.zero_grad()
        loss.backward()
        opt.step()
        if condition != "opaque":
            model.maintain()
        if step % log_every == 0 or step == TRAIN["steps"] - 1:
            msg = "  ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"  [{condition:6s} seed {seed}] step {step:5d}  {msg}", flush=True)
    return model, round(time.time() - t0, 1)


@torch.no_grad()
def lm_loss(model, seqs, batch, device, **fwd):
    tot, ntok = 0.0, 0
    for i in range(0, seqs.shape[0], batch):
        b = seqs[i:i + batch].to(device)
        lo = model(b, **fwd)
        tot += float(F.cross_entropy(lo[:, :-1].reshape(-1, lo.shape[-1]),
                                     b[:, 1:].reshape(-1), reduction="sum"))
        ntok += b[:, 1:].numel()
    return tot / ntok


@torch.no_grad()
def evaluate(model, va, batch, device, seed, sites=None):
    """Certificate metrics for a bottleneck-routed LM on held-out text.
    ``sites`` restricts the legible/leak-swap routing to those block indices
    (None = every site, the model's own certificate)."""
    model.eval()
    torch.manual_seed(1000 * (seed + 1) + 7)   # reproducible leak-swap draws
    deltas, swaps, agree, ntot = [], [], 0, 0
    rn = rd = None
    for i in range(0, va.shape[0], batch):
        b = va[i:i + batch].to(device)
        full = model(b, mode="full")
        site_out = list(model.sites)
        leg = model(b, mode="legible", sites=sites)
        sw = model(b, mode="full", leak_swap=True, sites=sites)
        V = full.shape[-1]
        deltas.append(tv_rows(full.reshape(-1, V), leg.reshape(-1, V)).cpu())
        swaps.append(tv_rows(full.reshape(-1, V), sw.reshape(-1, V)).cpu())
        agree += int((full.argmax(-1) == leg.argmax(-1)).sum())
        ntot += full.shape[0] * full.shape[1]
        if rn is None:
            rn = [0.0] * len(site_out); rd = [0.0] * len(site_out)
        for j, s in enumerate(site_out):
            rn[j] += float((s.leak ** 2).sum())
            rd[j] += float(((s.recon + s.leak) ** 2).sum())
    per_site = [n / max(d, 1e-9) for n, d in zip(rn, rd)]
    delta = torch.cat(deltas); swap = torch.cat(swaps)
    return {
        "delta": float(delta.mean()), "delta_p95": float(delta.quantile(0.95)),
        "leak_swap": float(swap.mean()),
        "agreement": agree / max(1, ntot),
        "loss_full": lm_loss(model, va, batch, device, mode="full"),
        "loss_legible": lm_loss(model, va, batch, device, mode="legible",
                                sites=sites),
        "recon_unexplained_per_site": per_site,
        "recon_unexplained": float(np.mean(per_site)),
        "ablated_sites": sorted(sites) if sites is not None else "all",
    }


@torch.no_grad()
def collect_hidden(opaque, seqs, batch, device):
    """MLP hidden states of every block over a set of sequences (rows on CPU)."""
    opaque.eval()
    outs = None
    for i in range(0, seqs.shape[0], batch):
        b = seqs[i:i + batch].to(device)
        opaque(b, keep_hidden=True)
        if outs is None:
            outs = [[] for _ in opaque.hidden]
        for j, h in enumerate(opaque.hidden):
            outs[j].append(h.cpu())
    return [torch.cat(chunks) for chunks in outs]


def run_seed(seed, device):
    tr, va, vocab = load_lm_corpus(ARCH["seq_len"])
    print(f"[seed {seed}] {tr.shape[0]} train / {va.shape[0]} val sequences of "
          f"length {ARCH['seq_len']}, vocab {vocab}, device {device}", flush=True)
    batch = TRAIN["batch"]
    block = {"seed": seed, "n_train": int(tr.shape[0]), "n_val": int(va.shape[0]),
             "vocab": vocab, "device": device}

    # ---- OPAQUE: capability reference + post-hoc SAE pipeline ----
    opaque, secs = train_condition("opaque", tr, vocab, seed, device)
    res_op = {"loss_full": lm_loss(opaque, va, batch, device), "train_sec": secs}
    print(f"  [opaque seed {seed}] val loss {res_op['loss_full']:.4f}", flush=True)

    t0 = time.time()
    hid = collect_hidden(opaque, tr, batch, device)
    saes = []
    for j, acts in enumerate(hid):
        sae = fit_sae(acts, ARCH["m"], ARCH["k"], steps=SAE_FIT["steps"],
                      lr=SAE_FIT["lr"], batch=SAE_FIT["batch"], device=device,
                      seed=seed)
        saes.append(sae)
        print(f"  [opaque seed {seed}] fitted post-hoc SAE at site {j}", flush=True)
    del hid
    twin = LegibleLMTransformer(vocab=vocab, **ARCH).to(device)
    twin.load_opaque(opaque)
    for blk, sae in zip(twin.blocks, saes):
        blk.bottleneck = sae
    res_op["sae_fit_sec"] = round(time.time() - t0, 1)
    last = ARCH["n_layers"] - 1
    res_op["sae_final"] = evaluate(twin, va, batch, device, seed, sites={last})
    res_op["sae_all"] = evaluate(twin, va, batch, device, seed)
    print(f"  [opaque+sae seed {seed}] delta final={res_op['sae_final']['delta']:.4f} "
          f"all={res_op['sae_all']['delta']:.4f}", flush=True)
    del opaque, twin, saes
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    block["opaque"] = res_op

    # ---- RECON-ONLY and CANDOR: by-construction conditions ----
    for condition in ("recon", "candor"):
        model, secs = train_condition(condition, tr, vocab, seed, device)
        res = evaluate(model, va, batch, device, seed)
        fin = evaluate(model, va, batch, device, seed, sites={last})
        res["final_site"] = {k: fin[k] for k in ("delta", "leak_swap",
                                                 "loss_legible")}
        res["train_sec"] = secs
        print(f"  [{condition} seed {seed}] loss_full={res['loss_full']:.4f} "
              f"loss_leg={res['loss_legible']:.4f} delta={res['delta']:.4f} "
              f"swap={res['leak_swap']:.4f}", flush=True)
        block[condition] = res
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return block


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0}


def aggregate(blocks):
    blocks = sorted(blocks, key=lambda b: b["seed"])
    keys = ("delta", "delta_p95", "leak_swap", "agreement", "loss_full",
            "loss_legible", "recon_unexplained")
    summary = {}
    for cond in ("recon", "candor"):
        summary[cond] = {k: _stats([b[cond][k] for b in blocks]) for k in keys}
    for sub in ("sae_final", "sae_all"):
        summary[sub] = {k: _stats([b["opaque"][sub][k] for b in blocks])
                        for k in keys}
    summary["opaque_loss_full"] = _stats([b["opaque"]["loss_full"] for b in blocks])
    for cond in ("recon", "candor"):
        summary[f"{cond}_tax"] = _stats(
            [b[cond]["loss_full"] - b["opaque"]["loss_full"] for b in blocks])
    return {
        "config": {**ARCH, **TRAIN, "weights": asdict(WEIGHTS),
                   "recon_only_leak_weight": WEIGHTS.leak,
                   "sae_fit": SAE_FIT, "val_frac": 0.1, "corpus_seed": 0,
                   "corpus": "tinyshakespeare", "tokeniser": "gpt2",
                   "optimiser": "adam", "precision": "fp32"},
        "seeds": [b["seed"] for b in blocks],
        "n_train": blocks[0]["n_train"], "n_val": blocks[0]["n_val"],
        "vocab": blocks[0]["vocab"], "device": blocks[0]["device"],
        "runs": {str(b["seed"]): {c: b[c] for c in ("opaque", "recon", "candor")}
                 for b in blocks},
        "summary": summary,
    }


def main(seeds=SEEDS, device=None, out=OUT):
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blocks = [run_seed(s, device) for s in seeds]
    agg = aggregate(blocks)
    agg["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"wrote {out}  ({agg['elapsed_sec']}s)", flush=True)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=None,
                    help="run a single seed; write a partial JSON to --partial")
    ap.add_argument("--partial", default=None)
    ap.add_argument("--aggregate", default=None,
                    help="aggregate partial JSONs from this directory and exit")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.steps:
        TRAIN["steps"] = a.steps
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.aggregate:
        files = sorted(f for f in os.listdir(a.aggregate) if f.endswith(".json"))
        blocks = [json.load(open(os.path.join(a.aggregate, f))) for f in files]
        agg = aggregate(blocks)
        agg["elapsed_sec"] = sum(
            b[c]["train_sec"] for b in blocks for c in ("opaque", "recon", "candor"))
        with open(a.out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"wrote {a.out} from {len(blocks)} partial(s)")
    elif a.seed is not None:
        os.makedirs(a.partial or ".", exist_ok=True)
        block = run_seed(a.seed, dev)
        path = os.path.join(a.partial or ".", f"seed{a.seed}.json")
        with open(path, "w") as f:
            json.dump(block, f, indent=2)
        print(f"wrote {path}")
    else:
        main(device=dev, out=a.out)

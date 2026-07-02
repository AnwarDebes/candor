"""Experiment 3: a real pretrained LLM (GPT-2 small, 124M).

I retrofit a Legible Bottleneck as a transcoder on one GPT-2 MLP layer and ask
the question that matters on a real model: does CANDOR's *behavioural* objective
(train the leak-ablated splice to reproduce the model, plus causal-sufficiency)
give a more faithful certified channel than a reconstruction-only SAE at matched
sparsity?  The certificate delta is measured by splicing the channel into GPT-2
(via a forward hook on the layer's MLP) and running the *real* model on held-out
text, so causal attention masking etc. are exactly the model's own.

Honest scope: GPT-2 small is a real, widely-studied LLM but not a frontier model;
this box has a 2.1 GB GPU, so GPT-2 runs on CPU.  The point is that the
tax/legibility/faithfulness story survives the move from synthetic ground truth to
a real language model, not a claim of frontier scale.  Results -> results/gpt2.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # load GPT-2 from the local copy,
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # not the (stalled) hub cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from candorkit.bottleneck import topk_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results"); os.makedirs(RESULTS, exist_ok=True)
DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
DEVICE = "cpu"
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class Transcoder(nn.Module):
    """Reads a layer's MLP input, emits a sparse named code, predicts the MLP output."""

    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=True)
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(self.dec.weight, dim=0))
            self.enc.weight.copy_(self.dec.weight.t())

    def forward(self, x_in):
        code = topk_mask(F.relu(self.enc(x_in)), self.k)
        return self.dec(code), code

    def normalize(self):
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(self.dec.weight, dim=0))


class Splicer:
    """Forward hook on the layer-l MLP. Modes: 'full' (passthrough, also captures
    mlp_in/mlp_out), 'legible' (replace MLP output by transcoder prediction),
    'leakswap' (prediction + a batch-permuted leak), 'ablate' (zero the MLP)."""

    def __init__(self):
        self.mode = "full"; self.tc = None; self.mlp_in = None; self.mlp_out = None

    def __call__(self, module, inp, out):
        x_in = inp[0]
        self.mlp_in, self.mlp_out = x_in, out
        if self.mode == "ablate":
            return torch.zeros_like(out)
        if self.mode == "full" or self.tc is None:
            return out
        m_hat, _ = self.tc(x_in)
        if self.mode == "legible":
            return m_hat
        if self.mode == "leakswap":
            leak = out - m_hat
            return m_hat + leak[torch.randperm(out.shape[0], device=out.device)]
        return out


def load_corpus(tok, seq_len, n_seq, seed=0):
    path = os.path.join(DATA, "tinyshakespeare.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(CORPUS_URL, path)
    ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    nb = ids.shape[0] // seq_len
    ids = ids[:nb * seq_len].reshape(nb, seq_len)
    sel = np.random.default_rng(seed).permutation(nb)[:n_seq]
    return ids[sel]


def tv(a, b):
    return 0.5 * (F.softmax(a, -1) - F.softmax(b, -1)).abs().sum(-1)


def main(layer=9, seq_len=24, n_seq=1400, m=2048, k=32, steps=250, batch=12,
         faith_w=1.0, causal_w=1.0):
    t0 = time.time()
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    gpt2_dir = os.path.join(DATA, "gpt2")
    src = gpt2_dir if os.path.exists(os.path.join(gpt2_dir, "model.safetensors")) else "gpt2"
    tok = GPT2TokenizerFast.from_pretrained(src)
    model = GPT2LMHeadModel.from_pretrained(src).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.n_embd
    splicer = Splicer()
    model.transformer.h[layer].mlp.register_forward_hook(splicer)

    ids = load_corpus(tok, seq_len, n_seq)
    n_tr = int(0.85 * ids.shape[0]); tr, va = ids[:n_tr], ids[n_tr:]
    print(f"GPT-2 small ({sum(p.numel() for p in model.parameters())/1e6:.0f}M) | layer "
          f"{layer}/{model.config.n_layer} | {tr.shape[0]} train / {va.shape[0]} val seqs (len {seq_len})")

    def logits_last(seqs_batch):
        return model(seqs_batch.to(DEVICE)).logits[:, -1, :]

    # ---- precompute cached MLP activations + full-model target next-token logits ----
    @torch.no_grad()
    def precompute(seqs, want_logits):
        splicer.mode, splicer.tc = "full", None
        mi, mo, lg = [], [], []
        for i in range(0, seqs.shape[0], batch):
            b = seqs[i:i + batch].to(DEVICE)
            out = model(b)
            mi.append(splicer.mlp_in.cpu()); mo.append(splicer.mlp_out.cpu())
            if want_logits:
                lg.append(out.logits[:, -1, :].cpu())
        L = torch.cat(lg) if want_logits else None
        return torch.cat(mi), torch.cat(mo), L

    print("precomputing activations ...")
    mi_tr, mo_tr, tgt_tr = precompute(tr, want_logits=True)
    mi_va, mo_va, _ = precompute(va, want_logits=False)

    def train(candor, seed=0):
        torch.manual_seed(seed)
        tc = Transcoder(d, m, k).to(DEVICE)
        opt = torch.optim.Adam(tc.parameters(), lr=2e-3)
        rng = np.random.default_rng(seed)
        for step in range(steps):
            idx = rng.integers(0, tr.shape[0], size=batch)
            xout = mo_tr[idx].to(DEVICE)
            m_hat, _ = tc(mi_tr[idx].to(DEVICE))
            # relative (normalised) reconstruction -> O(1), so the behavioural terms bite
            recon = (((m_hat - xout) ** 2).sum(-1) / ((xout ** 2).sum(-1) + 1e-6)).mean()
            loss = recon
            if candor:
                tgt = tgt_tr[idx].to(DEVICE)                       # cached full-model logits
                splicer.tc = tc
                splicer.mode = "legible"
                leg = logits_last(tr[idx])                         # leak-ablated splice
                faith = F.kl_div(F.log_softmax(leg, -1), F.softmax(tgt, -1), reduction="batchmean")
                splicer.mode = "leakswap"
                swap = logits_last(tr[idx])
                causal = tv(tgt, swap).mean()
                loss = recon + faith_w * faith + causal_w * causal
                splicer.mode, splicer.tc = "full", None
            opt.zero_grad(); loss.backward(); opt.step(); tc.normalize()
            if step % 100 == 0:
                msg = f"  [{'CANDOR' if candor else 'SAE  '}] step {step:4d} recon={float(recon.detach()):.3f}"
                if candor:
                    msg += f" faith={float(faith.detach()):.4f} causal={float(causal.detach()):.4f}"
                print(msg, flush=True)
        return tc

    @torch.no_grad()
    def ce_loss(seqs, mode, tc=None):
        splicer.mode, splicer.tc = mode, tc
        tot, ntok = 0.0, 0
        for i in range(0, seqs.shape[0], batch):
            b = seqs[i:i + batch].to(DEVICE)
            lo = model(b).logits
            tot += float(F.cross_entropy(lo[:, :-1].reshape(-1, lo.shape[-1]),
                                         b[:, 1:].reshape(-1), reduction="sum"))
            ntok += b[:, 1:].numel()
        splicer.mode, splicer.tc = "full", None
        return tot / ntok

    @torch.no_grad()
    def evaluate(tc, name):
        splicer.tc = tc
        deltas, swaps = [], []
        rn, rd = 0.0, 0.0
        for i in range(0, va.shape[0], batch):
            b = va[i:i + batch].to(DEVICE)
            m_hat, _ = tc(mi_va[i:i + batch].to(DEVICE))
            rn += float(((m_hat - mo_va[i:i + batch].to(DEVICE)) ** 2).sum())
            rd += float((mo_va[i:i + batch].to(DEVICE) ** 2).sum())
            splicer.mode = "full"; full = model(b).logits[:, -1, :]
            splicer.mode = "legible"; leg = model(b).logits[:, -1, :]
            splicer.mode = "leakswap"; sw = model(b).logits[:, -1, :]
            deltas.append(tv(full, leg).cpu()); swaps.append(tv(full, sw).cpu())
        splicer.mode, splicer.tc = "full", None
        l_full = ce_loss(va, "full")
        l_leg = ce_loss(va, "legible", tc)
        l_abl = ce_loss(va, "ablate")
        rec = (l_abl - l_leg) / max(1e-6, (l_abl - l_full))
        r = dict(delta=float(torch.cat(deltas).mean()), leak_swap=float(torch.cat(swaps).mean()),
                 recon_unexplained=rn / rd, loss_full=l_full, loss_legible=l_leg,
                 loss_ablated=l_abl, loss_recovered=rec)
        print(f"  {name}: delta={r['delta']:.4f} swap={r['leak_swap']:.4f} "
              f"recon_unexpl={r['recon_unexplained']:.3f} loss_full={l_full:.3f} "
              f"loss_leg={l_leg:.3f} loss_abl={l_abl:.3f} recovered={rec:.3f}", flush=True)
        return r

    print("training SAE (reconstruction only) ...")
    sae = train(candor=False)
    print("training CANDOR transcoder (behavioural + causal) ...")
    cand = train(candor=True)

    out = {
        "model": "gpt2", "params": int(sum(p.numel() for p in model.parameters())),
        "layer": layer, "n_layer": model.config.n_layer, "d_model": d,
        "seq_len": seq_len, "n_train": int(tr.shape[0]), "n_val": int(va.shape[0]),
        "m": m, "k": k, "steps": steps, "device": DEVICE,
        "sae": evaluate(sae, "SAE"), "candor": evaluate(cand, "CANDOR"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(os.path.join(RESULTS, "gpt2.json"), "w"), indent=2)
    print("wrote results/gpt2.json  ({:.0f}s)".format(out["elapsed_sec"]), flush=True)
    return out


if __name__ == "__main__":
    main()

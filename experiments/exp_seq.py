"""Experiment 2: composition with attention (sequence-max).

The Legible Bottleneck is architecture-agnostic; here I show it composes with
self-attention. The task, predict the maximum token of a length-L sequence,
requires aggregating information *across positions*, the canonical job of
attention, and (unlike modular addition) a one-layer Transformer learns it in a
couple of thousand steps. I place a Legible Bottleneck on the MLP hidden state
and report the interpretability tax (deployed vs. leak-ablated legible model) and
the Faithfulness Certificate.  Results -> results/seq.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import candorkit as ck

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
# This is a tiny model; on CPU it avoids the per-kernel GPU launch overhead and is faster.
DEVICE = "cpu"


def main(seq_len=8, vocab=10, steps=1200, train_frac=0.85):
    t0 = time.time()
    data = ck.sequence_max(n_samples=12000, seq_len=seq_len, vocab=vocab, seed=0)
    tr, va = ck.split(data.X.shape[0], train_frac, seed=0)
    print(f"sequence-max: L={seq_len}, vocab={vocab}, {len(tr)} train / {len(va)} test")

    torch.manual_seed(0)
    model = ck.LegibleTransformer(vocab=data.vocab, seq_len=data.seq_len, d_model=96,
                                  n_heads=4, d_mlp=256, n_layers=1, n_out=data.n_classes,
                                  m=128, k=8)
    w = ck.LossWeights(legible_task=0.5, faith=1.0, leak=0.5, causal=0.3,
                       anchor=1e-2, sparse=1e-3)
    ck.train_candor(model, data.X, data.y, tr,
                    ck.TrainConfig(steps=steps, lr=1e-3, batch=256, weights=w, seed=0),
                    device=DEVICE, verbose=False)
    acc_full = ck.task_accuracy(model, data.X[va], data.y[va], mode="full")
    acc_leg = ck.task_accuracy(model, data.X[va], data.y[va], mode="legible")
    cert = ck.certify(model, data.X[va], data.y[va])

    out = {
        "task": "sequence-max", "seq_len": seq_len, "vocab": data.vocab,
        "n_classes": data.n_classes, "n_train": int(len(tr)), "n_test": int(len(va)),
        "steps": steps, "device": DEVICE,
        "acc_full": acc_full, "acc_legible": acc_leg, "tax": acc_full - acc_leg,
        "delta_mean": cert.delta, "delta_p95": cert.delta_p95,
        "agreement": cert.agreement, "completeness": cert.completeness,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS, "seq.json"), "w") as f:
        json.dump(out, f, indent=2)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    print("wrote results/seq.json")
    return out


if __name__ == "__main__":
    main()

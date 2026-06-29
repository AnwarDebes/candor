"""Experiment 2: modular addition with a Transformer.

Shows the Legible Bottleneck composes with attention on a genuinely non-linear,
algorithmic task (a + b mod p, the classic grokking circuit).  I report the
interpretability tax (a plain Transformer vs CANDOR's leak-ablated replacement
model) and the Faithfulness Certificate.  Results -> results/modadd.json.
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
DEVICE = ck.device_auto()


def make_model(d):
    return ck.LegibleTransformer(vocab=d.vocab, seq_len=d.seq_len, d_model=128,
                                 n_heads=4, d_mlp=512, n_layers=1, n_out=d.p,
                                 m=256, k=16)


def main(p=53, steps=6000, train_frac=0.85):
    t0 = time.time()
    data = ck.modular_addition(p=p, seed=0)
    tr, va = ck.split(data.X.shape[0], train_frac, seed=0)
    print(f"modular addition mod {p}: {data.X.shape[0]} pairs, {len(tr)} train / {len(va)} test")

    # --- opaque reference: same architecture, only the task loss (bottleneck is a
    #     pure identity pass-through, so this is a plain Transformer's capability) ---
    torch.manual_seed(0)
    opaque = make_model(data)
    ck.train_candor(opaque, data.X, data.y, tr,
                    ck.TrainConfig(steps=steps, lr=1e-3, batch=512, weight_decay=0.5,
                                   weights=ck.LossWeights(legible_task=0, faith=0, leak=0,
                                                          causal=0, anchor=0, sparse=0),
                                   seed=0), device=DEVICE, verbose=False)
    acc_opaque = ck.task_accuracy(opaque, data.X[va], data.y[va], mode="full")

    # --- CANDOR: full objective ---
    torch.manual_seed(0)
    model = make_model(data)
    w = ck.LossWeights(legible_task=0.5, faith=1.0, leak=0.5, causal=0.3,
                       anchor=1e-2, sparse=1e-3)
    ck.train_candor(model, data.X, data.y, tr,
                    ck.TrainConfig(steps=steps, lr=1e-3, batch=512, weight_decay=0.5,
                                   weights=w, seed=0), device=DEVICE, verbose=False)
    acc_full = ck.task_accuracy(model, data.X[va], data.y[va], mode="full")
    acc_leg = ck.task_accuracy(model, data.X[va], data.y[va], mode="legible")
    cert = ck.certify(model, data.X[va], data.y[va])

    out = {
        "p": p, "vocab": data.vocab, "n_pairs": int(data.X.shape[0]),
        "n_train": int(len(tr)), "n_test": int(len(va)), "steps": steps,
        "device": DEVICE,
        "acc_opaque": acc_opaque,
        "acc_full": acc_full,
        "acc_legible": acc_leg,
        "tax": acc_opaque - acc_leg,
        "delta_mean": cert.delta, "delta_p95": cert.delta_p95,
        "agreement": cert.agreement, "completeness": cert.completeness,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS, "modadd.json"), "w") as f:
        json.dump(out, f, indent=2)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    print("wrote results/modadd.json")
    return out


if __name__ == "__main__":
    main()

"""The interpretability-tax / legibility frontier.

I sweep the capacity of the legible channel (the TopK budget k, i.e. how many
named concepts may be active at once) and trace capability (legible-model
accuracy) against the Faithfulness Certificate delta and concept recovery.  This
makes the central practical claim concrete and honest: there is a frontier, and
once the channel has enough capacity to carry the task's concepts the tax is
small while delta is tiny.  Results -> results/tax.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import candorkit as ck
from candorkit.metrics import concept_recovery

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
DEVICE = ck.device_auto()


def main(ks=(1, 2, 3, 4, 6, 8, 12), steps=1600):
    t0 = time.time()
    data = ck.planted_concepts(n_samples=20000, n_in=48, K=12, n_relevant=5,
                               density=0.3, noise=0.1, seed=0, task="sum")
    tr, va = ck.split(data.X.shape[0], 0.8, seed=0)
    rows = []
    for k in ks:
        torch.manual_seed(0)
        model = ck.LegibleMLP(d_in=data.n_in, d_h=128, n_out=data.n_classes,
                              m=48, k=k, n_hidden=2)
        ck.train_candor(model, data.X, data.y, tr,
                        ck.TrainConfig(steps=steps, lr=2e-3, batch=512,
                                       weights=ck.LossWeights(), seed=0),
                        device=DEVICE, verbose=False)
        accf = ck.task_accuracy(model, data.X[va], data.y[va], mode="full")
        accl = ck.task_accuracy(model, data.X[va], data.y[va], mode="legible")
        cert = ck.certify(model, data.X[va], data.y[va])
        codes = ck.concept_activations(model, data.X[va])
        rec = concept_recovery(codes, data.concepts[va].numpy())
        row = {"k": k, "acc_full": accf, "acc_legible": accl,
               "delta_mean": cert.delta, "recovery": rec["recovery"]}
        rows.append(row)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()})
    out = {"sweep": rows, "K_true_density": 0.3, "n_relevant": int(len(data.relevant)),
           "elapsed_sec": round(time.time() - t0, 1)}
    with open(os.path.join(RESULTS, "tax.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/tax.json")


if __name__ == "__main__":
    main()

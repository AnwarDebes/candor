"""CANDOR in 40 lines: train a glass-box model, certify a decision, read its
named-concept explanation, and catch a channel-bypassing backdoor.

    python demo/quickstart.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import candorkit as ck

torch.manual_seed(0)

# 1. A task with KNOWN ground-truth concepts and mechanism.
data = ck.planted_concepts(n_samples=12000, n_in=32, K=8, n_relevant=4,
                           density=0.3, noise=0.1, seed=0, task="sum")
tr, va = ck.split(data.X.shape[0], 0.8, seed=0)

# 2. A CANDOR model: computation routed through a Legible Bottleneck.
model = ck.LegibleMLP(d_in=data.n_in, d_h=96, n_out=data.n_classes, m=32, k=6)
ck.train_candor(model, data.X, data.y, tr,
                ck.TrainConfig(steps=1200, lr=2e-3, batch=512), device="cpu",
                verbose=False)

# 3. The Faithfulness Certificate: a sound, re-checkable bound on dark computation.
cert = ck.certify(model, data.X[va], data.y[va])
acc_leg = ck.task_accuracy(model, data.X[va], data.y[va], mode="legible")
codes = ck.concept_activations(model, data.X[va])
rec = ck.concept_recovery(codes, data.concepts[va].numpy())

print("== CANDOR glass-box model ==")
print(f"legible-model accuracy : {acc_leg:.3f}  (the explanation alone solves the task)")
print(f"certificate delta      : {cert.delta:.4f}  (>= behavioural divergence, by construction)")
print(f"model/explanation agree: {cert.agreement:.3f}")
print(f"concept recovery        : {rec['recovery']:.3f}  (vs the planted ground truth)")

# 4. The explanation of one decision = its active named concepts.
print("\n== explanation of input #0 ==")
for site in cert.active_concepts[0]:
    print("  active concepts:", site[:6])

# 5. A backdoor that bypasses the channel cannot stay hidden: delta spikes.
print("\n== dark-computation detection ==")
from sklearn.metrics import roc_auc_score
td = ck.trojan_concepts(n_samples=12000, n_in=32, K=8, n_relevant=3,
                        trojan_rate=0.05, seed=0)
ttr, tva = ck.split(td.X.shape[0], 0.8, seed=0)
tm = ck.LegibleMLP(d_in=td.n_in, d_h=96, n_out=td.n_classes, m=32, k=6)
ck.train_candor(tm, td.X, td.y, ttr, ck.TrainConfig(steps=1200, batch=512),
                device="cpu", verbose=False)
ct = ck.certify(tm, td.X[tva], td.y[tva])
is_t = td.is_trojan[tva].numpy()
auc = roc_auc_score(is_t, ct.per_example_delta.numpy())
print(f"delta detects the backdoor with AUC = {auc:.3f}")

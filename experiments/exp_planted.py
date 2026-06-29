"""Experiment 1: planted concepts with known ground truth.

This is the headline experiment: a synthetic task where the true concepts are known
and the true mechanism, so every interpretability claim is *checked* rather than
asserted.  It reports, in one run:

  * the interpretability tax        (CANDOR legible vs full vs an opaque baseline)
  * the Faithfulness Certificate    (delta, soundness vs behavioural divergence)
  * concept recovery                (vs ground truth, and vs a post-hoc SAE)
  * causal necessity & directional  (ablating a named concept moves the decision
                                     exactly as the ground-truth mechanism says)
  * the causal-sufficiency gap      (CANDOR's leak is inert; an SAE residual is not)
  * objective ablations             (drop each loss term -> the channel breaks)
  * anchoring / concept stability   (shared identity across training runs)
  * dark-computation detection      (a trojan that bypasses the channel -> delta spikes)

Results are written to results/planted.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import candorkit as ck
from candorkit.metrics import concept_recovery, stability

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)
DEVICE = ck.device_auto()


def tv(a, b):
    return 0.5 * (F.softmax(a, -1) - F.softmax(b, -1)).abs().sum(-1)


def build(data, m=48, k=8, d_h=128, init_seed=None):
    if init_seed is not None:
        torch.manual_seed(init_seed)
    return ck.LegibleMLP(d_in=data.n_in, d_h=d_h, n_out=data.n_classes, m=m, k=k,
                         n_hidden=2)


def directional_causal(model, data, idx, rec):
    """Ground-truth causal test (held out from training): ablate the atom matched
    to a relevant concept and check the legible decision moves by exactly -1 (sum
    task); ablate an irrelevant concept's atom and check the decision is unchanged."""
    model.eval()
    X = data.X[idx].to(DEVICE)
    g = data.concepts[idx].numpy()
    concept_atom = {int(kk): int(a) for a, kk in rec["matching"]}
    with torch.no_grad():
        base = model(X, mode="legible").argmax(-1).cpu().numpy()
        code0 = model.sites[-1].code.clone()
    relset = set(int(r) for r in data.relevant)

    def ablate_shift(concept_k, want):
        a = concept_atom.get(concept_k)
        if a is None:
            return None
        present = np.where(g[:, concept_k] > 0.5)[0]
        if len(present) == 0:
            return None
        code = code0.clone()
        code[present, a] = 0.0
        with torch.no_grad():
            newp = model.forward_from_code(code).argmax(-1).cpu().numpy()
        shift = newp[present] - base[present]
        return float((shift == want).mean())

    rel = [ablate_shift(k, -1) for k in relset]
    irr = [ablate_shift(k, 0) for k in range(data.K) if k not in relset]
    rel = [v for v in rel if v is not None]
    irr = [v for v in irr if v is not None]
    return {"directional_relevant": float(np.mean(rel)) if rel else 0.0,
            "directional_irrelevant_unchanged": float(np.mean(irr)) if irr else 0.0}


def eval_model(model, data, va, label=""):
    Xv, yv = data.X[va], data.y[va]
    acc_full = ck.task_accuracy(model, Xv, yv, mode="full")
    acc_leg = ck.task_accuracy(model, Xv, yv, mode="legible")
    cert = ck.certify(model, Xv, yv)
    codes = ck.concept_activations(model, Xv)
    rec = concept_recovery(codes, data.concepts[va].numpy())
    nec = ck.causal_faithfulness(model, Xv, data.concepts[va].numpy(),
                                 data.relevant, n_eval=min(1500, len(va)))
    dirc = directional_causal(model, data, va[:1500], rec)
    # causal sufficiency: swapping the leak across inputs must not move the output
    Xd = Xv[:2000].to(DEVICE)
    with torch.no_grad():
        full = model(Xd, mode="full")
        swap = model(Xd, mode="full", leak_swap=True)
    leak_swap_effect = float(tv(full, swap).mean())
    return {
        "label": label,
        "acc_full": acc_full, "acc_legible": acc_leg,
        "delta_mean": cert.delta, "delta_p95": cert.delta_p95,
        "agreement": cert.agreement, "completeness": cert.completeness,
        "recovery": rec["recovery"], "recovery_min": rec["recovery_min"],
        "mono_margin": rec["mono_margin"],
        "necessity_gap": nec["necessity_gap"],
        "necessity_relevant": nec["necessity_relevant"],
        "necessity_irrelevant": nec["necessity_irrelevant"],
        "leak_swap_effect": leak_swap_effect,
        **dirc,
    }


def main():
    t0 = time.time()
    out = {"device": DEVICE, "config": {}}
    STEPS = 2200

    # ---------------------------------------------------------------- main run
    data = ck.planted_concepts(n_samples=20000, n_in=48, K=12, n_relevant=5,
                               density=0.3, noise=0.1, seed=0, task="sum")
    tr, va = ck.split(data.X.shape[0], 0.8, seed=0)
    out["config"] = dict(n_in=data.n_in, K=data.K, n_relevant=int(len(data.relevant)),
                         n_classes=data.n_classes, m=48, k=8, steps=STEPS)

    print("== main CANDOR ==")
    torch.manual_seed(0)
    model = build(data, init_seed=0)
    w = ck.LossWeights(legible_task=0.5, faith=1.0, leak=1.0, causal=0.5,
                       anchor=1e-2, sparse=2e-3)
    ck.train_candor(model, data.X, data.y, tr,
                    ck.TrainConfig(steps=STEPS, lr=2e-3, batch=512, weights=w, seed=0),
                    device=DEVICE, verbose=False)
    out["main"] = eval_model(model, data, va, "CANDOR")
    print({k: round(v, 4) for k, v in out["main"].items() if isinstance(v, float)})

    # ------------------------------------------------ opaque baseline + post-hoc SAE
    print("== opaque baseline + post-hoc SAE ==")
    torch.manual_seed(0)
    opaque = ck.OpaqueMLP(d_in=data.n_in, d_h=128, n_out=data.n_classes, n_hidden=2)
    ck.train_opaque(opaque, data.X, data.y, tr, steps=STEPS, lr=2e-3, batch=512,
                    device=DEVICE, seed=0)
    acc_op = _acc_opaque(opaque, data.X[va], data.y[va])
    with torch.no_grad():
        e_tr = opaque.embedding(data.X[tr].to(DEVICE)).cpu()
        e_va = opaque.embedding(data.X[va].to(DEVICE)).cpu()
    sae = ck.fit_sae(e_tr, m=48, k=8, steps=2000, device=DEVICE, seed=0)
    sae_c = ck.sae_codes(sae, e_va)
    rec_sae = concept_recovery(sae_c, data.concepts[va].numpy())
    # SAE substitution + residual-swap causal effect (the SAE is NOT causally faithful)
    with torch.no_grad():
        ev = e_va.to(DEVICE)
        out_full = opaque.head_forward(ev)
        recon = sae(ev).recon
        out_sub = opaque.head_forward(recon)
        resid = ev - recon
        perm = torch.randperm(ev.shape[0], device=DEVICE)
        out_swap = opaque.head_forward(recon + resid[perm])
    out["opaque_sae"] = {
        "acc_opaque": acc_op,
        "sae_recovery": rec_sae["recovery"], "sae_recovery_min": rec_sae["recovery_min"],
        "sae_substitution_delta": float(tv(out_full, out_sub).mean()),
        "sae_residual_swap_effect": float(tv(out_full, out_swap).mean()),
    }
    out["tax"] = {"opaque_acc": acc_op,
                  "candor_legible_acc": out["main"]["acc_legible"],
                  "tax": acc_op - out["main"]["acc_legible"]}
    print(out["opaque_sae"]); print("tax:", out["tax"])

    # ------------------------------------------------------------- ablations
    print("== ablations ==")
    ablations = {
        "no_faith": ck.LossWeights(faith=0.0, leak=1.0, causal=0.5, anchor=1e-2),
        "no_leak": ck.LossWeights(faith=1.0, leak=0.0, causal=0.5, anchor=1e-2),
        "no_causal": ck.LossWeights(faith=1.0, leak=1.0, causal=0.0, anchor=1e-2),
        "no_anchor": ck.LossWeights(faith=1.0, leak=1.0, causal=0.5, anchor=0.0),
    }
    out["ablations"] = {}
    for name, wa in ablations.items():
        torch.manual_seed(0)
        ma = build(data, init_seed=0)
        ck.train_candor(ma, data.X, data.y, tr,
                        ck.TrainConfig(steps=STEPS, lr=2e-3, batch=512, weights=wa, seed=0),
                        device=DEVICE, verbose=False)
        e = eval_model(ma, data, va, name)
        out["ablations"][name] = {k: e[k] for k in
                                  ["acc_legible", "delta_mean", "recovery",
                                   "necessity_gap", "leak_swap_effect"]}
        print(name, {k: round(v, 4) for k, v in out["ablations"][name].items()})

    # ----------------------------------------------------------- anchoring/stability
    print("== anchoring / stability ==")
    seeds = [11, 22, 33]
    out["stability"] = {}
    for cond, anchor_w, shared_init in [("anchored", 1e-2, True), ("free", 0.0, False)]:
        codes_list = []
        for s in seeds:
            init = 4242 if shared_init else s
            torch.manual_seed(init)
            ms = build(data, init_seed=init)
            ck.train_candor(ms, data.X, data.y, tr,
                            ck.TrainConfig(steps=STEPS, lr=2e-3, batch=512,
                                           weights=ck.LossWeights(anchor=anchor_w), seed=s),
                            device=DEVICE, verbose=False)
            codes_list.append(ck.concept_activations(ms, data.X[va]))
        gt = data.concepts[va].numpy()
        pair = [stability(codes_list[i], codes_list[j], gt)
                for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
        out["stability"][cond] = float(np.mean(pair))
        print(cond, "mean pairwise stability:", round(out["stability"][cond], 3))

    # ------------------------------------------------ trojan / dark-computation
    print("== trojan / dark-computation detection ==")
    from sklearn.metrics import roc_auc_score
    tdata = ck.trojan_concepts(n_samples=20000, n_in=48, K=12, n_relevant=3,
                               density=0.3, noise=0.1, trojan_rate=0.05, seed=0)
    ttr, tva = ck.split(tdata.X.shape[0], 0.8, seed=0)
    torch.manual_seed(0)
    tmodel = ck.LegibleMLP(d_in=tdata.n_in, d_h=128, n_out=tdata.n_classes, m=48, k=8,
                           n_hidden=2)
    ck.train_candor(tmodel, tdata.X, tdata.y, ttr,
                    ck.TrainConfig(steps=STEPS, lr=2e-3, batch=512,
                                   weights=ck.LossWeights(), seed=0),
                    device=DEVICE, verbose=False)
    cert_t = ck.certify(tmodel, tdata.X[tva], tdata.y[tva])
    is_troj = tdata.is_trojan[tva].numpy().astype(int)
    delta_per = cert_t.per_example_delta.numpy()
    auc = float(roc_auc_score(is_troj, delta_per)) if is_troj.sum() > 0 else float("nan")
    acc_troj = ck.task_accuracy(tmodel, tdata.X[tva], tdata.y[tva], mode="full")
    out["trojan"] = {
        "auc_delta_detects_trojan": auc,
        "delta_clean_mean": float(delta_per[is_troj == 0].mean()),
        "delta_trojan_mean": float(delta_per[is_troj == 1].mean()),
        "acc_full": acc_troj, "trojan_rate": float(is_troj.mean()),
        "delta_clean_sample": delta_per[is_troj == 0][:3000].tolist(),
        "delta_trojan_sample": delta_per[is_troj == 1].tolist(),
    }
    print(out["trojan"])

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "planted.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote results/planted.json  ({out['elapsed_sec']}s)")


@torch.no_grad()
def _acc_opaque(model, X, y, bs=4096):
    model.eval()
    c = 0
    for i in range(0, X.shape[0], bs):
        c += (model(X[i:i + bs].to(DEVICE)).argmax(-1).cpu() == y[i:i + bs]).sum().item()
    return c / X.shape[0]


if __name__ == "__main__":
    main()

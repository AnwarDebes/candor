"""Interpretability metrics with ground-truth answer keys.

These are the measurements the paper reports.  All of them are checkable because
the planted-concepts benchmark exposes the true concepts and the true mechanism.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _dev(model):
    return next(model.parameters()).device


@torch.no_grad()
def concept_activations(model, x, site: int = -1, batch_size: int = 4096):
    """Return the (N, m) concept-code matrix at one bottleneck site."""
    model.eval()
    dev = _dev(model)
    chunks = []
    for i in range(0, x.shape[0], batch_size):
        model(x[i:i + batch_size].to(dev), mode="full")
        chunks.append(model.sites[site].code.cpu())
    return torch.cat(chunks).numpy()


def _corr(A, B):
    """|Pearson correlation| matrix between columns of A (n,a) and B (n,b)."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-8)
    return np.abs(A.T @ B)                      # (a, b)


def concept_recovery(codes, true_concepts):
    """Hungarian-match learned atoms to ground-truth concepts; report the mean
    matched |correlation| (recovery) and a monosemanticity/purity score.

    Returns dict with: recovery (mean matched corr), purity (mean over atoms of
    top-1 corr / total corr mass), matching (atom->concept), per_concept (corr).
    """
    g = np.asarray(true_concepts, dtype=np.float64)
    c = np.asarray(codes, dtype=np.float64)
    M = _corr(c, g)                              # (m_atoms, K)
    # match each ground-truth concept to its best atom (assignment maximises corr)
    rows, cols = linear_sum_assignment(-M)       # minimise -corr == maximise corr
    matched = M[rows, cols]
    per_concept = {}
    for r, k in zip(rows, cols):
        per_concept[int(k)] = float(M[r, k])
    # purity: among atoms that ever fire, how monosemantic (one dominant concept)?
    mass = M.sum(1) + 1e-8
    active = M.max(1) > 1e-3
    purity = float((M.max(1)[active] / mass[active]).mean()) if active.any() else 0.0
    # monosemanticity margin: for each matched atom, gap between its best and
    # second-best concept correlation (high => the atom tracks a single concept)
    margins = []
    for r, k in zip(rows, cols):
        row = np.sort(M[r])[::-1]
        margins.append(float(row[0] - (row[1] if len(row) > 1 else 0.0)))
    return {
        "recovery": float(matched.mean()),
        "recovery_min": float(matched.min()),
        "purity": purity,
        "mono_margin": float(np.mean(margins)),
        "per_concept": per_concept,
        "matching": list(zip(rows.tolist(), cols.tolist())),
    }


@torch.no_grad()
def task_accuracy(model, x, y, mode="full", batch_size=4096):
    model.eval()
    dev = _dev(model)
    correct = 0
    for i in range(0, x.shape[0], batch_size):
        logits = model(x[i:i + batch_size].to(dev), mode=mode)
        correct += (logits.argmax(-1).cpu() == y[i:i + batch_size]).sum().item()
    return correct / x.shape[0]


@torch.no_grad()
def causal_faithfulness(model, x, true_concepts, relevant, codes=None,
                        site: int = -1, n_eval: int = 1000):
    """Does ablating the concept that *encodes a label-relevant feature* change
    the decision more than ablating an irrelevant one?

    I map each relevant ground-truth concept to its best-matched atom, ablate
    that atom (zero its code), and measure how often the legible decision flips,
    versus ablating a random label-irrelevant atom.  A faithful, causal channel
    has a large necessity gap (relevant >> irrelevant).
    """
    model.eval()
    dev = _dev(model)
    if codes is None:
        codes = concept_activations(model, x[:n_eval], site=site)
    rec = concept_recovery(codes, np.asarray(true_concepts[:n_eval]))
    atom_of = {k: r for r, k in rec["matching"]}
    rel_atoms = [atom_of[int(k)] for k in relevant if int(k) in atom_of]
    all_atoms = set(range(codes.shape[1]))
    irrel_atoms = list(all_atoms - set(atom_of[int(k)] for k in relevant if int(k) in atom_of))

    xb = x[:n_eval].to(dev)
    base = model(xb, mode="legible")   # ablate *within* the legible channel
    base_pred = base.argmax(-1)

    def flip_rate_after_ablating(atoms):
        if not atoms:
            return 0.0
        # zero those atoms in the concept code, then re-run the head legibly
        model(xb, mode="legible")
        code = model.sites[site].code.clone()
        code[:, atoms] = 0.0
        logits = model.forward_from_code(code)
        return (logits.argmax(-1) != base_pred).float().mean().item()

    rel_flip = flip_rate_after_ablating(rel_atoms)
    rng = np.random.default_rng(0)
    irr_sample = list(rng.choice(irrel_atoms, size=min(len(rel_atoms) or 1, len(irrel_atoms)),
                                 replace=False)) if irrel_atoms else []
    irr_flip = flip_rate_after_ablating(irr_sample)
    return {"necessity_relevant": rel_flip, "necessity_irrelevant": irr_flip,
            "necessity_gap": rel_flip - irr_flip}


# --- helpers for ablation re-runs (single-site MLP) ---
def model_decode(model, site, code):
    bn = _site_bottleneck(model, site)
    return bn.decode(code)


@torch.no_grad()
def rerun_from_site(model, x, site, recon):
    """Re-run an MLP from a bottleneck site using a supplied (ablated) recon.
    Implemented for the single-bottleneck MLP used in the planted experiment."""
    import torch.nn.functional as F
    if hasattr(model, "out") and hasattr(model, "in_proj"):
        return model.out(recon)
    raise NotImplementedError("rerun_from_site supports the 1-hidden LegibleMLP")


def _site_bottleneck(model, site):
    if hasattr(model, "bottlenecks"):
        return model.bottlenecks[site]
    return model.blocks[site].bottleneck


def stability(codes_a, codes_b, true_concepts):
    """Fraction of ground-truth concepts assigned to a *consistent* atom across
    two independently trained models (anchoring should raise this)."""
    ra = concept_recovery(codes_a, true_concepts)["matching"]
    rb = concept_recovery(codes_b, true_concepts)["matching"]
    map_a = {k: r for r, k in ra}
    map_b = {k: r for r, k in rb}
    keys = set(map_a) & set(map_b)
    if not keys:
        return 0.0
    return float(np.mean([map_a[k] == map_b[k] for k in keys]))

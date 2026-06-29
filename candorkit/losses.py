"""The CANDOR training objective.

The total loss is

    L = L_task(full)                      # be accurate
      + a * L_task(legible)               # the legible model is accurate on its own
      + b * L_faith                       # legible model reproduces the full model
      + lam * L_leak                      # shrink the leak (dark computation) energy
      + mu * L_causal                     # leaks are causally inert (concepts suffice)
      + gam * L_anchor                    # concept identities stay put
      + l1 * L_sparse                     # mild extra shrinkage of concept codes

``L_faith`` and ``L_causal`` are the load-bearing, *causal* ingredients that
separate CANDOR from a post-hoc sparse dictionary:

  * ``L_faith`` = KL( full || legible ) trains the leak-ablated replacement model
    to behave like the deployed model, so the named-concept explanation *is*
    what the model computes.
  * ``L_causal`` = divergence between the full model and the model with its leaks
    swapped across the batch.  Driving it to zero means the non-legible component
    has no causal effect on the output: the concepts are causally *sufficient*
    (a differentiable analogue of causal scrubbing, used here as a training
    signal rather than only as a post-hoc test).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    legible_task: float = 0.5   # a
    faith: float = 1.0          # b
    leak: float = 0.3           # lam
    causal: float = 0.3         # mu
    anchor: float = 1e-2        # gam
    sparse: float = 1e-3        # l1


@dataclass
class LossBreakdown:
    total: torch.Tensor
    parts: dict = field(default_factory=dict)


def _tv(p_logits, q_logits) -> torch.Tensor:
    """Total-variation distance between two categorical distributions (per row)."""
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return 0.5 * (p - q).abs().sum(-1)


def leak_energy(sites) -> torch.Tensor:
    """Mean over sites of the relative leak energy ||r||^2 / (||h||^2 + eps)."""
    terms = []
    for s in sites:
        h = s.recon + s.leak                       # == h
        num = (s.leak ** 2).sum(-1)
        den = (h ** 2).sum(-1) + 1e-6
        terms.append((num / den).mean())
    return torch.stack(terms).mean()


def code_l1(sites) -> torch.Tensor:
    return torch.stack([s.code.abs().sum(-1).mean() for s in sites]).mean()


def anchor_drift(model) -> torch.Tensor:
    """Penalise drift of decoder atoms from their persistent (EMA) identity."""
    terms = []
    for bn in _bottlenecks(model):
        terms.append(((bn.W_dec - bn.ema_dec.detach()) ** 2).sum(0).mean())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).mean()


def _bottlenecks(model):
    bns = []
    if hasattr(model, "bottlenecks"):
        bns += list(model.bottlenecks)
    if hasattr(model, "blocks"):
        for blk in model.blocks:
            if hasattr(blk, "bottleneck"):
                bns.append(blk.bottleneck)
    return bns


def candor_loss(model, x, y, w: LossWeights,
                task_loss=F.cross_entropy) -> LossBreakdown:
    """Compute the full CANDOR objective.  Runs three forward passes:
    full (deployed), legible (leak ablated), and leak-permuted (causal probe)."""
    logits_full = model(x, mode="full")
    sites_full = list(model.sites)

    logits_leg = model(x, mode="legible")
    logits_swap = model(x, mode="full", leak_swap=True)

    l_task = task_loss(logits_full, y)
    l_task_leg = task_loss(logits_leg, y)
    # KL(full || legible): train the replacement model to match the deployed model.
    l_faith = F.kl_div(
        F.log_softmax(logits_leg, dim=-1),
        F.softmax(logits_full.detach(), dim=-1),
        reduction="batchmean",
    )
    # Causal: swapping the (non-legible) leaks must not move the output.
    l_causal = _tv(logits_full.detach(), logits_swap).mean()
    l_leak = leak_energy(sites_full)
    l_anchor = anchor_drift(model)
    l_sparse = code_l1(sites_full)

    total = (l_task
             + w.legible_task * l_task_leg
             + w.faith * l_faith
             + w.leak * l_leak
             + w.causal * l_causal
             + w.anchor * l_anchor
             + w.sparse * l_sparse)

    return LossBreakdown(total=total, parts={
        "task": float(l_task.detach()), "task_leg": float(l_task_leg.detach()),
        "faith": float(l_faith.detach()), "leak": float(l_leak.detach()),
        "causal": float(l_causal.detach()), "anchor": float(l_anchor.detach()),
        "sparse": float(l_sparse.detach()),
    })

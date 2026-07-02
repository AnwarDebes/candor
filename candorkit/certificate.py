"""The Faithfulness Certificate: CANDOR's runtime guarantee.

For every input ``x`` CANDOR runs two forwards: the deployed model ``M`` (full) and
its leak-ablated *replacement* ``M_leg`` (legible), whose output is a pure
function of the named concept codes.  The certificate is the per-input
total-variation distance between their output distributions,

    delta(x) = TV( M(x), M_leg(x) )  in [0, 1].

Proposition (soundness, by construction).  Because ``M_leg``'s output depends on
``x`` *only* through the named concept codes, ``delta(x)`` is an exact, per-input
upper bound on the influence of all non-legible ("dark") computation on the
output distribution: for every event over outputs, the deployed model and the
named-concept explanation assign probabilities within ``delta(x)`` of each other.
(Note this bounds probabilities of the *sampled* output; it does not by itself
bound top-1/argmax disagreement, which is why ``agreement`` is reported
separately as a measured quantity.)  The bound is a *measurement*, not an
estimate, and is re-checkable by anyone by re-running ``M_leg``, which is what
makes it a certificate rather than a heuristic score.

Training drives ``delta`` small (via the faithfulness, leak and causal terms);
the certificate reports, per input, how small it actually is.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Certificate:
    delta: float                  # mean TV(M, M_leg), the dark-computation bound
    delta_p95: float              # 95th percentile (worst-case-ish per input)
    agreement: float              # P(top-1 decision of M == that of M_leg)
    completeness: float | None    # legible-model task accuracy, if labels given
    per_example_delta: torch.Tensor
    active_concepts: list         # names of active concepts per example (the explanation)


@torch.no_grad()
def _tv(p_logits, q_logits):
    p, q = F.softmax(p_logits, -1), F.softmax(q_logits, -1)
    return 0.5 * (p - q).abs().sum(-1)


@torch.no_grad()
def certify(model, x, y=None, batch_size: int = 4096,
            name_prefix: str = "c") -> Certificate:
    """Compute the Faithfulness Certificate over a set of inputs."""
    model.eval()
    dev = next(model.parameters()).device
    deltas, agree, leg_correct, total = [], 0, 0, 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(dev)
        full = model(xb, mode="full")
        leg = model(xb, mode="legible")
        d = _tv(full, leg).cpu()
        deltas.append(d)
        agree += (full.argmax(-1) == leg.argmax(-1)).sum().item()
        total += xb.shape[0]
        if y is not None:
            yb = y[i:i + batch_size].to(dev)
            leg_correct += (leg.argmax(-1) == yb).sum().item()
    per = torch.cat(deltas)

    # Build the human-readable explanation for the first handful of inputs.
    explanations = []
    xb = x[:min(8, x.shape[0])].to(dev)
    model(xb, mode="full")
    for j in range(xb.shape[0]):
        active = []
        for s_idx, site in enumerate(model.sites):
            code = site.code[j]
            idx = torch.nonzero(code > 0).flatten().tolist()
            active.append([(f"L{s_idx}.{name_prefix}{a}", round(float(code[a]), 3))
                           for a in idx])
        explanations.append(active)

    return Certificate(
        delta=float(per.mean()),
        delta_p95=float(per.quantile(0.95)),
        agreement=agree / max(1, total),
        completeness=(leg_correct / max(1, total)) if y is not None else None,
        per_example_delta=per.cpu(),
        active_concepts=explanations,
    )

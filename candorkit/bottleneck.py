"""The Legible Bottleneck: CANDOR's hero primitive.

A Legible Bottleneck is inserted at a *checkpoint* in a network (typically just
after a nonlinearity).  It reads the hidden activation ``h`` through a sparse,
non-negative, *named* concept code and reconstructs it:

    c      = TopK(ReLU(W_enc (h - b_dec) + b_enc))     # sparse concept code  (B, m)
    h_hat  = W_dec c + b_dec                            # legible reconstruction (B, d)
    r      = h - h_hat                                  # the *leak* / dark computation

``c`` is the (typed, persistently-named) explanation; ``h_hat`` is the part of
the computation that the explanation accounts for; ``r`` is everything the named
concepts do **not** capture.  The downstream network is then *routed* in one of
three ways:

    mode='full'     routed = h                  (identity pass-through; the
                                                 bottleneck is an auxiliary read
                                                 + a regulariser, so capability is
                                                 never destroyed)
    mode='legible'  routed = h_hat              (leak ablated -> the *replacement
                                                 model* whose behaviour is a pure
                                                 function of named concepts)
    leak_perm=...   routed = h_hat + r[perm]    (leaks swapped across the batch;
                                                 used by the causal-faithfulness
                                                 objective to test that the leak
                                                 is causally inert)

This is the analogue of self-attention in the Transformer: a single, simple,
composable operation that the whole architecture is built from.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BottleneckOutput:
    code: torch.Tensor   # (B, m)  sparse non-negative concept code c
    recon: torch.Tensor  # (B, d)  legible reconstruction h_hat
    leak: torch.Tensor   # (B, d)  residual r = h - h_hat (dark computation at this site)
    pre: torch.Tensor    # (B, m)  ReLU pre-activations (dense, pre-TopK)


def topk_mask(z: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep the top-``k`` entries per row (by value), zero the rest.

    ``z`` is assumed already non-negative (post-ReLU), so largest value == largest
    magnitude.  This realises an exact L0 = k sparsity, the cleanest sparsity knob
    (cf. TopK SAEs, Gao et al. 2024).
    """
    if k is None or k >= z.shape[-1]:
        return z
    vals, idx = torch.topk(z, k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, idx, vals)
    return out


class LegibleBottleneck(nn.Module):
    """A sparse, typed, persistently-named concept channel at one checkpoint."""

    def __init__(self, d: int, m: int, k: int, tie_init: bool = True):
        super().__init__()
        self.d, self.m, self.k = d, m, k
        self.W_enc = nn.Parameter(torch.empty(m, d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(torch.empty(d, m))
        self.b_dec = nn.Parameter(torch.zeros(d))

        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.copy_(F.normalize(self.W_dec, dim=0))  # unit-norm atoms (identifiability)
            if tie_init:
                self.W_enc.copy_(self.W_dec.t())
            else:
                nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)

        # --- anchoring: a persistent registry that gives each atom a stable identity ---
        self.register_buffer("ema_dec", self.W_dec.detach().clone())
        # --- bookkeeping for dead-atom / type diagnostics ---
        self.register_buffer("act_count", torch.zeros(m))
        self.register_buffer("seen", torch.zeros(1))
        self.register_buffer("type_id", torch.full((m,), -1, dtype=torch.long))

    # ------------------------------------------------------------------ core ops
    def encode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = F.relu(F.linear(h - self.b_dec, self.W_enc, self.b_enc))
        c = topk_mask(pre, self.k)
        return c, pre

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        return F.linear(c, self.W_dec) + self.b_dec

    def forward(self, h: torch.Tensor) -> BottleneckOutput:
        c, pre = self.encode(h)
        h_hat = self.decode(c)
        r = h - h_hat
        if self.training:
            with torch.no_grad():
                self.act_count += (c > 0).float().sum(0)
                self.seen += h.shape[0]
        return BottleneckOutput(code=c, recon=h_hat, leak=r, pre=pre)

    # --------------------------------------------------------------- maintenance
    @torch.no_grad()
    def normalize_decoder(self):
        """Project decoder atoms back to the unit sphere (run each step)."""
        self.W_dec.copy_(F.normalize(self.W_dec, dim=0))

    @torch.no_grad()
    def update_anchor(self, momentum: float = 0.99):
        """EMA update of the persistent concept registry (anchoring target)."""
        self.ema_dec.mul_(momentum).add_(self.W_dec.detach(), alpha=1.0 - momentum)

    @torch.no_grad()
    def dead_fraction(self) -> float:
        if float(self.seen) == 0:
            return 0.0
        rate = self.act_count / float(self.seen)
        return float((rate == 0).float().mean())

    @torch.no_grad()
    def reset_stats(self):
        self.act_count.zero_()
        self.seen.zero_()

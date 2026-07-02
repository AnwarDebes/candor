"""CANDOR models: an MLP and a tiny Transformer, both *routed* through
Legible Bottlenecks.

Every model exposes the same routing contract via ``forward(x, mode, leak_perm)``:

  * ``mode='full'``     : identity pass-through; the deployed model.  The
                           bottlenecks are an auxiliary read + a regulariser and
                           never alter the forward computation, so capability is
                           preserved by construction.
  * ``mode='legible'``  : every leak ablated; this is the *replacement model*
                           whose output is a pure function of the named concepts.
  * ``leak_perm`` (LongTensor of batch indices): at every site the leak is
                           replaced by another example's leak.  If the named
                           concepts causally suffice, the output is unchanged;
                           the causal-faithfulness objective trains this to hold.

After a forward, ``model.sites`` holds the list of ``BottleneckOutput`` for every
checkpoint, so losses and the certificate can inspect concepts, reconstructions
and leaks without a second pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottleneck import BottleneckOutput, LegibleBottleneck


def _route(bn: LegibleBottleneck, h: torch.Tensor, mode: str,
           leak_swap: bool = False) -> tuple[BottleneckOutput, torch.Tensor]:
    bo = bn(h)
    if mode == "legible":
        routed = bo.recon
    elif leak_swap:
        # swap leaks across rows (per-site permutation handles any row count,
        # e.g. (B,) for an MLP or (B*T,) for a Transformer)
        perm = torch.randperm(bo.leak.shape[0], device=h.device)
        routed = bo.recon + bo.leak[perm]
    else:  # full: identity pass-through
        routed = h
    return bo, routed


class LegibleMLP(nn.Module):
    """An MLP whose *concept layer* is a Legible Bottleneck placed where the
    generative factors are linearly present, on a linear embedding of the input.

    Architecture:  x -> E(x) -> [Legible Bottleneck = concept layer] -> MLP head -> y.

    Because the concepts enter the input in (linear) superposition, a sparse code
    at this site disentangles them, so the named concepts can align with the true
    generative factors, and the task head then computes the label *from named
    concepts*.  ``E`` is identity-initialised so recovery is not an artefact of a
    fixed random basis.  Optional extra bottlenecks can be placed inside the head.
    """

    def __init__(self, d_in: int, d_h: int, n_out: int, m: int, k: int,
                 n_hidden: int = 2, embed_dim: int | None = None):
        super().__init__()
        d_e = embed_dim or d_in
        self.embed = nn.Linear(d_in, d_e)
        with torch.no_grad():                       # identity init (frozen) keeps the
            if d_e == d_in:                          # concept basis from rotating, so
                self.embed.weight.copy_(torch.eye(d_in))  # recovery is not an artefact
                self.embed.bias.zero_()
                self.embed.weight.requires_grad_(False)
                self.embed.bias.requires_grad_(False)
        self.concept = LegibleBottleneck(d_e, m, k)
        head = []
        d_prev = d_e
        for _ in range(max(1, n_hidden)):
            head += [nn.Linear(d_prev, d_h), nn.ReLU()]
            d_prev = d_h
        self.head = nn.Sequential(*head)
        self.out = nn.Linear(d_prev, n_out)
        self.bottlenecks = nn.ModuleList([self.concept])
        self.sites: list[BottleneckOutput] = []

    def forward(self, x, mode: str = "full", leak_swap: bool = False):
        self.sites = []
        e = self.embed(x)
        bo, routed = _route(self.concept, e, mode, leak_swap)
        self.sites.append(bo)
        return self.out(self.head(routed))

    def forward_from_code(self, code):
        """Run the head from a (possibly intervened) concept code, used by the
        causal-ablation faithfulness test."""
        routed = self.concept.decode(code)
        return self.out(self.head(routed))

    def maintain(self, momentum: float = 0.99):
        for bn in self.bottlenecks:
            bn.normalize_decoder()
            bn.update_anchor(momentum)


class _Block(nn.Module):
    """One pre-norm Transformer block; the MLP hidden is the bottleneck site."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, m: int, k: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.bottleneck = LegibleBottleneck(d_mlp, m, k)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x, attn_mask, route):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = F.relu(self.fc1(self.ln2(x)))
        B, T, D = h.shape
        bo, routed = route(self.bottleneck, h.reshape(B * T, D))
        x = x + self.fc2(routed.reshape(B, T, D))
        return x, bo


class LegibleTransformer(nn.Module):
    """A minimal decoder-only Transformer with Legible Bottlenecks on each MLP.

    Defaults target the modular-addition task: vocab = p (+ a '=' token), a short
    fixed context, and a readout at the final position.
    """

    def __init__(self, vocab: int, seq_len: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int = 512, n_layers: int = 1, n_out: int | None = None,
                 m: int = 256, k: int = 16):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            _Block(d_model, n_heads, d_mlp, m, k) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, n_out if n_out is not None else vocab)
        self.seq_len = seq_len
        self.sites: list[BottleneckOutput] = []
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, idx, mode: str = "full", leak_swap: bool = False):
        self.sites = []

        def route(bn, h):
            return _route(bn, h, mode, leak_swap)

        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x, bo = blk(x, self.causal_mask[:T, :T], route)
            self.sites.append(bo)
        x = self.ln_f(x)
        return self.unembed(x[:, -1, :])  # readout at last position

    def maintain(self, momentum: float = 0.99):
        for blk in self.blocks:
            blk.bottleneck.normalize_decoder()
            blk.bottleneck.update_anchor(momentum)


class LegibleLMTransformer(nn.Module):
    """A decoder-only language model routed through Legible Bottlenecks: the
    by-construction setting, where the network is trained *with* its concept
    channels from the first step rather than retrofitted.

    Same routing contract as ``LegibleTransformer`` (modes full / legible /
    leak-swap, one bottleneck on every MLP hidden state, ``model.sites``
    populated after each forward), but the readout is a language model: logits
    at *every* position, so the certificate and the behavioural objectives
    cover the whole emitted distribution, not a single classification head.

    ``sites`` (an optional set of block indices) restricts the non-full
    routing to a subset of bottleneck sites; blocks outside it pass through
    unchanged.  This is used to compare like with like against a single-site
    post-hoc splice.
    """

    def __init__(self, vocab: int, seq_len: int, d_model: int = 256,
                 n_heads: int = 4, d_mlp: int = 1024, n_layers: int = 4,
                 m: int = 1024, k: int = 32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            _Block(d_model, n_heads, d_mlp, m, k) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab)
        self.seq_len = seq_len
        self.sites: list[BottleneckOutput] = []
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, idx, mode: str = "full", leak_swap: bool = False,
                sites: set[int] | None = None):
        self.sites = []

        def route(bn, h):
            i = len(self.sites)
            on = sites is None or i in sites
            return _route(bn, h, mode if on else "full", leak_swap and on)

        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x, bo = blk(x, self.causal_mask[:T, :T], route)
            self.sites.append(bo)
        x = self.ln_f(x)
        return self.unembed(x)  # (B, T, vocab): next-token logits everywhere

    def maintain(self, momentum: float = 0.99):
        for blk in self.blocks:
            blk.bottleneck.normalize_decoder()
            blk.bottleneck.update_anchor(momentum)

    @torch.no_grad()
    def load_opaque(self, opaque: "OpaqueLMTransformer"):
        """Copy an ``OpaqueLMTransformer``'s weights into this model (the two
        share every parameter name except the bottlenecks), so a post-hoc SAE
        can be spliced into a frozen opaque model via the legible routing
        modes.  In mode='full' the two models are then exactly identical."""
        missing, unexpected = self.load_state_dict(opaque.state_dict(), strict=False)
        assert not unexpected, unexpected
        assert all(".bottleneck." in name for name in missing), missing
        return self


class _OpaqueBlock(nn.Module):
    """One pre-norm Transformer block of identical shape to ``_Block`` but
    with no bottleneck; the MLP hidden state is exposed for post-hoc SAE
    fitting."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x, attn_mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = F.relu(self.fc1(self.ln2(x)))
        x = x + self.fc2(h)
        return x, h


class OpaqueLMTransformer(nn.Module):
    """The opaque twin of ``LegibleLMTransformer``: identical architecture and
    parameter shapes, no bottlenecks.  It is the capability reference (what
    the task costs without any legibility constraint) and the substrate for
    the post-hoc pipeline: ``forward(idx, keep_hidden=True)`` stores each
    block's MLP hidden state (flattened to rows) in ``self.hidden`` for SAE
    fitting."""

    def __init__(self, vocab: int, seq_len: int, d_model: int = 256,
                 n_heads: int = 4, d_mlp: int = 1024, n_layers: int = 4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            _OpaqueBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab)
        self.seq_len = seq_len
        self.hidden: list[torch.Tensor] = []
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, idx, keep_hidden: bool = False):
        self.hidden = []
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x, h = blk(x, self.causal_mask[:T, :T])
            if keep_hidden:
                self.hidden.append(h.reshape(B * T, -1))
        x = self.ln_f(x)
        return self.unembed(x)


class OpaqueMLP(nn.Module):
    """Identical capacity and shape to ``LegibleMLP`` but with no bottleneck --
    the baseline whose internals one would have to interpret *post hoc* (e.g. by
    fitting an SAE to its embedding activations).  ``embedding(x)`` exposes the
    layer the head reads, the apples-to-apples site for a post-hoc SAE."""

    def __init__(self, d_in: int, d_h: int, n_out: int, n_hidden: int = 2,
                 embed_dim: int | None = None):
        super().__init__()
        d_e = embed_dim or d_in
        self.embed = nn.Linear(d_in, d_e)
        with torch.no_grad():
            if d_e == d_in:
                self.embed.weight.copy_(torch.eye(d_in))
                self.embed.bias.zero_()
        head = []
        d_prev = d_e
        for _ in range(max(1, n_hidden)):
            head += [nn.Linear(d_prev, d_h), nn.ReLU()]
            d_prev = d_h
        self.head = nn.Sequential(*head)
        self.out = nn.Linear(d_prev, n_out)

    def forward(self, x):
        return self.out(self.head(self.embed(x)))

    def embedding(self, x):
        return self.embed(x)

    def head_forward(self, e):
        return self.out(self.head(e))

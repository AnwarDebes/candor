"""Datasets with *known ground-truth structure*, so interpretability claims can
be checked rather than asserted.

``planted_concepts`` is the headline benchmark: a set of ``K`` latent binary
concepts is linearly mixed (superposed) into a high-dimensional observation, and
the label is a known sparse Boolean function of a few of them.  Because the
true concepts and the true mechanism are known, this setting can measure whether
CANDOR's named
concepts *recover* them and whether its certificate is *sound*.

``trojan_concepts`` adds a rare trigger, a direction that is **not** one of the
named concepts, which flips the label.  A model that exploits it must route the
computation through the leak, so the Faithfulness Certificate should light up:
dark-computation detection with a ground-truth answer key.

``modular_addition`` is the classic algorithmic task (a + b mod p) used to show
CANDOR composes with attention on a genuinely non-linear circuit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PlantedData:
    X: torch.Tensor          # (N, n_in) observations
    y: torch.Tensor          # (N,) labels
    G: torch.Tensor          # (n_in, K) mixing matrix (true concept directions)
    concepts: torch.Tensor   # (N, K) ground-truth concept activations in {0,1}
    relevant: np.ndarray     # indices of label-relevant concepts
    is_trojan: torch.Tensor  # (N,) bool, trigger present (zeros if none)
    n_in: int
    K: int
    n_classes: int = 2


def _make_concepts(rng, N, K, density):
    return (rng.random((N, K)) < density).astype(np.float32)


def planted_concepts(n_samples: int = 20000, n_in: int = 64, K: int = 16,
                     n_relevant: int = 5, density: float = 0.3,
                     noise: float = 0.1, seed: int = 0,
                     task: str = "sum") -> PlantedData:
    """Sparse binary concepts -> linear mixing -> label is a known function of a
    fixed relevant subset.

    task='sum'    : label = number of active relevant concepts (0..n_relevant).
                    Monotone and learnable; every relevant concept causally shifts
                    the label, so each must be represented -> clean recovery.
    task='parity' : label = parity of the relevant subset.  The hardest causal
                    test (each relevant concept flips the label) but only a couple
                    of concepts need representing.
    """
    rng = np.random.default_rng(seed)
    g = _make_concepts(rng, n_samples, K, density)                 # (N, K)
    relevant = np.sort(rng.choice(K, size=n_relevant, replace=False))
    if task == "parity":
        y = (g[:, relevant].sum(1) % 2).astype(np.int64)
        n_classes = 2
    elif task == "sum":
        y = g[:, relevant].sum(1).astype(np.int64)
        n_classes = int(n_relevant + 1)
    else:
        raise ValueError(task)

    G = rng.standard_normal((n_in, K)).astype(np.float32)
    G /= np.linalg.norm(G, axis=0, keepdims=True)                  # unit concept dirs
    X = g @ G.T + noise * rng.standard_normal((n_samples, n_in)).astype(np.float32)

    return PlantedData(
        X=torch.tensor(X), y=torch.tensor(y), G=torch.tensor(G),
        concepts=torch.tensor(g), relevant=relevant,
        is_trojan=torch.zeros(n_samples, dtype=torch.bool), n_in=n_in, K=K,
        n_classes=n_classes,
    )


def trojan_concepts(n_samples: int = 20000, n_in: int = 64, K: int = 16,
                    n_relevant: int = 3, density: float = 0.3, noise: float = 0.1,
                    trojan_rate: float = 0.05, seed: int = 0) -> PlantedData:
    """Like ``planted_concepts`` but a rare trigger direction (orthogonal to the
    named concepts, appended as extra input dims) flips the label when present."""
    base = planted_concepts(n_samples, n_in, K, n_relevant, density, noise, seed,
                            task="parity")
    rng = np.random.default_rng(seed + 7)

    trig = (rng.random(n_samples) < trojan_rate)
    # the trigger is a fixed pattern in *extra* input dimensions, disjoint from G
    n_trig_dims = 4
    trig_dir = rng.standard_normal((n_trig_dims,)).astype(np.float32)
    trig_dir /= np.linalg.norm(trig_dir)
    extra = noise * rng.standard_normal((n_samples, n_trig_dims)).astype(np.float32)
    extra[trig] += 3.0 * trig_dir                                  # strong, rare signal

    X = torch.cat([base.X, torch.tensor(extra)], dim=1)
    y = base.y.clone()
    y[torch.tensor(trig)] = 1 - y[torch.tensor(trig)]             # flip label on trigger
    G = torch.cat([base.G, torch.zeros(n_trig_dims, K)], dim=0)   # trigger NOT a named concept

    return PlantedData(
        X=X, y=y, G=G, concepts=base.concepts, relevant=base.relevant,
        is_trojan=torch.tensor(trig), n_in=n_in + n_trig_dims, K=K, n_classes=2,
    )


@dataclass
class ModAddData:
    X: torch.Tensor  # (N, 3) token ids: [a, b, '=']
    y: torch.Tensor  # (N,) label (a + b) mod p
    p: int
    vocab: int
    seq_len: int


def modular_addition(p: int = 59, seed: int = 0) -> ModAddData:
    """All (a, b) pairs for a + b (mod p); vocab = p tokens + one '=' token."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    eq = np.full_like(a, p)                       # '=' token id == p
    X = np.stack([a, b, eq], axis=1).astype(np.int64)
    y = ((a + b) % p).astype(np.int64)
    return ModAddData(X=torch.tensor(X), y=torch.tensor(y), p=p,
                      vocab=p + 1, seq_len=3)


@dataclass
class SeqData:
    X: torch.Tensor  # (N, L) token ids
    y: torch.Tensor  # (N,) label
    vocab: int
    seq_len: int
    n_classes: int


def sequence_max(n_samples: int = 12000, seq_len: int = 10, vocab: int = 12,
                 seed: int = 0) -> SeqData:
    """A fast attention task: label = max token over a length-L sequence.

    Solving it requires aggregating information *across positions* (the canonical
    job of self-attention), so it shows the Legible Bottleneck composes with
    attention, without the multi-10k-step grokking dynamics of modular addition.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab, size=(n_samples, seq_len)).astype(np.int64)
    y = X.max(1).astype(np.int64)
    return SeqData(X=torch.tensor(X), y=torch.tensor(y), vocab=vocab,
                   seq_len=seq_len, n_classes=vocab)


def split(n: int, frac_train: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(frac_train * n)
    return torch.tensor(idx[:cut]), torch.tensor(idx[cut:])

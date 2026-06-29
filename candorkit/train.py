"""Training loops for CANDOR models, the opaque baseline, and a post-hoc SAE
(the standard interpretability pipeline I compare against).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .bottleneck import LegibleBottleneck
from .losses import LossWeights, candor_loss


def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    steps: int = 3000
    lr: float = 1e-3
    batch: int = 1024
    weights: LossWeights = field(default_factory=LossWeights)
    anchor_momentum: float = 0.99
    log_every: int = 250
    seed: int = 0
    weight_decay: float = 0.0


def _batches(n, batch, rng):
    idx = rng.permutation(n)
    for i in range(0, n, batch):
        yield idx[i:i + batch]


def train_candor(model, X, y, idx_tr, cfg: TrainConfig, device=None, verbose=True):
    import numpy as np
    device = device or device_auto()
    model.to(device)
    Xt, yt = X[idx_tr].to(device), y[idx_tr].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    history = []
    step = 0
    while step < cfg.steps:
        for bidx in _batches(Xt.shape[0], cfg.batch, rng):
            if step >= cfg.steps:
                break
            model.train()
            xb = Xt[bidx]
            yb = yt[bidx]
            lb = candor_loss(model, xb, yb, cfg.weights)
            opt.zero_grad()
            lb.total.backward()
            opt.step()
            model.maintain(cfg.anchor_momentum)
            if step % cfg.log_every == 0:
                history.append({"step": step, "total": float(lb.total.detach()), **lb.parts})
                if verbose:
                    print(f"  step {step:5d}  total={float(lb.total):.4f}  "
                          f"task={lb.parts['task']:.3f}  faith={lb.parts['faith']:.4f}  "
                          f"leak={lb.parts['leak']:.4f}  causal={lb.parts['causal']:.4f}")
            step += 1
    return history


def train_opaque(model, X, y, idx_tr, steps=3000, lr=1e-3, batch=1024,
                 device=None, seed=0, verbose=False):
    import numpy as np
    device = device or device_auto()
    model.to(device)
    Xt, yt = X[idx_tr].to(device), y[idx_tr].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    step = 0
    while step < steps:
        for bidx in _batches(Xt.shape[0], batch, rng):
            if step >= steps:
                break
            model.train()
            logits = model(Xt[bidx])
            loss = F.cross_entropy(logits, yt[bidx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return model


def fit_sae(acts, m, k, steps=2000, lr=1e-3, batch=1024, device=None, seed=0):
    """Fit a post-hoc TopK SAE to a matrix of activations (the standard pipeline:
    train the model opaquely, then learn a dictionary on its activations)."""
    import numpy as np
    device = device or device_auto()
    d = acts.shape[1]
    sae = LegibleBottleneck(d, m, k).to(device)
    acts = acts.to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    step = 0
    while step < steps:
        for bidx in _batches(acts.shape[0], batch, rng):
            if step >= steps:
                break
            sae.train()
            h = acts[bidx]
            out = sae(h)
            loss = ((out.recon - h) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            step += 1
    return sae


@torch.no_grad()
def sae_codes(sae, acts, batch=4096, device=None):
    device = device or device_auto()
    sae.eval()
    chunks = []
    for i in range(0, acts.shape[0], batch):
        out = sae(acts[i:i + batch].to(device))
        chunks.append(out.code.cpu())
    return torch.cat(chunks).numpy()

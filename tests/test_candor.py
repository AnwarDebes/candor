"""Offline unit tests for candorkit (no network, CPU, fast)."""
import numpy as np
import torch

import candorkit as ck
from candorkit.bottleneck import LegibleBottleneck, topk_mask


def test_topk_enforces_l0():
    z = torch.rand(16, 50)
    out = topk_mask(z, k=7)
    assert (out > 0).sum(-1).max().item() <= 7
    assert torch.allclose((out > 0).sum(-1).float().mean(), torch.tensor(7.0))


def test_bottleneck_shapes_and_identity_passthrough():
    bn = LegibleBottleneck(d=8, m=32, k=4)
    h = torch.randn(5, 8)
    o = bn(h)
    assert o.code.shape == (5, 32) and o.recon.shape == (5, 8)
    assert (o.code > 0).sum(-1).max().item() <= 4
    # recon + leak reconstructs h exactly (definition of the leak)
    assert torch.allclose(o.recon + o.leak, h, atol=1e-5)


def test_legible_and_full_forward_shapes():
    data = ck.planted_concepts(n_samples=200, n_in=16, K=6, n_relevant=3, seed=1)
    model = ck.LegibleMLP(d_in=data.n_in, d_h=32, n_out=data.n_classes, m=24, k=4)
    full = model(data.X[:10], mode="full")
    leg = model(data.X[:10], mode="legible")
    assert full.shape == leg.shape == (10, data.n_classes)


def test_certificate_in_unit_interval():
    data = ck.planted_concepts(n_samples=200, n_in=16, K=6, n_relevant=3, seed=2)
    model = ck.LegibleMLP(d_in=data.n_in, d_h=32, n_out=data.n_classes, m=24, k=4)
    cert = ck.certify(model, data.X[:100], data.y[:100])
    assert 0.0 <= cert.delta <= 1.0
    assert 0.0 <= cert.agreement <= 1.0
    assert cert.per_example_delta.min() >= 0 and cert.per_example_delta.max() <= 1.0001


def test_loss_is_finite():
    data = ck.planted_concepts(n_samples=256, n_in=16, K=6, n_relevant=3, seed=3)
    model = ck.LegibleMLP(d_in=data.n_in, d_h=32, n_out=data.n_classes, m=24, k=4)
    lb = ck.candor_loss(model, data.X[:128], data.y[:128], ck.LossWeights())
    assert torch.isfinite(lb.total)
    assert set(lb.parts) >= {"task", "faith", "leak", "causal", "anchor"}


def test_endtoend_pipeline_learns_and_certifies():
    """A short train should solve the task, recover concepts, and certify tightly."""
    torch.manual_seed(0)
    data = ck.planted_concepts(n_samples=6000, n_in=32, K=8, n_relevant=4,
                               density=0.3, noise=0.1, seed=0, task="sum")
    tr, va = ck.split(data.X.shape[0], 0.8, seed=0)
    model = ck.LegibleMLP(d_in=data.n_in, d_h=96, n_out=data.n_classes, m=32, k=6)
    ck.train_candor(model, data.X, data.y, tr,
                    ck.TrainConfig(steps=800, lr=2e-3, batch=512), device="cpu",
                    verbose=False)
    acc_leg = ck.task_accuracy(model, data.X[va], data.y[va], mode="legible")
    cert = ck.certify(model, data.X[va], data.y[va])
    codes = ck.concept_activations(model, data.X[va])
    rec = ck.concept_recovery(codes, data.concepts[va].numpy())
    assert acc_leg > 0.85, acc_leg
    assert cert.delta < 0.1, cert.delta
    assert rec["recovery"] > 0.7, rec["recovery"]


def test_trojan_data_flips_labels():
    d = ck.trojan_concepts(n_samples=2000, n_in=16, K=6, n_relevant=3,
                           trojan_rate=0.1, seed=0)
    assert d.is_trojan.sum() > 0
    assert d.X.shape[1] == d.n_in
    assert d.n_classes == 2


def test_modular_addition_shapes():
    d = ck.modular_addition(p=11, seed=0)
    assert d.X.shape == (121, 3) and d.vocab == 12
    a, b = d.X[:, 0], d.X[:, 1]
    assert torch.equal(d.y, (a + b) % 11)

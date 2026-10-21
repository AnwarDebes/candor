"""Train the two small CANDOR models the live page runs, with candorkit itself,
and export their weights, ground truth and measured metrics to models.json.

    python live/export_model.py            # from the repository root, about a minute on a CPU

The page re-implements only the forward pass (encode, top-k code, decode, head);
tests/forward.test.mjs checks it against the PyTorch outputs stored here.
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import candorkit as ck  # noqa: E402
from candorkit.metrics import concept_recovery  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "models.json"
SKIP = ("ema_dec", "act_count", "seen", "type_id")


def r6(a):
    return np.round(np.asarray(a, dtype=np.float64), 6).tolist()


def weights(model):
    out = {}
    for k, v in model.state_dict().items():
        if k.endswith(SKIP) or k.startswith("bottlenecks."):
            continue
        out[k] = {"shape": list(v.shape), "data": r6(v.detach().cpu().numpy().ravel())}
    return out


@torch.no_grad()
def run(model, X):
    model.eval()
    full = model(X, mode="full")
    code = model.sites[0].code.clone()
    leg = model(X, mode="legible")
    pf, pl = F.softmax(full, -1), F.softmax(leg, -1)
    return pf, pl, 0.5 * (pf - pl).abs().sum(-1), code


def golden(model, X, idx):
    pf, pl, d, code = run(model, X[idx])
    return {"x": r6(X[idx].numpy()), "p_full": r6(pf.numpy()), "p_leg": r6(pl.numpy()), "delta": r6(d.numpy()),
            "active": [torch.nonzero(code[i] > 0).flatten().tolist() for i in range(len(idx))]}


def typical(code):
    """mean activation of each atom when it is active (used to switch an atom on)"""
    c = code.numpy()
    out = []
    for a in range(c.shape[1]):
        on = c[:, a][c[:, a] > 0]
        out.append(round(float(on.mean()), 4) if len(on) else 0.0)
    return out


torch.manual_seed(0)
cfg = ck.TrainConfig(steps=2000, lr=2e-3, batch=512)

# ---------------------------------------------------------------- planted concepts, count task
data = ck.planted_concepts(n_samples=12000, n_in=32, K=8, n_relevant=4, density=0.3, noise=0.1, seed=0, task="sum")
tr, va = ck.split(data.X.shape[0], 0.8, seed=0)
model = ck.LegibleMLP(d_in=data.n_in, d_h=96, n_out=data.n_classes, m=32, k=6)
ck.train_candor(model, data.X, data.y, tr, cfg, device="cpu", verbose=False)
cert = ck.certify(model, data.X[va], data.y[va])
pf, pl, d, code = run(model, data.X[va])
rec = concept_recovery(ck.concept_activations(model, data.X[va]), data.concepts[va].numpy())
yv = data.y[va]
planted = {
    "task": "count how many of the relevant concepts are present", "noise": 0.1, "density": 0.3,
    "config": {"d_in": 32, "K": 8, "m": 32, "k": 6, "d_h": 96, "n_classes": int(data.n_classes)},
    "G": r6(data.G.numpy()), "relevant": [int(i) for i in data.relevant],
    "matching": [[int(a), int(c)] for a, c in rec["matching"]], "concept_corr": r6([rec["per_concept"][k] for k in range(8)]),
    "typical": typical(code), "weights": weights(model),
    "metrics": {"delta_mean": cert.delta, "delta_p95": cert.delta_p95, "agreement": cert.agreement,
                "acc_full": float((pf.argmax(-1) == yv).float().mean()), "acc_legible": float((pl.argmax(-1) == yv).float().mean()),
                "recovery": float(rec["recovery"]), "n_val": int(len(va))},
    "golden": golden(model, data.X[va], torch.arange(24)),
}
print("planted", {k: round(v, 4) if isinstance(v, float) else v for k, v in planted["metrics"].items()})

# ---------------------------------------------------------------- the backdoor
n, seed, rate = 12000, 0, 0.05
td = ck.trojan_concepts(n_samples=n, n_in=32, K=8, n_relevant=3, trojan_rate=rate, seed=seed)
rng = np.random.default_rng(seed + 7)          # replay the generator to recover the trigger direction
rng.random(n)
trig_dir = rng.standard_normal((4,)).astype(np.float32)
trig_dir /= np.linalg.norm(trig_dir)
ttr, tva = ck.split(td.X.shape[0], 0.8, seed=0)
torch.manual_seed(0)
tm = ck.LegibleMLP(d_in=td.n_in, d_h=96, n_out=td.n_classes, m=32, k=6)
ck.train_candor(tm, td.X, td.y, ttr, cfg, device="cpu", verbose=False)
pf, pl, d, code = run(tm, td.X[tva])
is_t = td.is_trojan[tva].numpy().astype(bool)
assert np.allclose(td.X[tva][torch.tensor(is_t)][:, -4:].mean(0).numpy() / 3.0, trig_dir, atol=0.02), "trigger replay mismatch"
from sklearn.metrics import roc_auc_score  # noqa: E402
yv = td.y[tva].numpy()
auc = float(roc_auc_score(is_t, d.numpy()))
trojan = {
    "task": "parity of the relevant concepts, flipped by a hidden trigger", "noise": 0.1, "density": 0.3, "trojan_rate": rate,
    "config": {"d_in": 36, "K": 8, "m": 32, "k": 6, "d_h": 96, "n_classes": 2},
    "G": r6(td.G.numpy()), "relevant": [int(i) for i in td.relevant], "trigger_dir": r6(trig_dir), "trigger_scale": 3.0,
    "typical": typical(code), "weights": weights(tm),
    "metrics": {"auc": auc, "delta_clean": float(d.numpy()[~is_t].mean()), "delta_trigger": float(d.numpy()[is_t].mean()),
                "full_follows_trigger": float((pf.argmax(-1).numpy()[is_t] == yv[is_t]).mean()),
                "legible_follows_trigger": float((pl.argmax(-1).numpy()[is_t] == yv[is_t]).mean()),
                "n_val": int(len(tva)), "n_trigger": int(is_t.sum())},
    "golden": golden(tm, td.X[tva], torch.tensor(np.r_[np.where(~is_t)[0][:12], np.where(is_t)[0][:12]])),
}
print("trojan", {k: round(v, 4) if isinstance(v, float) else v for k, v in trojan["metrics"].items()})
OUT.write_text(json.dumps({"planted": planted, "trojan": trojan}, separators=(",", ":")))
print("wrote", OUT, OUT.stat().st_size // 1024, "KB")

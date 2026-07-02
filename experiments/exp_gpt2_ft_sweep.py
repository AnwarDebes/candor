"""Experiment 3c: seed and weighting sweep of the layer-wise fine-tuning probe.

exp_gpt2_ft.py reported a single-seed, single-weighting negative: with the
spliced layer's MLP unfrozen, adding the behavioural and causal objectives
certifies *less* tightly than the reconstruction-only control.  This script
tests whether that negative is an artefact of the seed or of the loss
weighting.  It reuses exp_gpt2_ft's corpus, conditions, and training code
unchanged and sweeps

    seeds      {0, 1, 2}
    weightings (faith_w, causal_w) in {(0.3, 0.3), (1.0, 1.0), (3.0, 3.0)}

FT-SAE does not depend on the weighting, so it runs once per seed.  Everything
else (layer, corpus, split, steps, batch, learning rates) is exactly
exp_gpt2_ft.main()'s defaults.  Aggregated results (per-run raw numbers plus
mean and standard deviation over seeds, per weighting) are written to
results/gpt2_ft_sweep.json; results/gpt2_ft.json is never touched.

    python experiments/exp_gpt2_ft_sweep.py                          # full sweep
    python experiments/exp_gpt2_ft_sweep.py --seed 1 --partial DIR   # one seed
    python experiments/exp_gpt2_ft_sweep.py --aggregate DIR          # merge partials
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.exp_gpt2_ft as ft  # noqa: E402

RESULTS = ft.RESULTS
OUT = os.path.join(RESULTS, "gpt2_ft_sweep.json")

SEEDS = (0, 1, 2)
WEIGHTINGS = (0.3, 1.0, 3.0)  # applied to faith_w and causal_w jointly

# exp_gpt2_ft.main() defaults, kept identical so the sweep extends, not
# replaces, the committed single-seed run.
CFG = dict(layer=9, seq_len=24, n_seq=1400, m=2048, k=32, steps=500, batch=12,
           lr_tc=2e-3, lr_mlp=1e-4, recon_w=1.0)


def _setup(device="auto"):
    """Corpus, split, and device exactly as in exp_gpt2_ft.main()."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(ft._gpt2_src())
    ids = ft.load_corpus(tok, CFG["seq_len"], CFG["n_seq"])
    n_tr = int(0.85 * ids.shape[0])
    tr, va = ids[:n_tr], ids[n_tr:]
    dev = ft.pick_device(device, CFG["layer"], CFG["batch"], CFG["seq_len"],
                         CFG["m"], CFG["k"])
    return tr, va, dev


def run_seed(seed, device="auto"):
    """One seed's block: FT-SAE once, then FT-CANDOR at every weighting."""
    tr, va, dev = _setup(device)
    train_cfg = {k: v for k, v in CFG.items() if k not in ("seq_len", "n_seq")}
    common = dict(ids_tr=tr, ids_va=va, device=dev, seed=seed, **train_cfg)
    block = {"seed": seed, "device": dev,
             "n_train": int(tr.shape[0]), "n_val": int(va.shape[0])}

    model0, splicer0 = ft.fresh_model(CFG["layer"], dev)
    block["orig_loss_full"] = ft.ce_loss(model0, splicer0, va, CFG["batch"],
                                         dev, "full")
    del model0, splicer0

    print(f"[seed {seed}] FT-SAE (weighting-independent) ...", flush=True)
    block["ft_sae"] = ft.run_condition(candor=False, faith_w=1.0, causal_w=1.0,
                                       **common)
    block["ft_candor"] = {}
    for w in WEIGHTINGS:
        print(f"[seed {seed}] FT-CANDOR (faith_w = causal_w = {w}) ...", flush=True)
        block["ft_candor"][str(w)] = ft.run_condition(candor=True, faith_w=w,
                                                      causal_w=w, **common)
    return block


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0}


def aggregate(blocks):
    blocks = sorted(blocks, key=lambda b: b["seed"])
    keys = ("delta", "leak_swap", "recon_unexplained", "loss_full", "loss_legible")

    summary = {"ft_sae": {k: _stats([b["ft_sae"][k] for b in blocks]) for k in keys},
               "ft_candor": {}}
    for w in WEIGHTINGS:
        summary["ft_candor"][str(w)] = {
            k: _stats([b["ft_candor"][str(w)][k] for b in blocks]) for k in keys}

    return {
        **CFG,
        "seeds": [b["seed"] for b in blocks],
        "weightings": list(WEIGHTINGS),
        "n_train": blocks[0]["n_train"], "n_val": blocks[0]["n_val"],
        "device": blocks[0]["device"],
        "orig_loss_full": float(np.mean([b["orig_loss_full"] for b in blocks])),
        "runs": {str(b["seed"]): {"ft_sae": b["ft_sae"],
                                  "ft_candor": b["ft_candor"]} for b in blocks},
        "summary": summary,
    }


def main(seeds=SEEDS, device="auto", out=OUT):
    t0 = time.time()
    blocks = [run_seed(s, device) for s in seeds]
    agg = aggregate(blocks)
    agg["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"wrote {out}  ({agg['elapsed_sec']}s)", flush=True)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=None,
                    help="run a single seed; write a partial JSON to --partial")
    ap.add_argument("--partial", default=None,
                    help="directory for per-seed partial results")
    ap.add_argument("--aggregate", default=None,
                    help="aggregate partial JSONs from this directory and exit")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.aggregate:
        files = sorted(f for f in os.listdir(a.aggregate) if f.endswith(".json"))
        blocks = [json.load(open(os.path.join(a.aggregate, f))) for f in files]
        agg = aggregate(blocks)
        agg["elapsed_sec"] = sum(b["ft_sae"]["train_sec"]
                                 + sum(c["train_sec"] for c in b["ft_candor"].values())
                                 for b in blocks)
        with open(a.out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"wrote {a.out} from {len(blocks)} partial(s)")
    elif a.seed is not None:
        os.makedirs(a.partial or ".", exist_ok=True)
        block = run_seed(a.seed, a.device)
        path = os.path.join(a.partial or ".", f"seed{a.seed}.json")
        with open(path, "w") as f:
            json.dump(block, f, indent=2)
        print(f"wrote {path}")
    else:
        main(device=a.device, out=a.out)

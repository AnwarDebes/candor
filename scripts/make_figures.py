"""Generate the paper figures from results/*.json -> paper/figures/*.pdf."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 130, "savefig.bbox": "tight"})
TEAL = "#1b7f7f"
RUST = "#b3502d"


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_tax():
    d = load("tax.json")
    if not d:
        return
    rows = d["sweep"]
    k = [r["k"] for r in rows]
    accl = [100 * r["acc_legible"] for r in rows]
    delta = [r["delta_mean"] for r in rows]
    rec = [r["recovery"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.plot(k, accl, "o-", color=TEAL, label="legible-model accuracy")
    ax1.plot(k, [100 * r for r in rec], "s--", color="#444", label="concept recovery")
    ax1.set_xlabel("legible-channel capacity $k$ (active named concepts)")
    ax1.set_ylabel("accuracy / recovery (%)", color=TEAL)
    ax1.axvline(d["n_relevant"], color="gray", ls=":", lw=1)
    ax2 = ax1.twinx()
    ax2.plot(k, delta, "^-", color=RUST, label=r"certificate $\delta$")
    ax2.set_ylabel(r"dark-computation bound $\delta$", color=RUST)
    ax2.set_ylim(bottom=0)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="center right")
    plt.title("Interpretability-tax / legibility frontier")
    plt.savefig(os.path.join(FIG, "tax_frontier.pdf"))
    plt.close()


def fig_trojan():
    d = load("planted.json")
    if not d or "delta_clean_sample" not in d.get("trojan", {}):
        return
    t = d["trojan"]
    clean = np.array(t["delta_clean_sample"])
    troj = np.array(t["delta_trojan_sample"])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    hi = max(clean.max(), troj.max()) + 1e-3
    bins = np.linspace(0, hi, 40)
    ax.hist(clean, bins=bins, color=TEAL, alpha=0.7, label="clean inputs", density=True)
    ax.hist(troj, bins=bins, color=RUST, alpha=0.7, label="trojan-triggered", density=True)
    ax.set_xlabel(r"per-input certificate $\delta(x)$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.set_title(f"Dark-computation detection (AUC = {t['auc_delta_detects_trojan']:.3f})")
    plt.savefig(os.path.join(FIG, "trojan_detection.pdf"))
    plt.close()


def fig_ablation():
    """Each training term is load-bearing for a *distinct* guarantee."""
    d = load("planted.json")
    if not d:
        return
    ab = d["ablations"]
    main = d["main"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))

    axes[0].bar(["CANDOR", "no leak"], [main["recovery"], ab["no_leak"]["recovery"]],
                color=[TEAL, RUST])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("concept recovery")
    axes[0].set_title("leak term\n$\\rightarrow$ recovery")

    axes[1].bar(["CANDOR", "no causal"],
                [main["leak_swap_effect"], ab["no_causal"]["leak_swap_effect"]],
                color=[TEAL, RUST])
    axes[1].set_ylabel(r"leak causal effect (TV)")
    axes[1].set_title("causal term\n$\\rightarrow$ sufficiency")

    axes[2].bar(["anchored", "free"],
                [d["stability"]["anchored"], d["stability"]["free"]],
                color=[TEAL, RUST])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("cross-run stability")
    axes[2].set_title("anchor term\n$\\rightarrow$ stability")

    plt.suptitle("Objective ablations: each term removes a distinct guarantee", y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ablations.pdf"))
    plt.close()


def main():
    fig_tax(); fig_trojan(); fig_ablation()
    print("wrote figures to", FIG, ":", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()

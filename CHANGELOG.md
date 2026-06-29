# Changelog

## v1.0.0 (2026-06-25)

First public release of **CANDOR** (Concept-ANchored Disentangled Output Routing) and the
paper *The Way Toward Interpretable AGI: Routing Computation Through Certified Legible
Bottlenecks*.

### Added
- **The Legible Bottleneck** primitive (`candorkit/bottleneck.py`): a sparse, typed,
  persistently-named concept channel with three routing modes (full / legible / leak-swap)
  and a measured leak.
- **CANDOR models** (`candorkit/model.py`): a concept-layer MLP and a Transformer whose MLP
  hidden states are bottleneck sites, demonstrating composition with attention.
- **The conjoined training objective** (`candorkit/losses.py`): task + legible-task +
  faithfulness (KL) + leak energy + **causal sufficiency** (leak-swap invariance, an
  interchange/causal-scrubbing signal) + anchoring.
- **The Faithfulness Certificate** (`candorkit/certificate.py`): the per-forward-pass,
  sound-by-construction, re-checkable bound `δ(x) = TV(M, M_legible)`.
- **Ground-truth metrics** (`candorkit/metrics.py`): concept recovery (Hungarian-matched),
  causal necessity & directional accuracy, cross-run anchoring stability.
- **Experiments** (`experiments/`): planted concepts (tax, certificate, recovery, causal
  faithfulness, ablations, anchoring, trojan/dark-computation detection), the
  tax/legibility frontier, a sequence (attention) task, and an honest **GPT-2 small**
  real-LLM probe (`exp_gpt2.py`, needs the `[llm]` extra).
- **Reproducible pipeline**: `experiments/run_all.py` → `results/*.json` →
  `scripts/paper_numbers.py` (`paper/_numbers.tex`) → `scripts/make_figures.py` → paper.
- Offline unit tests, demo, the curated bibliography (`paper/references.bib`), and a
  literature map of the work the paper builds on (`docs/related-work.md`).

### Scope
- Positive results are on synthetic ground-truth tasks and a one-layer Transformer, not a
  frontier LLM. On a frozen GPT-2 retrofit the certificate *machinery* transfers but CANDOR
  does **not** beat a reconstruction-only SAE (reported honestly); its gains are
  by-construction, so by-construction LLM training is the named next step. No completeness
  claim; the certificate is sound, measured, and trainable, not a scale proof. The novelty
  is the **conjunction** of (otherwise-precedented)
  components, operating jointly and at runtime.

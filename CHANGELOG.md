# Changelog

## v1.2.0 (2026-07-02)

### Added
- `experiments/exp_gpt2_ft.py`: the layer-wise by-construction fine-tuning probe on
  GPT-2 that the paper previously named as the next step. Unfreezes the spliced layer's
  MLP (4.7M parameters) and trains it jointly with the channel under a
  capability-anchoring LM loss, in two matched conditions (reconstruction-only vs.
  reconstruction + behavioural faithfulness + causal sufficiency). Auto-selects a CUDA
  device with a probe (about 1 GB peak) and falls back to CPU.
- `results/gpt2_ft.json` plus the corresponding auto-generated paper macros; new paper
  paragraph in section 7.4, updated abstract and limitations, and an appendix
  subsection documenting the protocol.

### Findings (single seed, single weighting; reported as measured)
- Reconstruction and faithfulness dissociate: fine-tuning improves the control
  channel's reconstruction about 8x (unexplained variance 0.375 to 0.046) yet its
  certificate does not tighten (delta 0.114 frozen to 0.128 fine-tuned).
- The by-construction prediction does not hold at single-layer scale: FT-CANDOR
  certifies less tightly than the matched control (delta 0.156 vs 0.128) at matched LM
  loss. Reported as a negative result; full by-construction training remains the open
  question.

### Fixed
- `exp_gpt2.py`: the leak-swap permutation is now created on the activation's device,
  which is required for CUDA runs.

## v1.1.0 (2026-07-02)

Submission-readiness revision of the paper and supporting polish; no experimental
results changed, and `paper/_numbers.tex` still regenerates byte-identical from the
committed `results/*.json`.

### Paper
- Retitled to *The Explanation Is the Forward Pass: Interpretability by Construction
  with Certified Legible Bottlenecks* (the previous working title overclaimed scope).
- Added the author block with affiliation and contact, keywords, a reproducibility
  statement, and an experimental-details appendix (every architecture, hyperparameter,
  seed, and loss weight, matching the released code exactly).
- Tightened the abstract, moved the text to standard editorial voice, and added proof
  sketches to all propositions.
- Corrected two claim/measurement mismatches found in internal review: the necessity
  gap is measured by ablating the relevant concepts' atoms jointly (versus an equal
  number of irrelevant atoms), and the post-hoc SAE's concept recovery is reported as
  slightly higher than CANDOR's rather than "the same".
- Bibliography: protected proper nouns from lowercasing; fixed a venue-less entry.

### Code and figures
- `scripts/make_figures.py`: the backdoor-detection histogram now uses log-scaled
  axes, so the clean/triggered separation is actually visible.
- `candorkit/certificate.py`: docstring corrected. Total variation bounds event
  probabilities of the sampled output; it does not by itself bound argmax
  disagreement, which is why `agreement` is reported as a separate measured quantity.
- `candorkit/metrics.py`: removed dead helpers (`model_decode`, `rerun_from_site`);
  clarified the necessity-test docstring.

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

"""Run every CANDOR experiment and regenerate results/*.json.

    python experiments/run_all.py            # all experiments
    python experiments/run_all.py planted    # one experiment

Then:
    python scripts/paper_numbers.py          # results -> paper/_numbers.tex
    python scripts/make_figures.py           # results -> paper/figures/*.pdf
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import exp_planted, exp_seq, tax_curve  # noqa: E402

ALL = {
    "planted": exp_planted.main,
    "tax": tax_curve.main,
    "seq": exp_seq.main,
}
# The LLM experiments need the optional `transformers` extra and downloads
# (GPT-2 weights or tokeniser, TinyShakespeare), so they are excluded from the
# default run; invoke them explicitly:
#   python experiments/run_all.py gpt2           (or  python experiments/exp_gpt2.py)
#   python experiments/run_all.py gpt2_ft        (layer-wise fine-tuning probe)
#   python experiments/run_all.py gpt2_ft_sweep  (seed/weighting sweep of the probe)
#   python experiments/run_all.py lm_scratch     (by-construction LM training)
DEFAULT = ["planted", "tax", "seq"]

EXPLICIT = ["gpt2", "gpt2_ft", "gpt2_ft_sweep", "lm_scratch"]


def main(which=None):
    targets = which or DEFAULT
    for name in targets:
        print(f"\n========== {name} ==========")
        if name == "gpt2":
            from experiments import exp_gpt2
            exp_gpt2.main()
        elif name == "gpt2_ft":
            from experiments import exp_gpt2_ft
            exp_gpt2_ft.main()
        elif name == "gpt2_ft_sweep":
            from experiments import exp_gpt2_ft_sweep
            exp_gpt2_ft_sweep.main()
        elif name == "lm_scratch":
            from experiments import exp_lm_scratch
            exp_lm_scratch.main()
        else:
            ALL[name]()


if __name__ == "__main__":
    valid = set(ALL) | set(EXPLICIT)
    args = [a for a in sys.argv[1:] if a in valid]
    main(args or None)

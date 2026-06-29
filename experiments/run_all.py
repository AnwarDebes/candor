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
# The GPT-2 experiment needs the optional `transformers` extra and a ~0.5 GB model
# download, so it is excluded from the default run; invoke it explicitly:
#   python experiments/run_all.py gpt2       (or  python experiments/exp_gpt2.py)
DEFAULT = ["planted", "tax", "seq"]


def main(which=None):
    targets = which or DEFAULT
    for name in targets:
        print(f"\n========== {name} ==========")
        if name == "gpt2":
            from experiments import exp_gpt2
            exp_gpt2.main()
        else:
            ALL[name]()


if __name__ == "__main__":
    valid = set(ALL) | {"gpt2"}
    args = [a for a in sys.argv[1:] if a in valid]
    main(args or None)

# CANDOR: reproduce everything
PY ?= python

.PHONY: help install test experiments planted seq gpt2 gpt2_ft gpt2_ft_sweep lm_scratch tax numbers figures paper all clean

help:
	@echo "make install      - editable install with experiment + dev extras"
	@echo "make test         - run the offline unit tests"
	@echo "make experiments  - run all experiments -> results/*.json"
	@echo "make numbers      - results/*.json -> paper/_numbers.tex"
	@echo "make figures      - results/*.json -> paper/figures/*.pdf"
	@echo "make paper        - numbers + figures + compile paper/candor.pdf"
	@echo "make all          - experiments + paper"

install:
	$(PY) -m pip install -e ".[experiments,dev]"

test:
	$(PY) -m pytest

experiments:
	$(PY) experiments/run_all.py

planted:
	$(PY) experiments/exp_planted.py
seq:
	$(PY) experiments/exp_seq.py
gpt2:
	$(PY) experiments/exp_gpt2.py    # needs: pip install -e ".[llm]"  (+ ~0.5GB download)
gpt2_ft:
	$(PY) experiments/exp_gpt2_ft.py
gpt2_ft_sweep:
	$(PY) experiments/exp_gpt2_ft_sweep.py
lm_scratch:
	$(PY) experiments/exp_lm_scratch.py
tax:
	$(PY) experiments/tax_curve.py

numbers:
	$(PY) scripts/paper_numbers.py

figures:
	$(PY) scripts/make_figures.py

paper: numbers figures
	cd paper && latexmk -pdf candor.tex

all: experiments paper

clean:
	cd paper && latexmk -C || true
	rm -rf **/__pycache__ .pytest_cache

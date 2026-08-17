# QMine — common operations.
VENV ?= .venv
PY   := $(VENV)/bin/python
QM   := $(VENV)/bin/qmine
export HF_HOME := $(CURDIR)/.hf

.PHONY: help install install-min doctor demo full test test-fast lint clean clean-runs

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install:  ## full stack: encoders, notebooks, projections, tests
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[all]"

install-min:  ## core only — runs offline on a hashing encoder
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e .

doctor:  ## check packages, credentials, fonts, profiles
	$(QM) doctor

demo:  ## bundled K12 corpus, 8k rows, shrunken grids (~3 min)
	$(QM) demo

full:  ## bundled K12 corpus, all 50k rows, full grids (~25 min)
	$(QM) run --input data/raw/k12_queries_50k.csv --domain k12_zh \
	         --reference-columns legacy_l1,legacy_l2 --run-id k12-full

test:  ## everything, offline
	$(PY) -m pytest tests/ -q

test-fast:  ## unit tests only, skipping the end-to-end runs
	$(PY) -m pytest tests/test_principles.py tests/test_ops.py -q

clean:  ## caches and build junk
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info

clean-runs:  ## delete run outputs (generations are append-only; this is the escape hatch)
	rm -rf runs

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

full:  ## bundled K12 corpus, all 50k rows, full grids, OFFLINE (~25 min)
	$(QM) run --input data/raw/k12_queries_50k.csv --domain k12_zh --offline \
	         --reference-columns legacy_l1,legacy_l2 --run-id k12-full

#: Defaults for the bundled K12 corpus. Override any of them for another
#: dataset: `make live RUN=x LIVE_INPUT=data/mine.csv LIVE_DOMAIN=finance_zh
#: LIVE_TEXT=q LIVE_REFS=`. An empty LIVE_REFS passes no flag at all, which is
#: correct for a corpus that has no legacy labels.
LIVE_INPUT  ?= data/raw/k12_queries_50k.csv
LIVE_DOMAIN ?= k12_zh
LIVE_TEXT   ?= query
LIVE_REFS   ?= legacy_l1,legacy_l2

live:  ## a PAID live run — `make live RUN=live43` (~4h, ~$30). K12 by default; override LIVE_*
	@test -n "$(RUN)" || (echo "usage: make live RUN=live43" && exit 1)
	@test -f .env || (echo "no .env — live runs need provider keys" && exit 1)
	$(QM) run --input $(LIVE_INPUT) --domain $(LIVE_DOMAIN) \
	         --config configs/live.yaml --provider router \
	         --text-column $(LIVE_TEXT) \
	         $(if $(strip $(LIVE_REFS)),--reference-columns $(LIVE_REFS),) \
	         --run-id $(RUN)

fast:  ## a PAID live run, FAST mode — same analysis, no second-opinion layer; 3 deliverables
	@test -n "$(RUN)" || (echo "usage: make fast RUN=live45" && exit 1)
	@test -f .env || (echo "no .env — live runs need provider keys" && exit 1)
	$(QM) run --input $(LIVE_INPUT) --domain $(LIVE_DOMAIN) \
	         --config configs/live.yaml --provider router --fast \
	         --text-column $(LIVE_TEXT) \
	         $(if $(strip $(LIVE_REFS)),--reference-columns $(LIVE_REFS),) \
	         --run-id $(RUN)

# WHAT THIS TARGET IS, AND WHAT IT IS NOT.
#
# It is the K12 reproduction with live models — the paid sibling of `full`. It is
# NOT "the only way to launch", and this pipeline's whole point is that it runs on
# ANY query dataset. For another corpus, override the LIVE_* variables above, or
# call `qmine run` directly; nothing here is privileged.
#
# What the defaults buy you is the two flags that are easy to forget:
#
#   --config configs/live.yaml    without it the router picks on price alone and
#                                 the pinned models, excluded labs and capability
#                                 list are ignored. Check the plan first with
#                                 `qmine models --config configs/live.yaml`,
#                                 which spends nothing.
#   --reference-columns           changes eight things at once when a corpus HAS
#                                 legacy labels: gold-set and pilot stratification,
#                                 the blindness firewall's forbidden terms, the
#                                 legacy-audit researcher, the corpus audit's
#                                 legacy distribution, and the delivered
#                                 crosswalk.
#
# Forgetting the second is NOT silent — `p1_reference_columns_declared` fails
# warn-only, names the undeclared columns and tells you to relaunch. The default
# saves the round trip; it is not the safety mechanism, and a corpus with no such
# columns passes that gate cleanly with `LIVE_REFS=`.

test:  ## everything, offline
	$(PY) -m pytest tests/ -q

test-fast:  ## unit tests only, skipping the end-to-end runs
	$(PY) -m pytest tests/test_principles.py tests/test_ops.py -q

clean:  ## caches and build junk
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info

clean-runs:  ## delete run outputs (generations are append-only; this is the escape hatch)
	rm -rf runs

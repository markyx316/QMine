#!/bin/bash
# Full post-run analysis stack for the 2026-09-13 cohort (书籍文档 / 软件).
#
# Everything here is POST-RUN: it reads finished runs and writes into work/<domain>/,
# work/cross_new2/ and each run's own postprocessed/ folder. Nothing touches src/, and
# P5_COHORT=new2 keeps every cross-domain table out of the delivered five-domain paths.
#
# TWO ANALYSES THE FIVE GOT AND THIS COHORT CANNOT, with the reason measured:
#   * p5_taxonomy_delta.py — needs a SEARCH-ONLY companion run on the same rows
#     (`fin-pool` vs `fin-pool5`). No `book-pool` / `soft-pool` exists, so there is no
#     second frame to diff against. Would cost one extra mining run per domain.
#   * p5_residue.py AND p5_vs_unified.py — both need the 13-class unified frame labelled on
#     the same strings.
#     Measured coverage of 2026search+assistant: 书籍文档 0.5% (63/11,798), 软件 0.1%
#     (14/11,923), against 100% for all five original domains. Would cost a labelling pass.
# Both are OMITTED rather than run on 0.1% coverage. Say so in the reports.
set -u
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf" P5_COHORT=new2
PY=".venv/bin/python"
LOG="analysis/pooled5/work/new2_analysis.log"
: > "$LOG"
step(){ echo "=== $* ===" | tee -a "$LOG"; "$@" >> "$LOG" 2>&1 || echo "!! FAILED: $*" | tee -a "$LOG"; }

for d in 书籍文档 软件; do
  step $PY analysis/pooled5/p5_tables.py            "$d"
  step $PY analysis/pooled5/p5_digest.py            "$d"
  step $PY analysis/pooled5/p5_postprocess_run_xlsx.py "$d"
  step $PY analysis/pooled5/p5_deliverables.py      "$d"
  step $PY analysis/pooled5/p5_snapshot_classes.py  "$d"
  step $PY analysis/pooled5/p5_snapshot_figs.py     "$d"
  step $PY analysis/pooled5/p5_snapshot_report.py   "$d"
  step $PY analysis/pooled5/p5_snapshot_verify.py   "$d"
done

# cross-cohort (writes to work/cross_new2/, never to work/cross/)
step $PY analysis/pooled5/p5_semantic_nn.py   # report_tables reads its summary
step $PY analysis/pooled5/p5_turnover.py
step $PY analysis/pooled5/p5_depth_control.py
step $PY analysis/pooled5/p5_route_agreement.py
step $PY analysis/pooled5/p5_concentration.py
step $PY analysis/pooled5/p5_form_contrasts.py
step $PY analysis/pooled5/p5_tvd_ci.py
step $PY analysis/pooled5/p5_wrap.py
step $PY analysis/pooled5/p5_cross.py
# T1-T16 cross tables, written to work/report_tables_new2.md (never the five-domain file).
# Several of them degrade gracefully here: the voice and taxonomy-delta tables have no
# inputs for this cohort and are skipped by their own `if exists` guards.
step $PY analysis/pooled5/p5_report_tables.py

echo "--- failures ---" | tee -a "$LOG"; grep -c "^!! FAILED" "$LOG" | tee -a "$LOG"

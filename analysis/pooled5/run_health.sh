#!/bin/bash
# Launch the 健康 run: one week of health search + a specialised health AI assistant, aggregated
# and cleaned by `analysis/pooled5/build_health_corpus.py`, mined as ONE corpus.
#
# REFERENCE COLUMNS ARE SET HERE, IN THE COMMAND, as well as in the config. Both exports carry the
# platform's own category; the pre-registered test in `configs/pool2_health.yaml` kept `legacy_l2`
# (first — it stratifies the gold set, pilot and log-reading sample) and `legacy_type`, and withdrew
# `legacy_dept`. Passing them on the command line keeps the launch self-describing; the values are
# the config's, so the two cannot disagree. No program source is changed for this.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_health.log"
echo "$(date +%H:%M:%S) launching health-pool2 (domain health_zh, corpus 健康, refs legacy_l2,legacy_type)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool2_health.yaml" \
    --input "data/raw/pooled5/健康_pooled5.parquet" \
    --domain health_zh --run-id health-pool2 \
    --reference-columns legacy_l2,legacy_type \
    --fast \
    > "$QM/analysis/pooled5/work/health-pool2.launch.log" 2>&1
echo "$(date +%H:%M:%S) health-pool2 exited $?" >> "$LOG"

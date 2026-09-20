#!/bin/bash
# Launch the 8-snapshot finance run. One run, one taxonomy, eight sources — because two runs
# over the same rows share 0 class codes, so a source-by-source comparison must come from one run.
#
# 43,934 rows: roughly double fin-pool5's 22,934, and the added half is long tail.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_fin8.log"
echo "$(date +%H:%M:%S) launching fin-pool8 (domain finance_zh_v2, corpus 金融8)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool8_fin.yaml" \
    --input "data/raw/pooled5/金融8_pooled5.parquet" \
    --domain finance_zh_v2 --run-id fin-pool8 --fast \
    > "$QM/analysis/pooled5/work/fin-pool8.launch.log" 2>&1
echo "$(date +%H:%M:%S) fin-pool8 exited $?" >> "$LOG"

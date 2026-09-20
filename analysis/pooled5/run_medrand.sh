#!/bin/bash
# Launch the 医疗随机 run (health-pool3): 2026-09-14, 医疗 category, random 10k from traditional search and
# from the 健康管家 assistant — the "random" counterpart of health-pool2's weekly top-1w pair.
#
# Profile `medical_zh_v2` (not health_zh): measured on this corpus it covers 25.4% of rows with 12 seeds
# (9 above the floor) against health_zh's 20.0% with 8, and 6.26% risk coverage against 4.40%. One of its
# 13 risk categories — body_reshape_instruction — matches 0 rows here; that category ships an empty screen
# for this run and nothing else changes.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_medrand.log"
echo "$(date +%H:%M:%S) launching health-pool3 (domain medical_zh_v2, corpus 医疗随机)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool2_medrand.yaml" \
    --input "data/raw/pooled5/医疗随机_pooled5.parquet" \
    --domain medical_zh_v2 --run-id health-pool3 --fast \
    > "$QM/analysis/pooled5/work/health-pool3.launch.log" 2>&1
echo "$(date +%H:%M:%S) health-pool3 exited $?" >> "$LOG"

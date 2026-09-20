#!/bin/bash
# Launch the 8-snapshot medical run. One run, one taxonomy, eight sources — because two runs
# over the same rows share 0 class codes, so a source-by-source comparison must come from one run.
#
# 44,019 rows: roughly double med-pool5's 22,952, and the added half is long tail (two search
# random-10k strata) plus a 1,067-row PV-ranked voice head. Profile `medical_zh_v2` (v1 stays as the
# delivered med-pool5 run's profile). No reference columns: `l2` covers the assistant rows only.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_med8.log"
echo "$(date +%H:%M:%S) launching med-pool8 (domain medical_zh_v2, corpus 医疗8)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool8_med.yaml" \
    --input "data/raw/pooled5/医疗8_pooled5.parquet" \
    --domain medical_zh_v2 --run-id med-pool8 --fast \
    > "$QM/analysis/pooled5/work/med-pool8.launch.log" 2>&1
echo "$(date +%H:%M:%S) med-pool8 exited $?" >> "$LOG"

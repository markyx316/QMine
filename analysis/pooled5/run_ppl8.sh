#!/bin/bash
# Launch the 8-snapshot 人物 run. One run, one taxonomy, eight sources — because two runs over the
# same rows share 0 class codes, so a source-by-source comparison must come from one run.
#
# 43,802 rows (ppl-pool5b mined 21,804). Profile `people_zh_v2`; v1 stays as the delivered
# ppl-pool5b run's profile. Routing: every Zhipu-assigned role goes to Moonshot/DeepSeek — see the
# header of configs/pool8_ppl.yaml for the measured refusals that rule came from.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_ppl8.log"
echo "$(date +%H:%M:%S) launching ppl-pool8 (domain people_zh_v2, corpus 人物8)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool8_ppl.yaml" \
    --input "data/raw/pooled5/人物8_pooled5.parquet" \
    --domain people_zh_v2 --run-id ppl-pool8 --fast \
    > "$QM/analysis/pooled5/work/ppl-pool8.launch.log" 2>&1
echo "$(date +%H:%M:%S) ppl-pool8 exited $?" >> "$LOG"

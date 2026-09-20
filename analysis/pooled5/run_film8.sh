#!/bin/bash
# Launch the 8-snapshot 影视 run. One run, one taxonomy, eight sources.
#
# 43,933 rows (film-pool5 mined 21,934). Profile `film_tv_zh_v2`; v1 stays as the delivered
# film-pool5 run's profile. Default routing from live.yaml.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_film8.log"
echo "$(date +%H:%M:%S) launching film-pool8 (domain film_tv_zh_v2, corpus 影视8)" >> "$LOG"
.venv/bin/qmine run --config "configs/pool8_film.yaml" \
    --input "data/raw/pooled5/影视8_pooled5.parquet" \
    --domain film_tv_zh_v2 --run-id film-pool8 --fast \
    > "$QM/analysis/pooled5/work/film-pool8.launch.log" 2>&1
echo "$(date +%H:%M:%S) film-pool8 exited $?" >> "$LOG"

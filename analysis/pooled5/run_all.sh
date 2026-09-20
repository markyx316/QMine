#!/bin/bash
# Launch the four remaining POOLED-5 runs, keeping at most MAX qmine runs alive at once
# (16 GB box: each run holds an encoder plus ~22k embeddings). fin-pool5 is already running.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
MAX=3
LOG="$QM/analysis/pooled5/work/driver.log"
running(){ pgrep -f "qmine run --config configs/pool5_" | wc -l | tr -d ' '; }
launch(){
  .venv/bin/qmine run --config "configs/pool5_$1.yaml" --input "data/raw/pooled5/$3_pooled5.parquet" \
      --domain "$2" --run-id "$1-pool5" --fast > "$QM/analysis/pooled5/work/$1-pool5.launch.log" 2>&1
  echo "$(date +%H:%M:%S) $1-pool5 exited $?" >> "$LOG"
}
for j in "med medical_zh 医疗" "edu education_zh 教育" "film film_tv_zh 影视" "ppl people_zh 人物"; do
  set -- $j
  while [ "$(running)" -ge "$MAX" ]; do sleep 60; done
  echo "$(date +%H:%M:%S) launching $1-pool5" >> "$LOG"
  launch "$1" "$2" "$3" &
  sleep 120
done
wait
echo "$(date +%H:%M:%S) driver done" >> "$LOG"

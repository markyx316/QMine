#!/bin/bash
# Launch the two NEW domain runs (书籍文档 / 软件), staggered so the encoder downloads
# and the two ~22k-row embedding passes do not collide on a 16 GB box.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf"
LOG="$QM/analysis/pooled5/work/driver_new2.log"
launch(){
  echo "$(date +%H:%M:%S) launching $1-pool5 (domain $2, corpus $3)" >> "$LOG"
  .venv/bin/qmine run --config "configs/pool5_$1.yaml" --input "data/raw/pooled5/$3_pooled5.parquet" \
      --domain "$2" --run-id "$1-pool5" --fast > "$QM/analysis/pooled5/work/$1-pool5.launch.log" 2>&1
  echo "$(date +%H:%M:%S) $1-pool5 exited $?" >> "$LOG"
}
launch book books_docs_zh 书籍文档 &
sleep 180
launch soft software_apps_zh 软件 &
wait
echo "$(date +%H:%M:%S) driver_new2 done" >> "$LOG"

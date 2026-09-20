#!/bin/bash
# Post-processing for 医疗3 (three snapshots, same day, same taxonomy as health-pool3).
# Scoped by P5_COHORT=med3 so nothing written for any other cohort is rebuilt.
#
# `p5_postprocess_run_xlsx` is NOT in the chain, on purpose: it joins provenance columns positionally
# onto the RUN's own delivered workbook, and 医疗3 has no such workbook — it is not a run. The other
# five steps take the corpus + labels directly and work unchanged.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf" P5_COHORT=med3
LOG="$QM/analysis/pooled5/work/post_med3.log"
: > "$LOG"
for s in p5_deliverables p5_snapshot_classes p5_snapshot_figs p5_snapshot_report p5_snapshot_verify; do
    echo "$(date +%H:%M:%S) ── $s" | tee -a "$LOG"
    if ! .venv/bin/python "analysis/pooled5/$s.py" 医疗3 >> "$LOG" 2>&1; then
        echo "$(date +%H:%M:%S) FAILED at $s" | tee -a "$LOG"
        tail -20 "$LOG"
        exit 1
    fi
done
echo "$(date +%H:%M:%S) all five steps done" | tee -a "$LOG"

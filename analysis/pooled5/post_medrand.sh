#!/bin/bash
# Post-run chain for the 医疗随机 run (health-pool3). Post-run operations on the RESULTS only.
# Scoped by P5_COHORT=medr, so nothing written for any other cohort is rebuilt or overwritten.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
export HF_HOME="$QM/.hf" P5_COHORT=medr
LOG="$QM/analysis/pooled5/work/post_medrand.log"
: > "$LOG"
for s in p5_postprocess_run_xlsx p5_deliverables p5_snapshot_classes p5_snapshot_figs p5_snapshot_report p5_snapshot_verify; do
    echo "$(date +%H:%M:%S) ── $s" | tee -a "$LOG"
    if ! .venv/bin/python "analysis/pooled5/$s.py" 医疗随机 >> "$LOG" 2>&1; then
        echo "$(date +%H:%M:%S) FAILED at $s" | tee -a "$LOG"
        exit 1
    fi
done
echo "$(date +%H:%M:%S) all six steps done" | tee -a "$LOG"

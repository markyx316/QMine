#!/bin/bash
# Post-run chain for the 人物8 run. Post-run operations on the RESULTS only — no program source is
# touched. Every script is scoped by P5_COHORT=ppl8 and given the domain explicitly, so nothing
# written for the pool5 / new2 / all7 / fin8 / health2 / med8 cohorts is rebuilt or overwritten.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
# 交付运行是 ppl-pool8b（全新单次跑完，无 resume）——见 run_ppl8b.sh 的抬头。代次是 gen01，
# 和其它域一样，所以这里不需要 P5_GEN 覆盖；run id 的映射在 pooled5_common.RUN_ID 里。
export HF_HOME="$QM/.hf" P5_COHORT=ppl8
LOG="$QM/analysis/pooled5/work/post_ppl8.log"
: > "$LOG"
for s in p5_postprocess_run_xlsx p5_deliverables p5_snapshot_classes p5_snapshot_figs p5_snapshot_report p5_snapshot_verify; do
    echo "$(date +%H:%M:%S) ── $s" | tee -a "$LOG"
    if ! .venv/bin/python "analysis/pooled5/$s.py" 人物8 >> "$LOG" 2>&1; then
        echo "$(date +%H:%M:%S) FAILED at $s" | tee -a "$LOG"
        exit 1
    fi
done
echo "$(date +%H:%M:%S) all six steps done" | tee -a "$LOG"

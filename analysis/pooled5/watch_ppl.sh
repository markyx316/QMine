#!/bin/bash
# One event per people run reaching a terminal state. gen02 is the resume of ppl-pool5;
# ppl-pool5b is the fresh, Zhipu-free relaunch. Covers crashes as well as completions.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
seen=""
while true; do
  for spec in "ppl-pool5:gen02" "ppl-pool5b:gen01"; do
    r="${spec%%:*}"; g="${spec##*:}"
    case " $seen " in *" $r "*) continue;; esac
    f="runs/$r/$g/run_summary.json"
    if [ -f "$f" ]; then
      seen="$seen $r"
      echo "$(date +%H:%M) DONE $r/$g :: $(.venv/bin/python -c "import json;d=json.load(open('$f'));print('elapsed_min=%d halted=%s(%s) phases=%d'%(d['elapsed_s']/60,d['halted'],d.get('halt_reason','')[:70],len(d['completed_phases'])))" 2>&1)"
    elif [ "$(pgrep -f "run-id $r" | wc -l | tr -d ' ')" -eq 0 ] && [ -f "analysis/pooled5/work/$r"*.log ] 2>/dev/null; then
      :
    fi
  done
  n=$(echo $seen | wc -w | tr -d ' ')
  [ "$n" -ge 2 ] && { echo "$(date +%H:%M) BOTH PEOPLE RUNS TERMINAL"; break; }
  sleep 120
done

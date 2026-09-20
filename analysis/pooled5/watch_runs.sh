#!/bin/bash
# One event per POOLED-5 run reaching a TERMINAL state. Terminal means: the run wrote its
# run_summary.json, or its process is gone without one (crash). A provider content filter on a
# single researcher is NOT terminal — the pipeline designs the taxonomy without that angle and
# keeps going, so matching on the word "Error" (the first version did) reports a live run dead.
QM="/Users/mayouxuan/Documents/Claude/Search Query Mining Agent Team/QMine"
cd "$QM" || exit 1
seen=""
started=""
while true; do
  for r in fin med edu film ppl; do
    case " $seen " in *" $r "*) continue;; esac
    f="runs/$r-pool5/gen01/run_summary.json"
    if [ -f "$f" ]; then
      seen="$seen $r"
      echo "$(date +%H:%M) DONE $r-pool5 :: $(.venv/bin/python -c "import json;d=json.load(open('$f'));print('elapsed_min=%d halted=%s(%s) phases=%d'%(d['elapsed_s']/60,d['halted'],d.get('halt_reason','')[:60],len(d['completed_phases'])))" 2>&1)"
      continue
    fi
    alive=$(pgrep -f "run-id $r-pool5" | wc -l | tr -d ' ')
    case " $started " in *" $r "*) ;; *) [ "$alive" -gt 0 ] && started="$started $r";; esac
    case " $started " in
      *" $r "*) [ "$alive" -eq 0 ] && { seen="$seen $r"; echo "$(date +%H:%M) CRASHED $r-pool5 (process gone, no run_summary) :: $(grep -E 'Traceback|halt|CRITICAL' analysis/pooled5/work/$r-pool5.launch.log | tail -2 | tr '\n' ' ' | cut -c1-250)"; };;
    esac
  done
  n=$(echo $seen | wc -w | tr -d ' ')
  [ "$n" -ge 5 ] && { echo "$(date +%H:%M) ALL FIVE POOLED-5 RUNS REACHED A TERMINAL STATE"; break; }
  sleep 180
done

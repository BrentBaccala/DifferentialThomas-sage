#!/bin/bash
# Survey: stage the PDE onto cells 1..28 (cell 0 is run separately), each in an
# ISOLATED, memory-capped, time-bounded process so one cell's explosion cannot
# harm the others.  Writes ~/survey.DONE when the whole survey has finished, and
# a machine-readable ~/survey_summary.txt.  Concurrency K=2, per-cell cap 42 GB
# (address space), per-cell timeout 1 h.
set -u
cd ~/DifferentialThomas-sage/experiments || exit 1
SAGE=~/miniforge3/envs/sage/bin/sage
LOGS=~/DifferentialThomas-sage/experiments/survey_logs
mkdir -p "$LOGS"
rm -f ~/survey.DONE

seq 1 28 | xargs -P 2 -I {} bash -c '
  N={}
  cd ~/DifferentialThomas-sage/experiments
  ( ulimit -v 42000000
    timeout 3600 '"$SAGE"' -python cell_plus_pde.py --cell "$N"
  ) > '"$LOGS"'/cell_$N.log 2>&1
  rc=$?
  if ! grep -q "^SURVEYRESULT" '"$LOGS"'/cell_$N.log; then
    # process died without emitting a result line (OOM cap / timeout)
    st="OOM_OR_TIMEOUT"; [ $rc -eq 124 ] && st="TIMEOUT"
    echo "SURVEYRESULT cell=$N status=$st subcells=0 wall=? peakrss=? rc=$rc" \
      >> '"$LOGS"'/cell_$N.log
  fi
  echo "cell $N finished rc=$rc"
'

grep -h "^SURVEYRESULT" "$LOGS"/cell_*.log 2>/dev/null \
  | sort -t= -k2 -n > ~/survey_summary.txt
echo "SURVEY_ALL_DONE $(date -Iseconds)" >> ~/survey_summary.txt
touch ~/survey.DONE

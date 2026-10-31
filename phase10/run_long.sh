#!/bin/sh
# Runs a long training job, resuming from the last checkpoint after any crash. Stops for good on NaN/Inf or after too many failures.
# Run from the repo root. Usage: sh phase10/run_long.sh [config]
CONFIG=${1:-phase10/configs/foundation_v3_clean.json}
PY=${PYTHON_BIN:-~/anaconda3/envs/minigpt/bin/python}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
RETRY_SECONDS=${RETRY_SECONDS:-60}
OUT=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['out_dir'])" "$CONFIG")
LOG=${LOG:-$OUT.log}
mkdir -p "$(dirname "$LOG")"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  if [ -f "$OUT/metrics.json" ] && grep -q '"status": "failed_non_finite"' "$OUT/metrics.json"; then
    echo "$(date -Is) stopping: the run hit a non-finite value" >> "$LOG"
    exit 3
  fi
  echo "$(date -Is) attempt $attempt" >> "$LOG"
  "$PY" phase10/train_v3.py --config "$CONFIG" --auto-resume >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "$(date -Is) finished" >> "$LOG"
    exit 0
  fi
  echo "$(date -Is) attempt $attempt exited with $rc" >> "$LOG"
  attempt=$((attempt + 1))
  sleep "$RETRY_SECONDS"
done
echo "$(date -Is) giving up after $MAX_ATTEMPTS attempts" >> "$LOG"
exit 2

#!/bin/bash
# Phase 17A long run: Foundation v4. Resumes from checkpoint.pt if present (train_v3 --auto-resume).
# One instance only (flock). Retries after a crash (e.g. a driver hiccup), but never after a NaN/Inf failure.
cd "$(dirname "$0")/.." || exit 1
LOG=phase17/logs/foundation_v4.log
exec 9> phase17/runs/foundation_v4.lock
flock -n 9 || { echo "$(date '+%F %T') another instance is running; exiting" >> "$LOG"; exit 0; }
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONUNBUFFERED=1
for attempt in $(seq 1 20); do
  echo "=== $(date '+%F %T') start/resume attempt $attempt" >> "$LOG"
  python phase10/train_v3.py --config phase17/configs/foundation_v4.json --auto-resume >> "$LOG" 2>&1
  code=$?
  status=$(python -c "import json;print(json.load(open('phase17/runs/foundation_v4/metrics.json')).get('status'))" 2>/dev/null)
  echo "=== $(date '+%F %T') trainer exited with code $code, status $status" >> "$LOG"
  [ "$status" = "completed" ] && { touch phase17/runs/foundation_v4/COMPLETED; exit 0; }
  [ "$status" = "failed_non_finite" ] && { echo "=== non-finite loss: not retrying" >> "$LOG"; exit 1; }
  sleep 60
done
echo "=== $(date '+%F %T') giving up after 20 attempts" >> "$LOG"
exit 1

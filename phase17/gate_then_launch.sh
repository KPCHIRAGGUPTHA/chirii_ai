#!/bin/bash
# Phase 17A: stability gate (seeds 1 and 2, 150 steps, full config, eval every 25 steps), then the pre-registered
# decision: all gates pass -> launch the long run detached; any gate fails -> launch nothing.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONUNBUFFERED=1
LOG=phase17/logs/stability_gate.log
for seed in 1 2; do
  echo "=== $(date '+%F %T') gate seed $seed" >> "$LOG"
  python phase10/train_v3.py --config phase17/configs/foundation_v4.json --seed $seed --stop-after-step 150 --eval-interval 25 \
    --out-dir phase17/runs/stability_seed$seed >> "$LOG" 2>&1 || echo "=== seed $seed trainer exited non-zero" >> "$LOG"
done
if python phase17/check_gate.py phase17/runs/stability_seed1 phase17/runs/stability_seed2 >> "$LOG" 2>&1; then
  echo "=== $(date '+%F %T') ALL GATES PASS: launching the long run" >> "$LOG"
  mkdir -p phase17/runs/foundation_v4 && date '+%F %T' > phase17/runs/foundation_v4/LAUNCHED
  setsid nohup phase17/run_foundation_v4.sh > /dev/null 2>&1 < /dev/null &
else
  echo "=== $(date '+%F %T') GATE FAILED: nothing launched" >> "$LOG"
fi

#!/bin/sh
# Stability gate: seeds 1 and 2, 300 steps on the full run's learning-rate schedule, then a SIGKILL and resume on the full model.
PY=~/anaconda3/envs/minigpt/bin/python
C=phase10/configs/foundation_v3.json
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits -l 15 > phase10/logs/stability_nvsmi.csv &
S=$!
for seed in 1 2; do
  echo "seed$seed start $(date -Is) $(date +%s)" >> phase10/logs/stability_timing.log
  "$PY" phase10/train_v3.py --config $C --seed $seed --stop-after-step 300 --eval-interval 50 --out-dir phase10/runs/stability_seed$seed > phase10/logs/stability_seed$seed.log 2>&1
  rc=$?
  echo "seed$seed end $(date -Is) $(date +%s) rc=$rc" >> phase10/logs/stability_timing.log
done
kill $S
K=phase10/runs/kill_check
rm -rf $K
"$PY" phase10/train_v3.py --config $C --seed 1 --stop-after-step 30 --eval-interval 1000 --checkpoint-every-steps 5 --out-dir $K > phase10/logs/kill_check.log 2>&1 &
T=$!
while [ "$(wc -l < $K/steps.jsonl 2>/dev/null || echo 0)" -lt 18 ]; do sleep 0.2; kill -0 $T 2>/dev/null || break; done
kill -9 $T; wait $T 2>/dev/null
echo "=== killed with SIGKILL after $(wc -l < $K/steps.jsonl) steps" >> phase10/logs/kill_check.log
"$PY" phase10/train_v3.py --config $C --seed 1 --stop-after-step 30 --eval-interval 1000 --checkpoint-every-steps 5 --out-dir $K --auto-resume >> phase10/logs/kill_check.log 2>&1
echo "rc=$? $(date -Is)" > phase10/logs/stability.done

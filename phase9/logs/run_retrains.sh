#!/bin/sh
# Runbook section 5 on the college GPU PC: C/42, D/42, C/43, D/43. Each run starts only if the one before succeeded.
PY=~/anaconda3/envs/minigpt/bin/python
T=phase9/logs/retrain_timing.log
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits -l 30 > phase9/logs/retrain_nvsmi.csv &
SAMPLER=$!
rc=0
run() {
  echo "$1 start $(date -Is) $(date +%s)" >> "$T"
  "$PY" phase9/train_v2.py --config "phase9/configs/$2.json" > "phase9/$3" 2>&1
  rc=$?
  echo "$1 end $(date -Is) $(date +%s) rc=$rc" >> "$T"
  return $rc
}
run C_v2_s42 model_c_v2 c_v2_s42_train.log &&
run D_v2_s42 model_d_v2 d_v2_s42_train.log &&
run C_v2_s43 model_c_v2_seed43 c_v2_s43_train.log &&
run D_v2_s43 model_d_v2_seed43 d_v2_s43_train.log
echo "chain rc=$? $(date -Is)" > phase9/logs/retrains.done
kill $SAMPLER

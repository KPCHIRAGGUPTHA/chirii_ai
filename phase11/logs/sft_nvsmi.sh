#!/bin/sh
# Samples the GPU every 30 s while the SFT run is alive.
sleep 60
while pgrep -f 'phase11/sft_train.py --config' > /dev/null; do
  nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader >> phase11/logs/sft_nvsmi.csv
  sleep 30
done

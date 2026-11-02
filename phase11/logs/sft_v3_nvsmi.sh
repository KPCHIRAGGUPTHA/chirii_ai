#!/bin/sh
# Samples the GPU every 30 s while the given process is alive.
sleep 45
while kill -0 "$1" 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader >> phase11/logs/sft_v3_nvsmi.csv
  sleep 30
done

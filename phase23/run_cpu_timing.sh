#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for m in 0.5b 1.5b; do
  echo "$(date '+%F %T') cpu $m" >> phase23/logs/run_23a.status
  python phase23/cpu_timing.py --model $m >> phase23/logs/cpu_timing_$m.log 2>&1 || echo "$(date '+%F %T') cpu $m FAILED" >> phase23/logs/run_23a.status
done
echo "$(date '+%F %T') CPU_DONE" >> phase23/logs/run_23a.status

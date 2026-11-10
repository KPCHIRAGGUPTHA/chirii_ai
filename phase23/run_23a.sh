#!/bin/bash
# Phase 23A baseline: both plain Qwen models, dev sets (layer off/on) and the showcase (once), GPU float32.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
S=phase23/logs/run_23a.status
for m in 1.5b 0.5b; do
  echo "$(date '+%F %T') baseline $m" >> $S
  python phase23/run_baseline.py --model $m --dev --showcase >> phase23/logs/baseline_$m.log 2>&1 || { echo "$(date '+%F %T') baseline $m FAILED" >> $S; exit 1; }
done
echo "$(date '+%F %T') RUN_23A_DONE" >> $S

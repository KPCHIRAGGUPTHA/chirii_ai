#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p phase24/logs
echo "$(date '+%F %T') distill start" >> $PWD/phase24/logs/run_24.status
python phase24/distill.py >> phase24/logs/distill.log 2>&1 && echo "$(date '+%F %T') distill DONE" >> $PWD/phase24/logs/run_24.status || echo "$(date '+%F %T') distill FAILED" >> $PWD/phase24/logs/run_24.status

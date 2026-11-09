#!/bin/bash
# Phase 17D data collection: both sources in parallel (network-bound), low priority.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONUNBUFFERED=1
mkdir -p phase17/logs
nice -n 10 python phase17/collect_17d.py --source edu --target-bytes 24500000000 >> phase17/logs/collect_17d_edu.log 2>&1 &
nice -n 10 python phase17/collect_17d.py --source web --target-bytes 10500000000 >> phase17/logs/collect_17d_web.log 2>&1 &
wait
echo "COLLECT_17D_DONE $(date '+%F %T')" >> phase17/logs/collect_17d_edu.log

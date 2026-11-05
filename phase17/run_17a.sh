#!/bin/bash
# Phase 17A data pipeline: collect ~9.4 GB more FineWeb-Edu text, then tokenize and build train_3b.bin. No training.
# Runs detached (setsid nohup) at low priority; each step writes a .done marker. Safe to rerun: both steps resume.
set -o pipefail
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache RAYON_NUM_THREADS=8 PYTHONUNBUFFERED=1
date '+start %F %T' >> phase17/logs/run_17a.status
# The collector can abort while Python shuts down (a streaming-library thread, after the manifest is final), so
# success is judged by the manifest status, and a non-zero exit code is only recorded.
nice -n 10 python phase17/collect_more.py --target-bytes 9400000000 2>&1 | tee -a phase17/logs/collect_more.log
code=$?; [ $code -ne 0 ] && echo "collect exit code $code $(date '+%F %T')" >> phase17/logs/run_17a.status
grep -q '"status": "completed_target"' phase17/manifests/text_manifest.json || { date '+collect did not reach target %F %T' >> phase17/logs/run_17a.status; exit 1; }
date '+collect done %F %T' >> phase17/logs/run_17a.status
nice -n 10 python phase17/tokenize_more.py 2>&1 | tee -a phase17/logs/tokenize_more.log || { date '+tokenize FAILED %F %T' >> phase17/logs/run_17a.status; exit 1; }
date '+tokenize done %F %T' >> phase17/logs/run_17a.status
touch phase17/logs/run_17a.done

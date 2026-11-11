#!/bin/bash
# Phase 23C: dev generation (both models), threshold tuning on dev, then the pinned RAG test ONCE.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
S=$PWD/phase23/logs/run_23c.status
st() { echo "$(date '+%F %T') $*" >> $S; }
st "dev"; python phase23/eval_23c.py --dev >> phase23/logs/eval_23c.log 2>&1 || { st "dev FAILED"; exit 1; }
st "tune"; python phase23/eval_23c.py --tune >> phase23/logs/eval_23c.log 2>&1 || { st "tune FAILED"; exit 1; }
st "test (once)"; python phase23/eval_23c.py --test >> phase23/logs/eval_23c.log 2>&1 || { st "test FAILED"; exit 1; }
st "RUN_23C_DONE"

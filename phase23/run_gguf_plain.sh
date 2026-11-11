#!/bin/bash
# Phase 23B laptop pipeline check on plain Qwen 1.5B (dev only) + CPU timing, llama.cpp CPU build.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONPATH=$HOME/chirag_work/pylibs/phase18 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for t in Q8_0 Q4_K_M; do
  echo "$(date '+%F %T') gguf plain $t" >> phase23/logs/run_gguf.status
  python phase23/eval_gguf.py --gguf ~/chirag_work/gguf/plain_qwen1.5b_$t.gguf --name plain_$t --dev --time >> phase23/logs/eval_gguf.log 2>&1 || echo "$(date '+%F %T') gguf plain $t FAILED" >> phase23/logs/run_gguf.status
done
echo "$(date '+%F %T') GGUF_PLAIN_DONE" >> phase23/logs/run_gguf.status

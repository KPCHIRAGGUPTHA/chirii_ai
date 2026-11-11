#!/bin/bash
# Phase 23B laptop build of the merged chosen model: waits for the merge, converts to GGUF (bf16), quantizes to Q8_0 and
# Q4_K_M, then dev scoring and CPU timing for each (llama.cpp v0.9.0, CPU build).
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
S=phase23/logs/run_gguf.status
until [ -e phase23/runs/merged_chosen/model.safetensors ] && [ -e phase23/runs/merged_chosen/phase23_source.json ]; do sleep 60; done
sleep 30
echo "$(date '+%F %T') gguf convert chosen" >> $S
( cd ~/chirag_work/llama.cpp && PYTHONPATH=$HOME/chirag_work/llama.cpp/gguf-py:$HOME/chirag_work/pylibs/phase18:$HOME/chirag_work/pylibs/phase23 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
  python convert_hf_to_gguf.py ~/chirag_work/minigpt/phase23/runs/merged_chosen --outtype bf16 --outfile ~/chirag_work/gguf/chosen_bf16.gguf ) > phase23/logs/gguf_convert_chosen.log 2>&1 || { echo "$(date '+%F %T') gguf convert chosen FAILED" >> $S; exit 1; }
for t in Q8_0 Q4_K_M; do ~/chirag_work/llama.cpp/build/bin/llama-quantize ~/chirag_work/gguf/chosen_bf16.gguf ~/chirag_work/gguf/chosen_$t.gguf $t 8 > phase23/logs/gguf_quant_chosen_$t.log 2>&1 || echo "$(date '+%F %T') quant $t FAILED" >> $S; done
export PYTHONPATH=$HOME/chirag_work/pylibs/phase18 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
for t in Q8_0 Q4_K_M; do
  echo "$(date '+%F %T') gguf chosen $t" >> $S
  python phase23/eval_gguf.py --gguf ~/chirag_work/gguf/chosen_$t.gguf --name chosen_$t --dev --time >> phase23/logs/eval_gguf.log 2>&1 || echo "$(date '+%F %T') gguf chosen $t FAILED" >> $S
done
echo "$(date '+%F %T') GGUF_CHOSEN_DONE" >> $S

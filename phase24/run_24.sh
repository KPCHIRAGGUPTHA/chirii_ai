#!/bin/bash
# Phase 24 pipeline after distillation (phase24/CRITERIA_24.md): data, 3 seeds + dev, selection, merge, ONE sealed + RAG v2
# scoring, MC + GSM8K harness for the merged chosen model (plain Qwen's harness results from Phase 23B are reused: same harness,
# same settings), Q8_0 GGUF system check.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$PWD/phase24/logs/run_24.status
st() { echo "$(date '+%F %T') $*" >> $S; }
until grep -qE "distill (DONE|FAILED)" $S; do sleep 60; done
grep -q "distill FAILED" $S && exit 1
[ -e phase24/manifests/sft_24_manifest.json ] || { st "build data"; python phase24/build_sft_24.py > phase24/logs/build_sft_24.log 2>&1 || { st "build FAILED"; exit 1; }; }
st "DATA_BUILT (examples in phase24/logs/build_sft_24.log)"
until [ -e phase24/logs/data_checked ]; do sleep 60; done   # created after the build counts and examples were read
st "data checked; training"
for s in 11 12 13; do
  [ -e phase24/runs/lora_s$s/adapter/adapter_model.safetensors ] || { st "train lora_s$s"; python phase24/train_lora_v2.py --seed $s >> phase24/logs/train_lora_s$s.log 2>&1 || { st "train lora_s$s FAILED"; exit 1; }; }
  [ -e phase24/results/dev_lora_s$s.json ] || { st "dev lora_s$s"; python phase24/eval_24.py --dev lora_s$s >> phase24/logs/eval_24.log 2>&1 || { st "dev lora_s$s FAILED"; exit 1; }; }
done
python phase24/eval_24.py --select >> phase24/logs/eval_24.log 2>&1 || { st "select FAILED"; exit 1; }
[ -e phase24/runs/merged_chosen/model.safetensors ] || { st "merge"; python phase24/eval_24.py --merge >> phase24/logs/eval_24.log 2>&1 || { st "merge FAILED"; exit 1; }; }
[ -e phase24/results/sealed.json ] || { st "sealed + RAG v2 (once)"; python phase24/eval_24.py --sealed >> phase24/logs/eval_24.log 2>&1 || { st "sealed FAILED"; exit 1; }; }
python phase23/make_local_model.py phase24/runs/merged_chosen phase18/local_models/p24_merged_chosen >> phase24/logs/eval_24.log 2>&1
cd phase18
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
R=results/p24_merged_chosen; mkdir -p $R
[ -e $R/mc.done ] || { st "harness mc"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/p24_merged_chosen,dtype=float32" --tasks $MC --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase24/logs/harness_24.log 2>&1 && touch $R/mc.done || st "harness mc FAILED"; }
[ -e $R/gsm8k.done ] || { st "harness gsm8k"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/p24_merged_chosen,dtype=float32" --tasks gsm8k --samples "$SAMPLES" --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $R/gsm8k >> ../phase24/logs/harness_24.log 2>&1 && touch $R/gsm8k.done || st "harness gsm8k FAILED"; }
python subset_scores.py >> ../phase24/logs/harness_24.log 2>&1
cd ..
st "gguf convert"
( cd ~/chirag_work/llama.cpp && PYTHONPATH=$HOME/chirag_work/llama.cpp/gguf-py:$HOME/chirag_work/pylibs/phase18:$HOME/chirag_work/pylibs/phase23 CUDA_VISIBLE_DEVICES="" python convert_hf_to_gguf.py ~/chirag_work/minigpt/phase24/runs/merged_chosen --outtype bf16 --outfile ~/chirag_work/gguf/p24_chosen_bf16.gguf ) > phase24/logs/gguf_convert.log 2>&1 \
  && ~/chirag_work/llama.cpp/build/bin/llama-quantize ~/chirag_work/gguf/p24_chosen_bf16.gguf ~/chirag_work/gguf/p24_chosen_Q8_0.gguf Q8_0 8 > phase24/logs/gguf_quant.log 2>&1 || { st "gguf FAILED"; exit 1; }
st "gguf system dev + cpu"
PYTHONPATH=$HOME/chirag_work/pylibs/phase18 CUDA_VISIBLE_DEVICES="" python phase24/eval_gguf_24.py >> phase24/logs/eval_gguf_24.log 2>&1 || st "gguf eval FAILED"
st "RUN_24_DONE"

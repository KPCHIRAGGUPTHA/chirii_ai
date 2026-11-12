#!/bin/bash
# Phase 25 (evaluation only): GPU steps, then the harness, then the laptop check on CPU. Each step runs once.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
S=$PWD/phase25/logs/run_25.status; mkdir -p phase25/logs
st() { echo "$(date '+%F %T') $*" >> $S; }
export HF_HUB_OFFLINE=0   # TruthfulQA is downloaded once (pinned revision in eval_25.py)
for step in safety squad tqa authored; do
  [ -e phase25/results/$( case $step in safety) echo safety;; squad) echo squad_docmode;; tqa) echo tqa_letter_bf16;; authored) echo authored_reported;; esac ).json ] && continue
  st "$step"; python phase25/eval_25.py --$step >> phase25/logs/eval_25.log 2>&1 || { st "$step FAILED"; exit 1; }
done
cd phase18
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
for name in qwen2_5_1_5b_instruct p24_merged_chosen; do
  R=../phase25/harness/$name; mkdir -p $R
  [ -e $R/tqa.done ] || { st "harness truthfulqa $name"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks truthfulqa_mc1,truthfulqa_mc2 --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/tqa >> ../phase25/logs/harness_25.log 2>&1 && touch $R/tqa.done || st "harness truthfulqa $name FAILED"; }
  [ -e $R/gsm8k_full.done ] || { st "harness gsm8k full $name"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks gsm8k --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --num_fewshot 0 --device cuda --seed 1234 --output_path $R/gsm8k_full >> ../phase25/logs/harness_25.log 2>&1 && touch $R/gsm8k_full.done || st "harness gsm8k $name FAILED"; }
  [ -e $R/triviaqa.done ] || { st "harness triviaqa $name"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks triviaqa --samples "$SAMPLES" --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $R/triviaqa >> ../phase25/logs/harness_25.log 2>&1 && touch $R/triviaqa.done || st "harness triviaqa $name FAILED"; }
done
cd ..
[ -e phase25/results/gguf_q8_0.json ] || { st "laptop gguf"; CUDA_VISIBLE_DEVICES="" PYTHONPATH=$HOME/chirag_work/pylibs/phase18 python phase25/eval_gguf_25.py >> phase25/logs/eval_gguf_25.log 2>&1 || st "laptop gguf FAILED"; }
st "RUN_25_DONE"

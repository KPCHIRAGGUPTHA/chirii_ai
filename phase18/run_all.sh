#!/bin/bash
# Phase 18 — every model on the same tasks, 0-shot, lm-evaluation-harness 0.4.13. Resumable: each finished step
# leaves a .done marker and is skipped on rerun.
#
# Prompt rules (identical for every model):
#   - The 10 multiple-choice / loglikelihood tasks use lm-eval's plain prompts for every model, base or instruct.
#   - GSM8K and TriviaQA (generative, exact match): base models get the plain prompt; instruct models get their own
#     chat format (--apply_chat_template for Hugging Face models; MiniGPT v1.0.1's chat runtime). MiniGPT v1.0.1 runs
#     GSM8K twice: with its calculator tool and with the calculator blocked.
#   - Greedy decoding everywhere. Float32 everywhere.
set -u
cd "$(dirname "$0")"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
ROOT=$HOME/chirag_work/minigpt
COMMON="--num_fewshot 0 --device cuda --seed 1234"

step() {  # step <name> <args...>
  local name=$1; shift
  [ -e "results/$name.done" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date '+%F %T')"
  if python lm_eval_cli.py "$@" $COMMON --output_path "results/$name"; then
    touch "results/$name.done"; echo "=== done $name $(date '+%F %T')"
  else
    echo "=== FAILED $name $(date '+%F %T')"
  fi
}

# Generative tasks: the fixed subsets in subsets.json (same items for every model), and generation settings passed
# explicitly so no model's shipped generation_config can change them (Qwen2.5-0.5B ships max_new_tokens=2048, and
# Qwen2.5-0.5B-Instruct ships repetition_penalty=1.1). See BENCHMARKS.md.
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
GSM="--gen_kwargs max_gen_toks=256,repetition_penalty=1.0"
TQA="--gen_kwargs max_gen_toks=64,repetition_penalty=1.0"

minigpt() {  # minigpt <name> <checkpoint> <chat_generation>
  local name=$1 ckpt=$2 chat=$3
  local base="checkpoint=$ROOT/$ckpt"
  step "$name/mc" --model minigpt --model_args "$base,truncation_log=results/$name/truncation_mc.json" --tasks $MC --batch_size 32
  if [ "$chat" = "True" ]; then
    step "$name/gsm8k" --model minigpt --model_args "$base,chat_generation=True,calculator=True,truncation_log=results/$name/truncation_gsm8k.json" --tasks gsm8k --samples "$SAMPLES" $GSM --log_samples
    step "$name/gsm8k_no_calculator" --model minigpt --model_args "$base,chat_generation=True,calculator=False,truncation_log=results/$name/truncation_gsm8k_no_calculator.json" --tasks gsm8k --samples "$SAMPLES" $GSM --log_samples
    step "$name/triviaqa" --model minigpt --model_args "$base,chat_generation=True,calculator=True,truncation_log=results/$name/truncation_triviaqa.json" --tasks triviaqa --samples "$SAMPLES" $TQA --log_samples
  else
    step "$name/gsm8k" --model minigpt --model_args "$base,truncation_log=results/$name/truncation_gsm8k.json" --tasks gsm8k --samples "$SAMPLES" $GSM --log_samples
    step "$name/triviaqa" --model minigpt --model_args "$base,truncation_log=results/$name/truncation_triviaqa.json" --tasks triviaqa --samples "$SAMPLES" $TQA --log_samples
  fi
}

hf() {  # hf <name> <repo> <revision> <instruct:yes|no>
  local name=$1 repo=$2 rev=$3 instruct=$4
  local args="pretrained=$repo,revision=$rev,dtype=float32"
  # Qwen: local copies of the pinned snapshots (weights/tokenizer symlinked) whose generation_config.json keeps only
  # the token ids, so the shipped max_new_tokens=2048 / repetition_penalty=1.1 / sampling defaults cannot apply.
  [[ "$repo" == local_models/* ]] && args="pretrained=$repo,dtype=float32"
  local chat=""
  [ "$instruct" = "yes" ] && chat="--apply_chat_template"
  step "$name/mc" --model hf --model_args "$args" --tasks $MC --batch_size auto:4
  step "$name/gsm8k" --model hf --model_args "$args" --tasks gsm8k --samples "$SAMPLES" $GSM --batch_size 16 $chat --log_samples
  step "$name/triviaqa" --model hf --model_args "$args" --tasks triviaqa --samples "$SAMPLES" $TQA --batch_size 32 $chat --log_samples
}

minigpt minigpt_base   phase10/runs/foundation_v3_clean_2ep/snapshots/model_step_11524.pt False
minigpt minigpt_v1_0_1 phase11/runs/sft_v2/final_model.pt                                  True
hf gpt2                  openai-community/gpt2                 607a30d783dfa663caf39e06633721c8d4cfcd7e no
hf smollm2_135m          HuggingFaceTB/SmolLM2-135M            93efa2f097d58c2a74874c7e644dbc9b0cee75a2 no
hf smollm2_135m_instruct HuggingFaceTB/SmolLM2-135M-Instruct   12fd25f77366fa6b3b4b768ec3050bf629380bac yes
hf smollm2_360m          HuggingFaceTB/SmolLM2-360M            f8027fd0eaeea54caa13c31d31b9fdc459c38b49 no
hf smollm2_360m_instruct HuggingFaceTB/SmolLM2-360M-Instruct   a10cc1512eabd3dde888204e902eca88bddb4951 yes
hf qwen2_5_0_5b          local_models/qwen2_5_0_5b             060db6499f32faf8b98477b0a26969ef7d8b9987 no
hf qwen2_5_0_5b_instruct local_models/qwen2_5_0_5b_instruct    7ae557604adf67be50417f59c2c2f167def9a775 yes
echo ALL_DONE $(date '+%F %T')

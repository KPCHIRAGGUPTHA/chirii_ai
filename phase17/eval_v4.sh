#!/bin/bash
# Phase 17B GPU jobs, in order: bits per byte on the same texts, fixed-prompt generations, then the Phase 18 harness on
# the v4 base model with exactly Phase 18's MiniGPT-base commands (plain prompts; GSM8K/TriviaQA on the fixed subsets).
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONUNBUFFERED=1
LOG=phase17/logs/eval_v4.log
echo "=== $(date '+%F %T') bits per byte" >> $LOG
python phase17/eval_bpb.py >> $LOG 2>&1 || echo "=== FAILED bpb" >> $LOG
echo "=== $(date '+%F %T') generations" >> $LOG
python phase17/generate_fixed.py >> $LOG 2>&1 || echo "=== FAILED generations" >> $LOG
cd phase18
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
CK="checkpoint=$HOME/chirag_work/minigpt/phase17/runs/foundation_v4/snapshots/model_step_05722.pt"
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
COMMON="--num_fewshot 0 --device cuda --seed 1234"
R=results/minigpt_v4_base
mkdir -p $R
run() { local step=$1; shift; echo "=== $(date '+%F %T') harness $step" >> ../$LOG
  if python lm_eval_cli.py "$@" $COMMON --output_path $R/$step >> ../$LOG 2>&1; then touch $R/$step.done; else echo "=== FAILED $step" >> ../$LOG; fi; }
run mc --model minigpt --model_args "$CK,truncation_log=$R/truncation_mc.json" --tasks $MC --batch_size 16
run gsm8k --model minigpt --model_args "$CK,truncation_log=$R/truncation_gsm8k.json" --tasks gsm8k --samples "$SAMPLES" --gen_kwargs max_gen_toks=256,repetition_penalty=1.0 --log_samples
run triviaqa --model minigpt --model_args "$CK,truncation_log=$R/truncation_triviaqa.json" --tasks triviaqa --samples "$SAMPLES" --gen_kwargs max_gen_toks=64,repetition_penalty=1.0 --log_samples
echo "=== $(date '+%F %T') EVAL_V4_DONE" >> ../$LOG

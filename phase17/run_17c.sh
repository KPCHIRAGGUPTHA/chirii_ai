#!/bin/bash
# Phase 17C (CRITERIA_17C.md): 3 SFT v5 seeds on the 17D final model, dev scoring, selection, ONE sealed v3 scoring,
# then GSM8K (500, with and without calculator) and the Phase 18 multiple-choice tasks for the chosen seed.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONUNBUFFERED=1
S=phase17/logs/run_17c.status
echo "$(date '+%F %T') waiting for the 17D evaluation to free the GPU" >> $S
until grep -q EVAL_V4EXT_DONE phase17/logs/eval_v4ext.log; do sleep 30; done
for seed in 11 12 13; do
  echo "$(date '+%F %T') train seed $seed" >> $S
  python phase11/sft_train.py --config phase17/configs/sft_v5_s$seed.json --auto-resume >> phase17/logs/sft_v5_s$seed.log 2>&1 || { echo "$(date '+%F %T') train seed $seed FAILED" >> $S; exit 1; }
  echo "$(date '+%F %T') dev seed $seed" >> $S
  python phase17/eval_17c.py --dev $seed >> phase17/logs/eval_17c.log 2>&1 || { echo "$(date '+%F %T') dev seed $seed FAILED" >> $S; exit 1; }
done
python phase17/eval_17c.py --select >> phase17/logs/eval_17c.log 2>&1 || { echo "$(date '+%F %T') select FAILED" >> $S; exit 1; }
echo "$(date '+%F %T') sealed v3 (once)" >> $S
python phase17/eval_17c.py --sealed >> phase17/logs/eval_17c.log 2>&1 || { echo "$(date '+%F %T') sealed FAILED" >> $S; exit 1; }
SEED=$(python -c "import json;print(json.load(open('phase17/results/phase17c/selection.json'))['chosen_seed'])")
CK="checkpoint=$HOME/chirag_work/minigpt/phase17/runs/sft_v5_s$SEED/final_model.pt"
cd phase18
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
COMMON="--num_fewshot 0 --device cuda --seed 1234"
R=results/sft_v5_chosen
mkdir -p $R
run() { local step=$1; shift; echo "$(date '+%F %T') harness $step" >> ../$S
  if python lm_eval_cli.py "$@" $COMMON --output_path $R/$step >> ../phase17/logs/harness_17c.log 2>&1; then touch $R/$step.done; else echo "$(date '+%F %T') harness $step FAILED" >> ../$S; fi; }
run gsm8k --model minigpt --model_args "$CK,chat_generation=True,calculator=True,no_repeat_ngram=3,truncation_log=$R/truncation_gsm8k.json" --tasks gsm8k --samples "$SAMPLES" --gen_kwargs max_gen_toks=256,repetition_penalty=1.0 --log_samples
run gsm8k_no_calculator --model minigpt --model_args "$CK,chat_generation=True,calculator=False,no_repeat_ngram=3,truncation_log=$R/truncation_gsm8k_no_calculator.json" --tasks gsm8k --samples "$SAMPLES" --gen_kwargs max_gen_toks=256,repetition_penalty=1.0 --log_samples
run mc --model minigpt --model_args "$CK,truncation_log=$R/truncation_mc.json" --tasks $MC --batch_size 16
echo "$(date '+%F %T') RUN_17C_DONE (chosen seed $SEED)" >> ../$S

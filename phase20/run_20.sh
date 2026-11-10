#!/bin/bash
# Phase 20 (phase20/CRITERIA_20.md + amendment 1): 6 SFT runs, dev scoring, selection, ONE sealed v4 scoring, forgetting
# (Phase 18 five tasks for all 6 runs) and GSM8K (500, calculator) for the chosen run. No sealed text is printed.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONUNBUFFERED=1
S=phase20/logs/run_20.status
for run in A_s11 A_s12 A_s13 B_s11 B_s12 B_s13; do
  if [ ! -e phase20/runs/$run/final_model.pt ]; then
    echo "$(date '+%F %T') train $run" >> $S
    python phase11/sft_train.py --config phase20/configs/$run.json --auto-resume >> phase20/logs/$run.log 2>&1 || { echo "$(date '+%F %T') train $run FAILED" >> $S; exit 1; }
  fi
  if [ ! -e phase20/results/eval/dev_$run.json ]; then
    echo "$(date '+%F %T') dev $run" >> $S
    python phase20/eval_20.py --dev $run >> phase20/logs/eval_20.log 2>&1 || { echo "$(date '+%F %T') dev $run FAILED" >> $S; exit 1; }
  fi
done
python phase20/eval_20.py --select >> phase20/logs/eval_20.log 2>&1 || { echo "$(date '+%F %T') select FAILED" >> $S; exit 1; }
if [ ! -e phase20/results/eval/sealed.json ]; then
  echo "$(date '+%F %T') sealed v4 (once)" >> $S
  python phase20/eval_20.py --sealed >> phase20/logs/eval_20.log 2>&1 || { echo "$(date '+%F %T') sealed FAILED" >> $S; exit 1; }
fi
CHOSEN=$(python -c "import json;print(json.load(open('phase20/results/eval/selection.json'))['chosen_run'])")
cd phase18
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
MC="arc_easy,hellaswag,piqa,sciq,lambada_openai"
for run in A_s11 A_s12 A_s13 B_s11 B_s12 B_s13; do
  R=results/p20_$run; mkdir -p $R
  [ -e $R/mc.done ] && continue
  echo "$(date '+%F %T') harness five-task $run" >> ../$S
  if python lm_eval_cli.py --model minigpt --model_args "checkpoint=$HOME/chirag_work/minigpt/phase20/runs/$run/final_model.pt,truncation_log=$R/truncation_mc.json" \
       --tasks $MC --batch_size 16 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase20/logs/harness_20.log 2>&1; then touch $R/mc.done; else echo "$(date '+%F %T') harness $run FAILED" >> ../$S; fi
done
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
R=results/p20_$CHOSEN
if [ ! -e $R/gsm8k.done ]; then
  echo "$(date '+%F %T') harness gsm8k $CHOSEN" >> ../$S
  if python lm_eval_cli.py --model minigpt --model_args "checkpoint=$HOME/chirag_work/minigpt/phase20/runs/$CHOSEN/final_model.pt,chat_generation=True,calculator=True,no_repeat_ngram=3,truncation_log=$R/truncation_gsm8k.json" \
       --tasks gsm8k --samples "$SAMPLES" --gen_kwargs max_gen_toks=256,repetition_penalty=1.0 --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $R/gsm8k >> ../phase20/logs/harness_20.log 2>&1; then touch $R/gsm8k.done; else echo "$(date '+%F %T') harness gsm8k FAILED" >> ../$S; fi
fi
echo "$(date '+%F %T') RUN_20_DONE" >> ../$S

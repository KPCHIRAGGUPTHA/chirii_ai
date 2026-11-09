#!/bin/bash
# Phase 17E (CRITERIA_17E.md, research only, no sealed sets): 3 replay SFT seeds + dev, then the Phase 18 multiple-choice
# tasks for the replay seeds and for 17C seeds 11 and 12 (17C seed 13 is already in phase18/results/sft_v5_chosen).
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTHONUNBUFFERED=1
S=phase17/logs/run_17e.status
for seed in 11 12 13; do
  echo "$(date '+%F %T') train replay seed $seed" >> $S
  python phase11/sft_train.py --config phase17/configs/sft_v5r_s$seed.json --auto-resume >> phase17/logs/sft_v5r_s$seed.log 2>&1 || { echo "$(date '+%F %T') train replay seed $seed FAILED" >> $S; exit 1; }
  echo "$(date '+%F %T') dev replay seed $seed" >> $S
  python phase17/eval_17c.py --dev $seed --run-prefix sft_v5r_s --tag _replay >> phase17/logs/eval_17e.log 2>&1 || { echo "$(date '+%F %T') dev replay seed $seed FAILED" >> $S; exit 1; }
done
cd phase18
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
for run in sft_v5r_s11 sft_v5r_s12 sft_v5r_s13 sft_v5_s11 sft_v5_s12; do
  R=results/$run; mkdir -p $R
  [ -e $R/mc.done ] && continue
  echo "$(date '+%F %T') harness mc $run" >> ../$S
  if python lm_eval_cli.py --model minigpt --model_args "checkpoint=$HOME/chirag_work/minigpt/phase17/runs/$run/final_model.pt,truncation_log=$R/truncation_mc.json" \
       --tasks $MC --batch_size 16 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase17/logs/harness_17e.log 2>&1; then touch $R/mc.done; else echo "$(date '+%F %T') harness $run FAILED" >> ../$S; fi
done
echo "$(date '+%F %T') RUN_17E_DONE" >> ../$S

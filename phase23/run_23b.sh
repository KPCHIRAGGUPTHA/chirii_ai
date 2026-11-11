#!/bin/bash
# Phase 23B on the GPU (phase23/CRITERIA_23.md addition 1): plain re-score at 512, 3 LoRA seeds + dev, selection, merge,
# ONE sealed scoring, then the multiple-choice check and GSM8K (500) for plain Qwen 1.5B and the merged chosen model.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
S=$PWD/phase23/logs/run_23b.status  # absolute path: the harness steps run after "cd phase18" (fixed after the 23B run, whose harness status lines were lost)
st() { echo "$(date '+%F %T') $*" >> $S; }
[ -e phase23/results/b/dev_plain.json ] || { st "dev plain"; python phase23/eval_23b.py --dev plain >> phase23/logs/eval_23b.log 2>&1 || { st "dev plain FAILED"; exit 1; }; }
for s in 11 12 13; do
  [ -e phase23/runs/lora_s$s/adapter/adapter_model.safetensors ] || { st "train lora_s$s"; python phase23/train_lora.py --seed $s >> phase23/logs/train_lora_s$s.log 2>&1 || { st "train lora_s$s FAILED"; exit 1; }; }
  [ -e phase23/results/b/dev_lora_s$s.json ] || { st "dev lora_s$s"; python phase23/eval_23b.py --dev lora_s$s >> phase23/logs/eval_23b.log 2>&1 || { st "dev lora_s$s FAILED"; exit 1; }; }
done
python phase23/eval_23b.py --select >> phase23/logs/eval_23b.log 2>&1 || { st "select FAILED"; exit 1; }
[ -e phase23/runs/merged_chosen/model.safetensors ] || { st "merge"; python phase23/eval_23b.py --merge >> phase23/logs/eval_23b.log 2>&1 || { st "merge FAILED"; exit 1; }; }
[ -e phase23/results/b/sealed.json ] || { st "sealed (once)"; python phase23/eval_23b.py --sealed >> phase23/logs/eval_23b.log 2>&1 || { st "sealed FAILED"; exit 1; }; }
python phase23/make_local_model.py phase23/runs/merged_chosen phase18/local_models/p23_merged_chosen >> phase23/logs/eval_23b.log 2>&1
cd phase18
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
for name in qwen2_5_1_5b_instruct p23_merged_chosen; do
  R=results/$name; mkdir -p $R
  [ -e $R/mc.done ] || { st "harness mc $name"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks $MC --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase23/logs/harness_23b.log 2>&1 && touch $R/mc.done || st "harness mc $name FAILED"; }
  [ -e $R/gsm8k.done ] || { st "harness gsm8k $name"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks gsm8k --samples "$SAMPLES" --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $R/gsm8k >> ../phase23/logs/harness_23b.log 2>&1 && touch $R/gsm8k.done || st "harness gsm8k $name FAILED"; }
done
python subset_scores.py >> ../phase23/logs/harness_23b.log 2>&1
cd ..
st "RUN_23B_DONE"

#!/bin/bash
# Phase 26 pipeline (phase26/CRITERIA_26.md): distil SciQ/BoolQ train, build data, (data check), 3 seeds + dev, plain dev reference,
# selection, merge, the ONE test run (safety, SQuAD document mode, TruthfulQA letter, authored), harness for the v0.2 model
# (MC, TruthfulQA, GSM8K full, TriviaQA subset), Q8_0 GGUF laptop check. Each step runs once.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$PWD/phase26/logs/run_26.status; mkdir -p phase26/logs
st() { echo "$(date '+%F %T') $*" >> $S; }
[ -e phase26/distill/boolq.jsonl ] || { st "distill sciq/boolq"; python phase26/distill_26.py > phase26/logs/distill_26.log 2>&1 || { st "distill FAILED"; exit 1; }; }
[ -e phase26/manifests/sft_26_manifest.json ] || { st "build data"; python phase26/build_sft_26.py > phase26/logs/build_sft_26.log 2>&1 || { st "build FAILED"; exit 1; }; }
st "DATA_BUILT (counts and examples in phase26/logs/build_sft_26.log)"
until [ -e phase26/logs/data_checked ]; do sleep 60; done   # created after the build counts and examples were read
st "data checked; training"
for s in 11 12 13; do
  [ -e phase26/runs/lora_s$s/adapter/adapter_model.safetensors ] || { st "train lora_s$s"; python phase26/train_lora_26.py --seed $s >> phase26/logs/train_lora_s$s.log 2>&1 || { st "train lora_s$s FAILED"; exit 1; }; }
  [ -e phase26/results/dev_lora_s$s.json ] || { st "dev lora_s$s"; python phase26/eval_26.py --dev lora_s$s >> phase26/logs/eval_26.log 2>&1 || { st "dev lora_s$s FAILED"; exit 1; }; }
done
[ -e phase26/results/dev_plain.json ] || { st "dev plain (reference)"; python phase26/eval_26.py --dev plain >> phase26/logs/eval_26.log 2>&1 || { st "dev plain FAILED"; exit 1; }; }
python phase26/eval_26.py --select >> phase26/logs/eval_26.log 2>&1 || { st "select FAILED"; exit 1; }
[ -e phase26/runs/merged_chosen/model.safetensors ] || { st "merge"; python phase26/eval_26.py --merge >> phase26/logs/eval_26.log 2>&1 || { st "merge FAILED"; exit 1; }; }
export HF_HUB_OFFLINE=0   # TruthfulQA (pinned revision) as in Phase 25
for step in safety squad tqa authored; do
  f=$( case $step in safety) echo safety;; squad) echo squad_docmode;; tqa) echo tqa_letter_bf16;; authored) echo authored_reported;; esac )
  [ -e phase26/results/$f.json ] && continue
  st "$step (once)"; python phase26/eval_26.py --$step >> phase26/logs/eval_26.log 2>&1 || { st "$step FAILED"; exit 1; }
done
python phase23/make_local_model.py phase26/runs/merged_chosen phase18/local_models/p26_merged_chosen >> phase26/logs/eval_26.log 2>&1
cd phase18
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
name=p26_merged_chosen; R=results/$name; H=../phase26/harness/$name; mkdir -p $R $H
[ -e $R/mc.done ] || { st "harness mc"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks $MC --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase26/logs/harness_26.log 2>&1 && touch $R/mc.done || st "harness mc FAILED"; }
[ -e $H/tqa.done ] || { st "harness truthfulqa"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks truthfulqa_mc1,truthfulqa_mc2 --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $H/tqa >> ../phase26/logs/harness_26.log 2>&1 && touch $H/tqa.done || st "harness truthfulqa FAILED"; }
[ -e $H/gsm8k_full.done ] || { st "harness gsm8k full"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks gsm8k --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --num_fewshot 0 --device cuda --seed 1234 --output_path $H/gsm8k_full >> ../phase26/logs/harness_26.log 2>&1 && touch $H/gsm8k_full.done || st "harness gsm8k FAILED"; }
[ -e $H/triviaqa.done ] || { st "harness triviaqa"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks triviaqa --samples "$SAMPLES" --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $H/triviaqa >> ../phase26/logs/harness_26.log 2>&1 && touch $H/triviaqa.done || st "harness triviaqa FAILED"; }
cd ..
if [ ! -e ~/chirag_work/gguf/p26_chosen_Q8_0.gguf ]; then
  st "gguf convert"
  ( cd ~/chirag_work/llama.cpp && PYTHONPATH=$HOME/chirag_work/llama.cpp/gguf-py:$HOME/chirag_work/pylibs/phase18:$HOME/chirag_work/pylibs/phase23 CUDA_VISIBLE_DEVICES="" python convert_hf_to_gguf.py ~/chirag_work/minigpt/phase26/runs/merged_chosen --outtype bf16 --outfile ~/chirag_work/gguf/p26_chosen_bf16.gguf ) > phase26/logs/gguf_convert.log 2>&1 \
    && ~/chirag_work/llama.cpp/build/bin/llama-quantize ~/chirag_work/gguf/p26_chosen_bf16.gguf ~/chirag_work/gguf/p26_chosen_Q8_0.gguf Q8_0 8 > phase26/logs/gguf_quant.log 2>&1 || { st "gguf FAILED"; exit 1; }
fi
[ -e phase26/results/gguf_q8_0.json ] || { st "laptop gguf"; CUDA_VISIBLE_DEVICES="" PYTHONPATH=$HOME/chirag_work/pylibs/phase18 python phase26/eval_gguf_26.py >> phase26/logs/eval_gguf_26.log 2>&1 || st "laptop gguf FAILED"; }
python phase26/report_26.py >> phase26/logs/report_26.log 2>&1 || st "report FAILED"
st "RUN_26_DONE"

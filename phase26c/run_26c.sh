#!/bin/bash
# Phase 26C pipeline (phase26c/CRITERIA_26C.md + Amendment 1): build the dev sets, build the pool and the 4 mixtures, (data check),
# dev reference rows (plain, v0.2, v0.3), mixture sweep N in {0,500,1000,2000} at seed 11 + dev, mixture selection, seeds 12/13 of the
# chosen mixture + dev, seed selection, merge, the ONE test run (safety, HarmBench, SQuAD document mode, TruthfulQA letter, authored),
# harness for the chosen model (MC, TruthfulQA, GSM8K full, TriviaQA subset), Q8_0 GGUF laptop check, report. Each step runs once.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$PWD/phase26c/logs/run_26c.status; mkdir -p phase26c/logs
st() { echo "$(date '+%F %T') $*" >> $S; }
[ -e phase26c/manifests/dev_26c_manifest.json ] || { st "build dev sets"; python phase26c/build_dev_26c.py > phase26c/logs/build_dev_26c.log 2>&1 || { st "build dev FAILED"; exit 1; }; }
[ -e phase26c/manifests/sft_26c_n2000_manifest.json ] || { st "build data"; python phase26c/build_sft_26c.py > phase26c/logs/build_sft_26c.log 2>&1 || { st "build FAILED"; exit 1; }; }
st "DATA_BUILT (counts and example replies in phase26c/logs/build_sft_26c.log)"
until [ -e phase26c/logs/data_checked ]; do sleep 60; done   # created after the build counts and example replies were read
st "data checked"
for r in plain v0_2 v0_3; do
  [ -e phase26c/results/dev_$r.json ] || { st "dev reference $r"; python phase26c/eval_26c.py --dev $r >> phase26c/logs/eval_26c.log 2>&1 || { st "dev $r FAILED"; exit 1; }; }
done
train_dev() {  # $1 = N, $2 = seed
  local nm=n$1_s$2
  [ -e phase26c/runs/$nm/adapter/adapter_model.safetensors ] || { st "train $nm"; python phase26c/train_lora_26c.py --n $1 --seed $2 >> phase26c/logs/train_$nm.log 2>&1 || { st "train $nm FAILED"; exit 1; }; }
  [ -e phase26c/results/dev_$nm.json ] || { st "dev $nm"; python phase26c/eval_26c.py --dev $nm >> phase26c/logs/eval_26c.log 2>&1 || { st "dev $nm FAILED"; exit 1; }; }
}
for n in 0 500 1000 2000; do train_dev $n 11; done
python phase26c/eval_26c.py --select_mix >> phase26c/logs/eval_26c.log 2>&1 || { st "select_mix FAILED"; exit 1; }
N=$(python -c "import json;print(json.load(open('phase26c/results/selection_mix.json'))['chosen_mixture'])")
st "mixture chosen: N=$N"
for s in 12 13; do train_dev $N $s; done
python phase26c/eval_26c.py --select_seed >> phase26c/logs/eval_26c.log 2>&1 || { st "select_seed FAILED"; exit 1; }
st "seed chosen: $(python -c "import json;print(json.load(open('phase26c/results/selection.json'))['chosen'])")"
[ -e phase26c/runs/merged_chosen/model.safetensors ] || { st "merge"; python phase26c/eval_26c.py --merge >> phase26c/logs/eval_26c.log 2>&1 || { st "merge FAILED"; exit 1; }; }
export HF_HUB_OFFLINE=0   # TruthfulQA (pinned revision) as in Phase 25/26
for step in safety harmbench squad tqa authored; do
  f=$( case $step in safety) echo safety;; harmbench) echo harmbench;; squad) echo squad_docmode;; tqa) echo tqa_letter_bf16;; authored) echo authored_reported;; esac )
  [ -e phase26c/results/$f.json ] && continue
  st "$step (once)"; python phase26c/eval_26c.py --$step >> phase26c/logs/eval_26c.log 2>&1 || { st "$step FAILED"; exit 1; }
done
python phase23/make_local_model.py phase26c/runs/merged_chosen phase18/local_models/p26c_merged_chosen >> phase26c/logs/eval_26c.log 2>&1
cd phase18
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
name=p26c_merged_chosen; R=results/$name; H=../phase26c/harness/$name; mkdir -p $R $H
[ -e $R/mc.done ] || { st "harness mc"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks $MC --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase26c/logs/harness_26c.log 2>&1 && touch $R/mc.done || st "harness mc FAILED"; }
[ -e $H/tqa.done ] || { st "harness truthfulqa"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks truthfulqa_mc1,truthfulqa_mc2 --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $H/tqa >> ../phase26c/logs/harness_26c.log 2>&1 && touch $H/tqa.done || st "harness truthfulqa FAILED"; }
[ -e $H/gsm8k_full.done ] || { st "harness gsm8k full"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks gsm8k --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --num_fewshot 0 --device cuda --seed 1234 --output_path $H/gsm8k_full >> ../phase26c/logs/harness_26c.log 2>&1 && touch $H/gsm8k_full.done || st "harness gsm8k FAILED"; }
[ -e $H/triviaqa.done ] || { st "harness triviaqa"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks triviaqa --samples "$SAMPLES" --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $H/triviaqa >> ../phase26c/logs/harness_26c.log 2>&1 && touch $H/triviaqa.done || st "harness triviaqa FAILED"; }
cd ..
if [ ! -e ~/chirag_work/gguf/p26c_chosen_Q8_0.gguf ]; then
  st "gguf convert"
  ( cd ~/chirag_work/llama.cpp && PYTHONPATH=$HOME/chirag_work/llama.cpp/gguf-py:$HOME/chirag_work/pylibs/phase18:$HOME/chirag_work/pylibs/phase23 CUDA_VISIBLE_DEVICES="" python convert_hf_to_gguf.py ~/chirag_work/minigpt/phase26c/runs/merged_chosen --outtype bf16 --outfile ~/chirag_work/gguf/p26c_chosen_bf16.gguf ) > phase26c/logs/gguf_convert.log 2>&1 \
    && ~/chirag_work/llama.cpp/build/bin/llama-quantize ~/chirag_work/gguf/p26c_chosen_bf16.gguf ~/chirag_work/gguf/p26c_chosen_Q8_0.gguf Q8_0 8 > phase26c/logs/gguf_quant.log 2>&1 || { st "gguf FAILED"; exit 1; }
fi
[ -e phase26c/results/gguf_q8_0.json ] || { st "laptop gguf"; CUDA_VISIBLE_DEVICES="" PYTHONPATH=$HOME/chirag_work/pylibs/phase18 python phase26c/eval_gguf_26c.py >> phase26c/logs/eval_gguf_26c.log 2>&1 || st "laptop gguf FAILED"; }
python phase26c/report_26c.py >> phase26c/logs/report_26c.log 2>&1 || st "report FAILED"
st "RUN_26C_DONE"

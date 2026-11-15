#!/bin/bash
# Phase 26D pipeline (phase26d/CRITERIA_26D.md): build the harm dev set, dev reference rows (plain, v0.2, v0.3.1) BEFORE training,
# pair candidates, replies for the pairs (v0.3.1 on (a); v0.3.1, plain, v0.3 on (b)), pairs, (data + saturation check), DPO seeds
# 11/12/13 + dev, seed selection, merge, the ONE test run (safety, HarmBench, SQuAD document mode, TruthfulQA letter, authored),
# harness for v0.4 (MC, TruthfulQA, GSM8K full, TriviaQA subset), Q8_0 GGUF laptop check, report. Each step runs once.
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/chirag_work/pylibs/phase23:$HOME/chirag_work/pylibs/phase18 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$PWD/phase26d/logs/run_26d.status; mkdir -p phase26d/logs
st() { echo "$(date '+%F %T') $*" >> $S; }
[ -e phase26d/manifests/dev_harm_manifest.json ] || { st "build harm dev"; python phase26d/build_dev_26d.py > phase26d/logs/build_dev_26d.log 2>&1 || { st "build dev FAILED"; exit 1; }; }
for r in plain v0_2 v0_3_1; do
  [ -e phase26d/results/dev_$r.json ] || { st "dev reference $r"; python phase26d/eval_26d.py --dev $r >> phase26d/logs/eval_26d.log 2>&1 || { st "dev $r FAILED"; exit 1; }; }
done
st "DEV_REFERENCES_DONE (harm dev saturation check before training)"
B="python phase26d/build_pairs_26d.py"
[ -e phase26d/data/cand_b.jsonl ] || { st "pair candidates"; $B --candidates >> phase26d/logs/build_pairs_26d.log 2>&1 || { st "candidates FAILED"; exit 1; }; }
for g in "v0_3_1 a" "v0_3_1 b" "plain b" "v0_3 b"; do
  set -- $g
  [ -e phase26d/data/gen_$2_$1.jsonl ] || { st "generate $1 $2"; $B --generate $1 $2 >> phase26d/logs/build_pairs_26d.log 2>&1 || { st "generate $1 $2 FAILED"; exit 1; }; }
done
[ -e phase26d/data/pairs_train.jsonl ] || { st "pairs"; $B --pairs >> phase26d/logs/build_pairs_26d.log 2>&1 || { st "pairs FAILED"; exit 1; }; }
st "DATA_BUILT (counts in phase26d/manifests/pairs_26d_manifest.json)"
until [ -e phase26d/logs/data_checked ]; do sleep 60; done   # created after the counts and the dev reference rows were read
st "data checked"
for s in 11 12 13; do
  [ -e phase26d/runs/d_s$s/adapter/adapter_model.safetensors ] || { st "train d_s$s"; python phase26d/train_dpo_26d.py --seed $s >> phase26d/logs/train_d_s$s.log 2>&1 || { st "train d_s$s FAILED"; exit 1; }; }
  [ -e phase26d/results/dev_d_s$s.json ] || { st "dev d_s$s"; python phase26d/eval_26d.py --dev d_s$s >> phase26d/logs/eval_26d.log 2>&1 || { st "dev d_s$s FAILED"; exit 1; }; }
done
[ -e phase26d/results/selection.json ] || { python phase26d/eval_26d.py --select >> phase26d/logs/eval_26d.log 2>&1 || { st "select FAILED"; exit 1; }; }
st "seed chosen: $(python -c "import json;d=json.load(open('phase26d/results/selection.json'));print(d['chosen'],'-',d['how'])")"
[ -e phase26d/runs/merged_chosen/model.safetensors ] || { st "merge"; python phase26d/eval_26d.py --merge >> phase26d/logs/eval_26d.log 2>&1 || { st "merge FAILED"; exit 1; }; }
export HF_HUB_OFFLINE=0   # TruthfulQA (pinned revision) as in Phase 25/26
for step in safety harmbench squad tqa authored; do
  f=$( case $step in safety) echo safety;; harmbench) echo harmbench;; squad) echo squad_docmode;; tqa) echo tqa_letter_bf16;; authored) echo authored_reported;; esac )
  [ -e phase26d/results/$f.json ] && continue
  st "$step (once)"; python phase26d/eval_26d.py --$step >> phase26d/logs/eval_26d.log 2>&1 || { st "$step FAILED"; exit 1; }
done
python phase23/make_local_model.py phase26d/runs/merged_chosen phase18/local_models/p26d_merged_chosen >> phase26d/logs/eval_26d.log 2>&1
cd phase18
SAMPLES=$(python -c "import json;print(json.dumps(json.load(open('subsets.json'))['samples'],separators=(',',':')))")
MC="arc_easy,arc_challenge,hellaswag,piqa,winogrande,sciq,openbookqa,boolq,lambada_openai,mmlu"
name=p26d_merged_chosen; R=results/$name; H=../phase26d/harness/$name; mkdir -p $R $H
[ -e $R/mc.done ] || { st "harness mc"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks $MC --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $R/mc >> ../phase26d/logs/harness_26d.log 2>&1 && touch $R/mc.done || st "harness mc FAILED"; }
[ -e $H/tqa.done ] || { st "harness truthfulqa"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks truthfulqa_mc1,truthfulqa_mc2 --batch_size auto:4 --num_fewshot 0 --device cuda --seed 1234 --output_path $H/tqa >> ../phase26d/logs/harness_26d.log 2>&1 && touch $H/tqa.done || st "harness truthfulqa FAILED"; }
[ -e $H/gsm8k_full.done ] || { st "harness gsm8k full"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks gsm8k --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --num_fewshot 0 --device cuda --seed 1234 --output_path $H/gsm8k_full >> ../phase26d/logs/harness_26d.log 2>&1 && touch $H/gsm8k_full.done || st "harness gsm8k FAILED"; }
[ -e $H/triviaqa.done ] || { st "harness triviaqa"; python lm_eval_cli.py --model hf --model_args "pretrained=local_models/$name,dtype=float32" --tasks triviaqa --samples "$SAMPLES" --gen_kwargs max_gen_toks=512,repetition_penalty=1.0 --batch_size 16 --apply_chat_template --log_samples --num_fewshot 0 --device cuda --seed 1234 --output_path $H/triviaqa >> ../phase26d/logs/harness_26d.log 2>&1 && touch $H/triviaqa.done || st "harness triviaqa FAILED"; }
cd ..
if [ ! -e ~/chirag_work/gguf/p26d_chosen_Q8_0.gguf ]; then
  st "gguf convert"
  ( cd ~/chirag_work/llama.cpp && PYTHONPATH=$HOME/chirag_work/llama.cpp/gguf-py:$HOME/chirag_work/pylibs/phase18:$HOME/chirag_work/pylibs/phase23 CUDA_VISIBLE_DEVICES="" python convert_hf_to_gguf.py ~/chirag_work/minigpt/phase26d/runs/merged_chosen --outtype bf16 --outfile ~/chirag_work/gguf/p26d_chosen_bf16.gguf ) > phase26d/logs/gguf_convert.log 2>&1 \
    && ~/chirag_work/llama.cpp/build/bin/llama-quantize ~/chirag_work/gguf/p26d_chosen_bf16.gguf ~/chirag_work/gguf/p26d_chosen_Q8_0.gguf Q8_0 8 > phase26d/logs/gguf_quant.log 2>&1 || { st "gguf FAILED"; exit 1; }
fi
[ -e phase26d/results/gguf_q8_0.json ] || { st "laptop gguf"; CUDA_VISIBLE_DEVICES="" PYTHONPATH=$HOME/chirag_work/pylibs/phase18 python phase26d/eval_gguf_26d.py >> phase26d/logs/eval_gguf_26d.log 2>&1 || st "laptop gguf FAILED"; }
python phase26d/report_26d.py >> phase26d/logs/report_26d.log 2>&1 || st "report FAILED"
st "RUN_26D_DONE"

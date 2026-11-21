#!/bin/bash
# Chirii S1 GPU job: benchmarks of plain Qwen2.5-1.5B-Instruct and MiniGPT, with the exact Phase 18 settings
# (lm-evaluation-harness 0.4.13, 0-shot, greedy, float32), so the numbers can be compared with phase18/results.
#
# What it runs:
#   1. Plain Qwen2.5-1.5B-Instruct: ARC-Easy, HellaSwag, MMLU (all 57 subjects), and the FULL GSM8K test set
#      (1,319 questions, chat template, 256 new tokens). Phase 18 scored only 500 GSM8K questions; the full set is the
#      baseline for the "no more than 2 points lower" fine-tuning rule.
#   2. MiniGPT v4 extended (98M base) and MiniGPT v1.0.1 (34M, released): ARC-Easy, HellaSwag, MMLU, and the Phase 18
#      GSM8K 500-question subset.
#   3. A comparison with the Phase 18 numbers, written to chirii/results/.
#
# Resumable: each finished step leaves a .done marker and is skipped on rerun. Run from anywhere:
#   bash chirii/gpu/benchmarks/run_benchmarks.sh
set -u

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$REPO/chirii/results/gpu_benchmarks
mkdir -p "$OUT"
LOG=$OUT/run.log
exec > >(tee -a "$LOG") 2>&1

# Same environment as Phase 18; override any of these if the GPU PC differs
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-minigpt}
export HF_HOME=${HF_HOME:-$HOME/chirag_work/hf_cache}
export PYTHONPATH=${PYLIBS:-$HOME/chirag_work/pylibs/phase18}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# The Phase 18 local copy of the pinned Qwen snapshot (989aa798), with the shipped sampling defaults removed
QWEN=${QWEN:-local_models/qwen2_5_1_5b_instruct}
V4EXT=${V4EXT:-phase17/runs/foundation_v4x/snapshots/model_step_15258.pt}
V101=${V101:-phase11/runs/sft_v2/final_model.pt}

source "$CONDA_SH" && conda activate "$CONDA_ENV" || { echo "Could not activate conda env $CONDA_ENV"; exit 1; }
cd "$REPO/phase18" || exit 1

echo "=== start $(date '+%F %T')"
{
  echo "git_commit $(git -C "$REPO" rev-parse HEAD)"
  echo "date $(date '+%F %T %Z')"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import lm_eval, torch, transformers; print('lm_eval', lm_eval.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__)"
  pip freeze
} > "$OUT/environment.txt" 2>&1
grep -E "^(lm_eval|torch|transformers) " "$OUT/environment.txt"
python -c "import lm_eval, sys; sys.exit(0 if lm_eval.__version__ == '0.4.13' else 1)" \
  || { echo "lm_eval must be 0.4.13 to match Phase 18; see RUN_ON_GPU.md"; exit 1; }
[ -e "$QWEN/config.json" ] || { echo "Missing $QWEN in phase18/; see RUN_ON_GPU.md"; exit 1; }
for ckpt in "$V4EXT" "$V101"; do
  [ -e "$REPO/$ckpt" ] || { echo "Missing MiniGPT checkpoint $REPO/$ckpt; see RUN_ON_GPU.md"; exit 1; }
done

MC="arc_easy,hellaswag,mmlu"
COMMON="--num_fewshot 0 --device cuda --seed 1234"
GSM="--gen_kwargs max_gen_toks=256,repetition_penalty=1.0"
SAMPLES=$(python -c "import json;print(json.dumps({'gsm8k': json.load(open('subsets.json'))['samples']['gsm8k']},separators=(',',':')))")
FAILED=0

step() {  # step <name> <args...>
  local name=$1; shift
  [ -e "$OUT/$name.done" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date '+%F %T')"
  mkdir -p "$OUT/$name"
  if python lm_eval_cli.py "$@" $COMMON --output_path "$OUT/$name"; then
    touch "$OUT/$name.done"; echo "=== done $name $(date '+%F %T')"
  else
    echo "=== FAILED $name $(date '+%F %T')"; FAILED=1
  fi
}

step qwen2_5_1_5b_instruct/mc --model hf --model_args "pretrained=$QWEN,dtype=float32" --tasks $MC --batch_size auto:4
step qwen2_5_1_5b_instruct/gsm8k_full --model hf --model_args "pretrained=$QWEN,dtype=float32" --tasks gsm8k $GSM --batch_size 16 --apply_chat_template --log_samples

step minigpt_v4ext_base/mc --model minigpt --model_args "checkpoint=$REPO/$V4EXT,truncation_log=$OUT/minigpt_v4ext_base/truncation_mc.json" --tasks $MC --batch_size 16
step minigpt_v4ext_base/gsm8k --model minigpt --model_args "checkpoint=$REPO/$V4EXT,truncation_log=$OUT/minigpt_v4ext_base/truncation_gsm8k.json" --tasks gsm8k --samples "$SAMPLES" $GSM --log_samples

step minigpt_v1_0_1/mc --model minigpt --model_args "checkpoint=$REPO/$V101,truncation_log=$OUT/minigpt_v1_0_1/truncation_mc.json" --tasks $MC --batch_size 32
step minigpt_v1_0_1/gsm8k --model minigpt --model_args "checkpoint=$REPO/$V101,chat_generation=True,calculator=True,truncation_log=$OUT/minigpt_v1_0_1/truncation_gsm8k.json" --tasks gsm8k --samples "$SAMPLES" $GSM --log_samples

echo "=== compare $(date '+%F %T')"
python "$REPO/chirii/gpu/benchmarks/compare_to_phase18.py" --results "$OUT" --phase18 "$REPO/phase18/results" || FAILED=1
echo "=== end $(date '+%F %T') failed=$FAILED"
exit $FAILED

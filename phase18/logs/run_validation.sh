#!/bin/bash
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate minigpt
export HF_HUB_OFFLINE=1 HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
M="pretrained=HuggingFaceTB/SmolLM2-135M,revision=93efa2f097d58c2a74874c7e644dbc9b0cee75a2,dtype=float32"
python -m lm_eval --model hf --model_args "$M" --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,commonsense_qa,mmlu_continuation,triviaqa \
  --num_fewshot 0 --batch_size auto:4 --device cuda --seed 1234 --output_path results/validation/zero_shot
python -m lm_eval --model hf --model_args "$M" --tasks gsm8k --num_fewshot 5 --batch_size 16 --device cuda --seed 1234 --output_path results/validation/gsm8k_5shot
echo VALIDATION_DONE

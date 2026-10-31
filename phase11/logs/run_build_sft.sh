#!/bin/sh
# CPU only: the GPU belongs to the pretraining run.
export CUDA_VISIBLE_DEVICES="" RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_DISABLE_TELEMETRY=1
~/anaconda3/envs/minigpt/bin/python phase11/build_sft_data.py > phase11/logs/build_sft_data.log 2>&1
echo "rc=$? $(date -Is)" > phase11/logs/build_sft_data.done

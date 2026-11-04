#!/bin/bash
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache PYTHONPATH=$HOME/chirag_work/pylibs/phase18 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nice -n 10 python contamination.py --workers 6 && echo CONTAMINATION_DONE

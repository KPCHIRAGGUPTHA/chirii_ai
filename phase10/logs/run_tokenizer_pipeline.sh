#!/bin/sh
# Trains tokenizer v2, reports stats, then tokenizes every shard. 8 threads: the PC is shared and already heavily loaded.
PY=~/anaconda3/envs/minigpt/bin/python
export RAYON_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true
"$PY" phase10/tokenizer_v2.py train > phase10/logs/tokenizer_train.log 2>&1 &&
"$PY" phase10/tokenizer_v2.py stats > phase10/logs/tokenizer_stats.log 2>&1 &&
"$PY" phase10/tokenize_shards.py > phase10/logs/tokenize_shards.log 2>&1
echo "pipeline rc=$? $(date -Is)" > phase10/logs/tokenizer_pipeline.done

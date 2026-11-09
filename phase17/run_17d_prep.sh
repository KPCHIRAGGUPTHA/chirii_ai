#!/bin/bash
# Phase 17D: wait for collection, then filter (benchmark decontamination, cross-source dedupe), tokenize + mix,
# write the base config, and run the pre-registered launch gate (which launches the long run only if it passes).
cd "$(dirname "$0")/.." || exit 1
source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate minigpt
export HF_HOME=$HOME/chirag_work/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 RAYON_NUM_THREADS=8
S=phase17/logs/run_17d.status
echo "$(date '+%F %T') waiting for collection" >> $S
until grep -q COLLECT_17D_DONE phase17/logs/collect_17d_edu.log; do sleep 30; done
for src in edu web; do grep -q '"status": "completed_target"' phase17/manifests17d/${src}_text_manifest.json || { echo "$(date '+%F %T') $src collection incomplete: STOP" >> $S; exit 1; }; done
echo "$(date '+%F %T') filter" >> $S
PYTHONHASHSEED=0 PYTHONPATH=$HOME/chirag_work/pylibs/phase18 nice -n 10 python phase17/filter_17d.py --workers 8 >> phase17/logs/filter_17d.log 2>&1 || { echo "$(date '+%F %T') filter FAILED" >> $S; exit 1; }
echo "$(date '+%F %T') tokenize + mix" >> $S
nice -n 10 python phase17/tokenize_17d.py >> phase17/logs/tokenize_17d.log 2>&1 || { echo "$(date '+%F %T') tokenize FAILED" >> $S; exit 1; }
python phase17/make_config_17d.py >> phase17/logs/tokenize_17d.log 2>&1 || { echo "$(date '+%F %T') config FAILED" >> $S; exit 1; }
echo "$(date '+%F %T') gate" >> $S
python phase17/gate_17d.py >> phase17/logs/gate_17d.log 2>&1
echo "$(date '+%F %T') gate finished: $(python -c "import json;r=json.load(open('phase17/runs/gate_17d.json'));print('ALL PASS, launched' if r['launched'] else 'NOT launched')")" >> $S

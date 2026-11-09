# Cleanup plan for 11 November 2026 (nothing has been deleted yet)

This is a plan only. Nothing listed here was removed on 10 November. Paths are written from the home directory
(`~`) of the shared PC account. Sizes were measured on 10 November.

## 0. Before deleting anything: the backup check (done 10 November)

**Private release assets** (repository KPCHIRAGGUPTHA/minigpt). All 13 assets match their local files by SHA-256
(compared with GitHub's own digests):

| Release | Assets |
| :--- | :--- |
| `weights-2026-11-03` | v1.0 and v1.0.1 demo zips; 34M pretrain `model_step_11524.pt`; `sft_v1`, `sft_v2` (= v1.0 / v1.0.1 model), `sft_v3`; `tokenizer_v2.json`; `SHA256SUMS` |
| `weights-2026-11-05` | Foundation v4 best model; Foundation v4 `model_step_05722.pt` |
| `weights-2026-11-09` | Foundation v4 extended `model_step_15258.pt` (SHA-256 `f67f325e…`); 17E replay models `sft_v5r_s11/12/13` (`3688570f…`, `d27dd231…`, `81fd943a…`); `SHA256SUMS` |

**Branches and tags.** Every ref on GitHub was compared with the local copy; all match.

| Private repo (KPCHIRAGGUPTHA/minigpt) | Commit | Public repo (KPCHIRAGGUPTHA/chirii_ai) | Commit |
| :--- | :--- | :--- | :--- |
| main | `93f0f5f595` | main | `3e74e07f36` (before this update) |
| phase9-establish-truth | `dd34d8565b` | phase9-establish-truth | `bf87edfd6a` |
| phase10-foundation-v2 | `a313c2de42` | phase10-foundation-v2 | `d6dd993ac0` |
| phase11-instruct | `04e92645e6` | phase11-instruct | `0f278d3170` |
| phase13-overrefusal | `79dc3b3b8b` | phase13-overrefusal | `8ef3d840b2` |
| phase14-safety | `7b3f10c5ac` | phase14-safety | `3ea1a88576` |
| phase15-rag | `cc0482c3cd` | phase15-rag | `c661c7fbf2` |
| phase16-reference | `cf9d4eef79` | wrapup | `79e1912ae2` |
| phase17-data | `2443bc1973` | tag minigpt-v1.0 | `6647429872` |
| phase17-foundation-v4 | `b1fe7ec65b` | tag minigpt-v1.0.1 | `5220b56599` |
| phase17b-eval | `d1210c5996` | | |
| phase17d-extended | `a9a960920d` (before this update) | | |
| phase18-benchmarks | `0588503004` | | |
| wrapup | `c6246cad5e` | | |
| tag minigpt-v1.0 / v1.0.1 | `54e94a47f4` / `858effb07a` | | |
| tags weights-2026-11-03 / -05 / -09 | `93f0f5f595` | | |

The final-wrap-up commit adds one branch to each repository (see the commit message); the final report states the
new hashes.

**Exists ONLY on this PC (will be lost on deletion unless copied first):**

| What | Where | Size |
| :--- | :--- | ---: |
| **17C SFT models `sft_v5_s11/12/13`, including the chosen seed 13** (not in any release) | `minigpt/phase17/runs/` | (part of 26G) |
| v4 snapshots 2000/4000, v4-extended snapshots 2000–14000, resumable `checkpoint.pt` files, probe and stability runs | `minigpt/phase17/runs/` | 26G total |
| Tokenised 17D data (8.0B tokens) and its raw shards | `minigpt/phase17/data17d/` | 97G |
| Tokenised 17A data (3.0B tokens) | `minigpt/phase17/data/` | 19G |
| Phase 10 data and runs | `minigpt/phase10/data/`, `minigpt/phase10/runs/` | 7.3G, 2.8G |
| Phase 11, 13, 14, 15 runs and data (RAG index, SFT variants, Phase 13 rebalanced seeds) | `minigpt/phase11/`, `phase13/runs/`, `phase14/runs/`, `phase15/` | 2.6G, 1.9G, 1.3G, 2.9G |
| SFT v5 data and the 17E replay rows | `minigpt/phase17/sft_v5/` | 63M |
| Full lm-evaluation-harness per-sample outputs | `minigpt/phase18/results/` | 364M |
| Old checkpoints and data (Phases 3–9) | `minigpt/checkpoints/`, `minigpt/data/`, `minigpt/phase9/` | 944M, 668M, 2.1G |
| The development repository's own full, uncleaned git history | `minigpt/.git` | 367M |
| Phase 9 retrain checkpoints archive | `~/chirag_work/phase9_v2_checkpoints.tar.gz` (+ `.sha256`) | 917M |

Everything else of value (code, reports, small evaluation files, the five weights releases) is on GitHub. All
pretraining and instruction data can be re-downloaded from Hugging Face (section 3). An older partial copy exists
on the author's USB drive.

**Recommended before deletion:** if the 17C chosen model may be needed again, upload
`phase17/runs/sft_v5_s13/final_model.pt` (and seeds 11 and 12) to a new private release; everything else on the
list can be regenerated from code and public data, at the cost of GPU time (about 30 h + 80.5 h for the two 98M
pretraining runs).

## 1. What to remove on 11 November

Run in this order. Each step is safe to skip.

1. **Crontab entry** (the only crontab line; there was no crontab before this project). It is inert: the run it
   resumes has finished.

   ```
   crontab -l                                         # shows the one MINIGPT-PHASE17A line
   crontab -l | grep -v "MINIGPT-PHASE17A" | crontab -
   crontab -l                                         # check: the line is gone
   ```

2. **Make sure nothing of this project is running**: `nvidia-smi` shows no python process of this account, and
   `ps -u $USER -o pid,cmd | grep -E "phase1[0-9]|minigpt"` shows nothing.

3. **The project folder** (about 182G in total):

   ```
   rm -rf ~/chirag_work
   ```

   This removes the repository and all data and runs (`~/chirag_work/minigpt`, 164G), the Hugging Face cache
   (`~/chirag_work/hf_cache`, 5.3G), the pip install targets for Phases 16 and 18 (`~/chirag_work/pylibs`, 468M),
   working clones and scratch files (`~/chirag_work/tmp`, 725M), the Phase 9 archive (917M), and **the GitHub
   credentials for KPCHIRAGGUPTHA (`~/chirag_work/.gh`)**. Optionally revoke that token as well on GitHub
   (Settings → Applications / Developer settings).

4. **The conda environment** created for this project:

   ```
   conda env remove -n minigpt        # ~6.1G, ~/anaconda3/envs/minigpt
   ```

   An environment named `minigpt-e` (6.9G, created 29 October, contains git-lfs) also exists. It is not referenced
   by any file in this project and its origin was not confirmed; check with its owner before removing it.

5. **Small caches this project created in the home directory:**

   ```
   rm -rf ~/.triton                   # Triton kernel cache (68K)
   rm -rf ~/.nv                       # CUDA compute cache (8K)
   rm -rf /tmp/torchinductor_$USER     # torch.compile cache (158M; also cleared at reboot)
   ```

6. **pip cache** (`~/.cache/pip`, 5.1G). Probably mostly this project's downloads (torch, transformers, lm-eval),
   but it is shared by everything on this account. Clear it only if nobody else relies on it: `pip cache purge`.
   `~/.cache/huggingface` (12K) can be left or removed.

**Do not touch:** `~/.config/gh` and `~/.cache/gh` belong to the other GitHub login on this account; the base conda
environment and system settings were never changed by this project.

## 2. GNOME power settings to restore

Changed so the long training runs were not interrupted by suspend. Current values and how to restore them:

| Setting | Now | Restored by `gsettings reset` to |
| :--- | :--- | :--- |
| `org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type` | `'nothing'` | `'suspend'` |
| `org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout` | `3600` | Ubuntu's default (`0` from Ubuntu's override; schema default 900) |
| `org.gnome.settings-daemon.plugins.power power-button-action` | `'nothing'` | Ubuntu's default (`'interactive'`; schema default `'suspend'`) |
| `org.gnome.desktop.session idle-delay` | `0` | `300` (5 minutes) |

```
gsettings reset org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type
gsettings reset org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout
gsettings reset org.gnome.settings-daemon.plugins.power power-button-action
gsettings reset org.gnome.desktop.session idle-delay
```

`gsettings reset` returns each key to the system default, including Ubuntu's own overrides, which is what the PC
had before. Screen lock (`org.gnome.desktop.screensaver lock-enabled`) was not changed. `/etc/systemd/logind.conf`
was not changed.

## 3. How to rebuild later from GitHub

1. **Code and reports**: `git clone https://github.com/KPCHIRAGGUPTHA/minigpt` (private; every phase is a branch;
   the latest state is branch `final-wrapup`) or the public `KPCHIRAGGUPTHA/chirii_ai`.
2. **Environment**: Python 3.11, torch 2.15.0 (CUDA build), tokenizers 0.23.3, numpy 2.4.6, datasets 5.1.0,
   pyarrow 26.0.0, tqdm 4.70.1. For Phases 16 and 18 also transformers 5.22.1 and lm-eval 0.4.13 (installed to a
   separate folder and added with `PYTHONPATH`). The demo needs only `phase14/dist/minigpt_v1_demo/requirements.txt`.
3. **Weights**: `gh release download weights-2026-11-03 -R KPCHIRAGGUPTHA/minigpt` (and `-11-05`, `-11-09`); check
   with `sha256sum -c SHA256SUMS`. Put each file at the path given in the release notes.
4. **Data**: `phase10/collect_data.py`, `phase17/collect_more.py` and `phase17/collect_17d.py` re-download FineWeb-Edu
   and FineWeb; the committed manifests record what was collected (the 17D collector pins the dataset revision),
   and the trainers check each token file's SHA-256 before training, so a rebuilt file that differs is detected
   rather than silently used.
5. **Retraining** is only needed for the models listed in section 0 as PC-only.

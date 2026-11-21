# GPU job: December benchmarks (S1)

Benchmarks plain Qwen2.5-1.5B-Instruct and two MiniGPT models on ARC-Easy, HellaSwag, MMLU and GSM8K, with exactly
the Phase 18 settings, and checks the results against Phase 18.

**Why run it when Phase 18 already has these numbers?**
1. It checks that the lab setup still reproduces them.
2. It scores plain Qwen on all 1,319 GSM8K questions. Phase 18 used 500, which has a standard error of about 2.1
   points: as large as the "no more than 2 points lower" fine-tuning rule.
3. The same script will score any S7 fine-tune, so every comparison uses identical settings.

**Run it before 31 December.** Lab access may end then.

## Before you start

The job uses the Phase 18 environment on the GPU PC. It expects:

| What | Default location (change with the variable in brackets) |
| :--- | :--- |
| Repository clone | wherever you `git pull`; the script finds it itself |
| Conda env with lm-eval 0.4.13 | `~/anaconda3`, env `minigpt` (`CONDA_SH`, `CONDA_ENV`) |
| Hugging Face cache | `~/chirag_work/hf_cache` (`HF_HOME`) |
| Phase 18 extra libraries | `~/chirag_work/pylibs/phase18` (`PYLIBS`) |
| Pinned Qwen copy | `phase18/local_models/qwen2_5_1_5b_instruct` (`QWEN`) |
| MiniGPT v4 extended | `phase17/runs/foundation_v4x/snapshots/model_step_15258.pt` (`V4EXT`) |
| MiniGPT v1.0.1 | `phase11/runs/sft_v2/final_model.pt` (`V101`) |

The script checks each of these and stops with a clear message if one is missing. It also stops unless lm-eval is
0.4.13.

If the conda env is gone, recreate it with:

```bash
conda create -n minigpt python=3.11 -y && conda activate minigpt
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r chirii/gpu/requirements-gpu.txt
```

## Run

```bash
cd ~/chirag_work/minigpt && git checkout chirii-poc && git pull
```

```bash
nohup bash chirii/gpu/benchmarks/run_benchmarks.sh > /dev/null 2>&1 &
```

Watch progress:

```bash
tail -f chirii/results/gpu_benchmarks/run.log
```

If it stops, run the same command again. Finished steps are skipped.

**Expected runtime: about 2.5 to 3.5 hours.** From the Phase 18 timings:

| Step | Estimate |
| :--- | ---: |
| Plain Qwen: ARC-Easy, HellaSwag, MMLU | about 1 h 20 min |
| Plain Qwen: GSM8K, all 1,319 questions | about 50 min |
| MiniGPT v4 extended: all four | about 25 min |
| MiniGPT v1.0.1: all four | about 10 min |

The last line of `run.log` says `failed=0` if everything worked.

## Commit and push the results

```bash
git add chirii/results/BENCHMARKS.md chirii/results/benchmarks.json chirii/results/gpu_benchmarks
```

```bash
git commit -m "S1: December GPU benchmarks of plain Qwen2.5-1.5B and MiniGPT" && git push
```

**What gets committed:** the summary (`BENCHMARKS.md`, `benchmarks.json`), the lm-eval result and sample files, the
environment record (`environment.txt`, including `pip freeze` and the GPU name) and `run.log`. No model weights are
written by this job.

**Check before committing:** `git status` should show nothing under `phase18/local_models` or `phase17/runs`.

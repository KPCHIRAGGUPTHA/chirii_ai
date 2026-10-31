# Phase 9 GPU Results

All four retrains, their evaluations and the bits-per-byte recheck of C_old and D_old ran on the college GPU PC on 2026-10-31. Figures from anywhere else are marked with †.

## Answer

**No. Under the rule, D is not better than C.** The rule needs both D seeds to beat both C seeds in bits per byte, and only one does.

| Comparison (lower bits per byte is better) | C_v2 | D_v2 | D - C |
| :--- | ---: | ---: | ---: |
| Seed 42 | 2.2548 | 2.3980 | +0.1432 (D worse) |
| Seed 43 | 2.1403 | 2.1377 | -0.0026 (D slightly better) |
| Mean | 2.1976 | 2.2679 | +0.0703 (D worse) |

- **D's best seed**: D/43 beats both C seeds, but only just beats C/43. Its loss lead of 0.0035 is close to the standard error of each loss (about 0.0028), so treat it as a tie.
- **D's other seed**: D/42 is worse than both C seeds.
- **How settled this is**: the gap between seeds (C 0.1145, D 0.2603) is larger than the 0.0703 gap between the means. Two seeds show D did not clearly win, but they do not show that C is truly better.

## Results

| Model | Parameters | Bits per byte | Loss | Training time | Tokens/sec |
| :--- | ---: | ---: | ---: | ---: | ---: |
| C_old | 6,613,504 | 2.1531 | 2.9730 | 354.96 s (5.9 min) † | 28,849 † |
| C_v2 seed 42 | 6,613,504 | 2.2548 | 3.1136 | 250 s (4.2 min) | 40,960 |
| C_v2 seed 43 | 6,613,504 | 2.1403 | 2.9554 | 255 s (4.3 min) | 40,157 |
| **C_v2 mean** | 6,613,504 | **2.1976** | 3.0345 | 252.5 s (4.2 min) | 40,559 |
| D_old | 15,164,800 | 2.8804 | 3.9774 | 670.88 s (11.2 min) † | 15,263 † |
| D_v2 seed 42 | 15,164,800 | 2.3980 | 3.3112 | 370 s (6.2 min) | 27,676 |
| D_v2 seed 43 | 15,164,800 | 2.1377 | 2.9519 | 362 s (6.0 min) | 28,287 |
| **D_v2 mean** | 15,164,800 | **2.2679** | 3.1316 | 366 s (6.1 min) | 27,982 |

† **Not run on this machine.** Copied from the original training records: C_old from `results/phase5d/model_c_gpu_history.json` (NVIDIA L4) and D_old from `checkpoints/phase8/model_d/training_report.md` (GPU not named there). Don't read these as a speed comparison with this PC.

- **Bits per byte and loss**: `phase9/eval_standard.py` on all 1,765,632 tokens of the pinned shard `val_v1024.bin`, SHA-256 `ec9917aa49a17e6e31ea607bf1a9c11f07624a1411d81b855c3525519e39a006`. Means are simple averages of the two seeds.
- **C_old and D_old bits per byte and loss**: rerun here on the pinned checkpoints (`phase9/eval_results/old_models_gpu_recheck.json`). They match the laptop's values to within 1e-8 bits per byte, so the CPU and GPU evaluations agree.
- **Training time**: wall clock for the whole process, to the nearest second, including startup, validation and checkpoint saves (`phase9/logs/retrain_timing.log`).
- **Tokens/sec**: 10,240,000 tokens divided by training time. The mean rows average the two seeds.
- **Training setup**: all four runs completed 2,500 of 2,500 steps with bf16 autocast, and each picked step 2,500 as its best checkpoint.

### Compared with the old models

- **D_v2 vs D_old**: 0.6125 better on the mean (0.4824 for seed 42, 0.7427 for seed 43). Both seeds beat D_old by far more than the seed gap, so most of D_old's deficit came from the old trainer.
- **C_v2 vs C_old**: 0.0445 worse on the mean. C_old (2.1531) falls between the two C_v2 seeds, so the new trainer shows no clear change for Model C.

### Goal eval v2

| Model | All prompts | Without SFT-overlap prompts |
| :--- | ---: | ---: |
| C_v2 seed 42 | 0/202 | 0/195 |
| C_v2 seed 43 | 0/202 | 0/195 |
| D_v2 seed 42 | 0/202 | 0/195 |
| D_v2 seed 43 | 0/202 | 0/195 |

As the runbook expected, these models have no instruction training and score zero. The goal eval gives no evidence for or against D. C_old's 0/202 in `GOAL_EVAL_V2.md` is the laptop's result and was not rerun here. D_old has not been scored.

## Things to Know When Reading These Numbers

- **Large seed effect**: seed 42 was worse than seed 43 for both models. It was already behind by step 300 and stayed behind, with no failure or crash in the logs. I did not find the cause.
- **Seeds are not independent across models**: the batch sampler depends only on the seed, so by the code C and D with the same seed draw the same batch positions. That may be why seed 42 is worse for both, but I did not test it.
- **Training had not levelled off**: validation loss was still falling at step 2,500 in all four runs, which is the end of the fixed token budget. These numbers compare the models at 10.24M tokens, not at convergence.
- **Checkpoint selection**: the training logs' best validation losses (C 3.0693, 2.9142; D 3.2673, 2.9105) use only the first 64,000 validation tokens. The table's values use the whole shard.

## Environment

| Item | Value |
| :--- | :--- |
| GPU | NVIDIA RTX A2000 12GB (12,282 MiB), VBIOS 94.06.32.00.22 |
| Driver | 580.178.04 |
| CUDA | 13.0 reported by the driver; 13.2 is the version torch was built with |
| torch | 2.15.0+cu132, cuDNN 9.26.0 (`92600`), bf16 supported |
| Python, numpy | 3.11.18, 2.4.6, conda env `minigpt` |
| pytest | 9.2.0, the only package installed (with its dependencies), in `minigpt` only |
| OS | Ubuntu 24.04, Linux 7.0.0-38-generic |

## Checks Before Training

- **Data**: `train_v1024.bin`, `val_v1024.bin` and `val_v1377.bin` match `phase9/data/SHA256SUMS`. Every pinned shard, tokenizer and source text hash in `phase9/eval_config.json` also matches.
- **Protected artifacts**: `phase9/verify_protected.py` passed all 5 before training and again after evaluation.
- **Tests**: the GPU-only test passed, and the full `phase9/tests` suite gave 129 passed and 0 skipped (`phase9/logs/gpu_only_test.log`, `phase9/logs/gpu_full_test_suite.log`). This is the first run of the GPU code path.

## Smoke Tests (50 Steps)

| | Model C | Model D |
| :--- | ---: | ---: |
| Autocast and device | bf16, RTX A2000 | bf16, RTX A2000 |
| Loss, step 1 to step 50 | 6.9907 to 5.3813 | 7.0179 to 5.3818 |
| Tokens/sec at step 50 | 33,072 | 24,458 |
| GPU utilization at step 50 | 48% | 58% |
| Peak GPU memory allocated by PyTorch | 0.257 GB | 0.54 GB |

## Throughput During the Retrains

| Run | GPU utilization, logged mean (range) | Peak GPU memory allocated | Warnings |
| :--- | ---: | ---: | ---: |
| C_v2 seed 42 | 60% (39 to 91) | 0.257 GB | 22 |
| C_v2 seed 43 | 60% (39 to 89) | 0.257 GB | 19 |
| D_v2 seed 42 | 78% (51 to 99) | 0.54 GB | 0 |
| D_v2 seed 43 | 79% (50 to 99) | 0.54 GB | 0 |

## Warnings and Anomalies

- **Throughput warnings**: all 41 are GPU utilization below 50% in Model C runs. None is about the 10,000 tokens/sec floor. They are harmless for these results, and I did not tune anything, as instructed.
- **GPU underuse**: the GPU had plenty of memory to spare, with at most 0.54 GB allocated of 12 GB. Likely causes are the micro-batch of 8 and data loading on a busy CPU. Another user's job held the load average near 300 on 24 cores throughout, so timings on a quieter machine may differ.
- **Gradient spike in D_v2 seed 43**: the norm was 74.85 with loss 6.01 at step 50, the end of warmup. Clipping limited it to 1.0, and the loss was 5.54 by step 100. This run went on to be the best of the four. Gradients are logged every 50 steps, so spikes between log lines would not appear.
- **Heat**: the GPU peaked at 89 °C and 68 W of 70 W in 30-second samples, averaging 82 °C. Its slowdown temperature is 100 °C. The driver's counters since it started show 17.9 s of software thermal slowdown, 715 s of software power capping and 0 s of hardware thermal slowdown. These counters also cover time before this work, so they can't be pinned on these runs. Heat can affect speed, not the correctness of the results.
- **No failures**: there were no crashes or non-finite losses, and no run needed a resume.

## Changes from the Runbook

- Ran directly in the copied repo with the `minigpt` conda env, with no bundle or venv.
- `phase9/logs/run_retrains.sh` replaces the one-line `nohup` command. It keeps the same order and stop-on-failure behaviour, and adds `setsid`, timing and a 30-second `nvidia-smi` log.
- **Added**: the recheck of C_old and D_old on this machine, written to a separate file so the laptop's records stay in `standard_eval.json`.

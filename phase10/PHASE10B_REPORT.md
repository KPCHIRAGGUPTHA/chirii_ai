# Phase 10B Report: Pretraining Run `foundation_v3_clean_2ep`

All numbers come from this run on the college GPU PC (NVIDIA RTX A2000 12GB). Training ran on 2026-10-31. Final evaluation ran on 2026-11-02.

## Summary

- **Completed**: 11,524 of 11,524 steps, 1,510,473,728 tokens, two passes over `train_clean.bin`.
- **Failures**: 1 attempt, 0 resumes, 0 NaN/Inf values.
- **Score**: 1.1479 bits per byte on the Phase 5D text, against Model C's 2.1531. Lower is better.
- **Generations**: the first sentence is often right, then greedy text falls into repetition. Arithmetic and code are poor. This model has had no instruction training.

## Run Health

| Item | Value |
| :--- | ---: |
| Started, finished (IST) | 2026-10-31 17:00:07, 22:48:10 |
| Wall time reported by the trainer | 20,853 s (5 h 47 min) |
| Training time | 20,644 s |
| Tokens/sec, training only | 73,166 |
| Tokens/sec, overall | 72,436 |
| Attempts, resumes | 1, 0 |
| Non-finite values | 0 |
| `steps.jsonl` | 11,524 consecutive entries, all finite |
| Max grad norm, all steps | 3.889 (step 2) |
| Max grad norm after warmup | 0.817 (step 665) |
| Peak GPU memory allocated, reserved by PyTorch | 8.181 GB, 8.926 GB |

**Slowdowns**: none of note. Across the 231 log lines after step 1, tokens/sec had a median of 73,398, a 5th percentile of 72,702, a minimum of 68,148 (step 250) and a maximum of 74,939. The six slowest lines were all at or before step 700. Logged GPU utilization averaged 99.8%, with a minimum of 95%.

**Temperature**: 47 samples, one at each evaluation, every 250 steps. The mean was **89.2 °C**, the peak **90 °C** and the minimum 88 °C. There was no continuous `nvidia-smi` log for this run, so a short peak between samples would not appear. The temperature was already 89 °C at the first sample, so the average does not include the warm-up period.

The driver's slowdown counters on 2026-11-02 read 22,886 s of software power capping, 28.7 s of software thermal slowdown and 0 s of hardware thermal slowdown. They count from driver start and include other work, so they cannot be pinned on this run. The steady tokens/sec above is the better evidence.

## Validation Curve

Validation here uses the run's pinned subset: the first 256,000 tokens of `val.bin` (1,084,941 bytes).

| Step | Val loss | Bits per byte | GPU temp (°C) |
| ---: | ---: | ---: | ---: |
| 1,000 | 4.0936 | 1.3935 | 89 |
| 2,000 | 3.7791 | 1.2865 | 90 |
| 3,000 | 3.6516 | 1.2431 | 89 |
| 4,000 | 3.5705 | 1.2154 | 89 |
| 5,000 | 3.5111 | 1.1952 | 90 |
| 6,000 | 3.4635 | 1.1790 | 89 |
| 7,000 | 3.4246 | 1.1658 | 89 |
| 8,000 | 3.3953 | 1.1558 | 89 |
| 9,000 | 3.3671 | 1.1462 | 89 |
| 10,000 | 3.3492 | 1.1401 | 89 |
| 11,000 | 3.3356 | 1.1355 | 89 |
| 11,524 (final) | 3.3329 | 1.1346 | 89 |

The best subset loss was 3.3327 at step 11,500, so `best_model.pt` is step 11,500. The final snapshot at step 11,524 differs by 0.0002 and is the one used from here on.

## Final Scores

`phase10/eval_v3.py` on `snapshots/model_step_11524.pt`, full precision, every whole block of each text (`eval_results/eval_v3.json`, `logs/eval_v3_final.log`).

| Text | Tokens | Bytes | Loss | Bits per byte |
| :--- | ---: | ---: | ---: | ---: |
| New val | 3,789,312 | 15,790,584 | 3.2741 | **1.1335** |
| Phase 5D, LF form | 839,680 | 3,516,278 | 3.3321 | **1.1479** |
| Model C on Phase 5D | | 3,517,395 | 2.9730 | 2.1531 |

The difference from Model C on Phase 5D is **-1.0051 bits per byte**.

- **Why compare bits per byte, not loss**: the two models use different tokenizers, with a different number of bytes per token.
- **Fairness**: `train_clean.bin` excludes documents with exact-line matches to the Phase 5D text. Reworded or re-wrapped copies would not have been found. The tokenizer's training sample may also include some of those documents, which affects the vocabulary only.

## Final Generations (Verbatim)

Greedy, 64 new tokens, step 11,524, from `generations.jsonl`.

**`The capital of France is`**

```
The capital of France is the city of Paris. The city is the capital of the French Republic. The capital of the French Republic is the city of Paris. The capital of the French Republic is the city of Paris. The capital of the French Republic is the city of Paris. The capital of the French Republic is the city of Paris. The
```

**`Water is made of hydrogen and`**

```
Water is made of hydrogen and oxygen.
The hydrogen is made of hydrogen and oxygen.
The hydrogen is made of hydrogen and oxygen.
The hydrogen is made of hydrogen and oxygen.
The hydrogen is made of hydrogen and oxygen.
The hydrogen is made of hydrogen and oxygen.
The hydrogen is made of hydrogen and oxygen.
The
```

**`2 + 2 =`**

```
2 + 2 = 100000000000000000000000000000000000000000000000000000000000000
```

**`The main reason plants need sunlight is`**

```
The main reason plants need sunlight is because they are able to absorb sunlight.
The plants that grow in the sun are called photosynthesis.
The plants that grow in the sun are called photosynthesis.
The plants that grow in the sun are called photosynthesis.
The plants that grow in the sun are called photosynthesis.
The plants that
```

**`def add(a, b):\n`**

```
def add(a, b):
- 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
```

## Snapshots

In `phase10/runs/foundation_v3_clean_2ep/snapshots/`.

| File | Bytes |
| :--- | ---: |
| `model_step_02000.pt` | 135,523,951 |
| `model_step_04000.pt` | 135,523,951 |
| `model_step_06000.pt` | 135,523,951 |
| `model_step_08000.pt` | 135,523,951 |
| `model_step_10000.pt` | 135,523,951 |
| `model_step_11524.pt` | 135,523,951 |

`model_step_11524.pt` has SHA-256 `68ca4deaef0c3e57c8fa01bdc7f53a4c4826324e902732b2292d923c90c1b8bc`.

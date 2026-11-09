# Phase 17C — SFT v5 on the extended base: report

Runs: the college GPU PC (RTX A2000 12 GB), 9 November 2026. Criteria: `phase17/CRITERIA_17C.md`, committed on
6 November before any SFT run. Every number below comes from a file in `phase17/results/phase17c/` or
`phase18/results/sft_v5_chosen/`.

## In short

**The chosen model does not pass the 17C bars, so there is no v2.0 release.** It passes harmful_v3 (95%) and unknown
(72%) but fails bare arithmetic by one item (16/18 = 88.9%, bar 90%) and fails over-refusal clearly (77/100, bar 90%).
On dev, no seed met the selection bar either: every seed failed the 90% over-refusal bar (37–43 of 60).

Compared with MiniGPT v1.0.1 on the same sealed sets, the new model is about level on safety and over-refusal, a little
better on definitions and facts, one item worse on bare arithmetic, and much better on the multi-step word problems —
but those 18 items are not independent of the SFT v5 templates, and on the independent GSM8K test both models stay
near zero (1.6% vs 0.6%).

## What was trained

- **Base**: the final Phase 17D model (98,425,344 parameters; SHA-256 `f67f325e…`).
- **Data**: SFT v5 as pinned (113,036 training rows: SFT v2 + 535 mined harmless "edgy" rows ×3 + 3,063 code-verified
  multi-step calculator problems; `phase17/manifests/sft_v5_manifest.json`).
- **Settings**: sft_v2's hyperparameters (2 epochs, micro-batch 16 × 2, learning rate 2e-4 → 2e-5, warmup 50), seeds
  11, 12 and 13. Each run completed 7,064 steps in about 28 minutes; final validation losses 1.5423, 1.5373 and 1.5422.
- **Decoding** for every 17C generation: chat runtime with the calculator, greedy with no-repeat-3-gram.

## Selection on dev (committed rule)

Goal dev v2.1 (176 items, scorer_v4) and the old over-refusal set (60 items, `score_overrefusal`).

| Seed | Dev total /176 | Arithmetic | Harmful | Unknown | Old over-refusal | Qualifies? |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 11 | 117 | 32/32 | 25/26 | 21/26 | 43/60 | no (over-refusal) |
| 12 | 116 | 31/32 | 26/26 | 21/26 | 37/60 | no (over-refusal) |
| 13 | **119** | 31/32 | 26/26 | 21/26 | 40/60 | no (over-refusal) |

**No seed qualified.** By the fallback in the rule, seed 13 was chosen: the highest dev total among seeds with dev
harmful ≥ 90% (all three had it). **The dev bar was not met.** Over-refusal varied by 6 items between seeds, which
matches Phase 14's finding that this measure is noisy between identical training runs.

## The sealed v3 sets, scored once

All four rows were scored in the same run. Over-refusal uses `score_overrefusal`; everything else scorer_v4.

| | Chosen SFT v5 (seed 13) | + safety layer (report-only) | v1.0.1 with layer (as released) | v1.0.1 without layer |
| :--- | ---: | ---: | ---: | ---: |
| **Bare arithmetic (bar ≥ 90%)** | **16/18 — FAIL** | 16/18 | 17/18 | 17/18 |
| **harmful_v3 (bar ≥ 90%)** | **57/60 — PASS** | 57/60 | 57/60 | 56/60 |
| **overrefusal_v3 (bar ≥ 90%)** | **77/100 — FAIL** | 77/100 | 76/100 | 76/100 |
| **Unknown (bar ≥ 70%)** | **26/36 — PASS** | 26/36 | 26/36 | 26/36 |
| Definitions | 10/30 | 10/30 | 8/30 | 8/30 |
| Facts | 8/36 | 8/36 | 4/36 | 4/36 |
| Instructions | 9/30 | 9/30 | 10/30 | 10/30 |
| Goal harmful (32) | 30/32 | 30/32 | 30/32 | 29/32 |
| Unknown: privacy / prediction / false premise | 10 / 11 / 5 of 12 | same | 11 / 11 / 4 | 11 / 11 / 4 |
| Over-refusal: edgy / emotional | 53/70, 24/30 | same | 53/70, 23/30 | 53/70, 23/30 |
| harmful_v3: refusal / self-harm support | 52/54, 5/6 | same | 52/54, 5/6 | 52/54, 4/6 |

- **Bars for the chosen model: 2 of 4 pass.** Only the first column counts.
- **The safety layer** (the report-only system row) fired on 9 prompts but changed no score: on those prompts the model
  alone already gave an acceptable reply. For v1.0.1 the layer adds one self-harm support item (4/6 → 5/6).

### The 18 multi-step word problems — not independent of SFT v5 templates

Reported separately and excluded from the bars, because the SFT v5 multi-step templates share scenarios with these
items (both were written in the same session).

| | Word problems (18) |
| :--- | ---: |
| Chosen SFT v5 | 10/18 |
| v1.0.1 | 1/18 |

### GSM8K — the independent multi-step test

The same 500 questions as Phase 18, chat runtime, exact match on the final number.

| | With calculator | Calculator blocked |
| :--- | ---: | ---: |
| Chosen SFT v5 | 1.6% ± 0.6 | 1.4% ± 0.5 |
| v1.0.1 (Phase 18; plain greedy decoding) | 0.6% ± 0.3 | 0.6% ± 0.3 |

Only 10 of the 500 replies differ between the two calculator settings. The gain on the 18 template-like word problems
(1 → 10) does **not** carry over to GSM8K, which is the clearest sign that the word-problem improvement is mostly
template familiarity. (v1.0.1's GSM8K was run in Phase 18 with plain greedy decoding, not no-repeat-3-gram.)

## Phase 18 multiple-choice benchmarks of the chosen model (reported, not required to improve)

| Task | Chosen SFT v5 | Its base (17D extended) | v1.0.1 |
| :--- | ---: | ---: | ---: |
| ARC-Easy | 37.2 ± 1.0 | 44.0 ± 1.0 | 36.2 ± 1.0 |
| ARC-Challenge | 23.8 ± 1.2 | 23.8 ± 1.2 | 23.1 ± 1.2 |
| HellaSwag | 27.9 ± 0.4 | 31.2 ± 0.5 | 27.4 ± 0.4 |
| PIQA | 56.6 ± 1.2 | 61.4 ± 1.1 | 56.4 ± 1.2 |
| WinoGrande | 47.5 ± 1.4 | 52.5 ± 1.4 | 51.4 ± 1.4 |
| SciQ | 59.3 ± 1.6 | 67.8 ± 1.5 | 57.5 ± 1.6 |
| OpenBookQA | 23.6 ± 1.9 | 29.2 ± 2.0 | 26.2 ± 2.0 |
| BoolQ | 56.6 ± 0.9 | 59.3 ± 0.9 | 54.5 ± 0.9 |
| LAMBADA | 12.8 ± 0.5 | 27.0 ± 0.6 | 10.8 ± 0.4 |
| MMLU | 25.6 ± 0.4 | 23.0 ± 0.4 | 24.0 ± 0.4 |

Instruction tuning again cost most of what pretraining gained on these tests: the chosen model is below its own base
on most tasks (LAMBADA 27.0 → 12.8, SciQ 67.8 → 59.3, ARC-Easy 44.0 → 37.2) and ends close to v1.0.1. The same happened
from the 34M base to v1.0.1 in Phase 18.

## What failed, plainly

1. **Over-refusal**: 77/100 on the sealed set and 37–43/60 on dev, against a 90% bar. The bigger, better-pretrained
   base did not fix it; nor did the mined harmless examples in SFT v5. It is the same weakness v1.0.1 has (76/100).
2. **Bare arithmetic**: 16/18, one item short of the bar (v1.0.1: 17/18).
3. **No independent multi-step gain**: GSM8K stays near zero with or without the calculator.

## Release decision

**No release.** CRITERIA_17C requires all four bars; two fail. No v2.0 demo package was built. MiniGPT v1.0.1 remains
the released model.

## Files

- `phase17/CRITERIA_17C.md`; `phase17/configs/sft_v5_s{11,12,13}.json`; `phase17/run_17c.sh`; `phase17/eval_17c.py`
- `phase17/results/phase17c/dev_s{11,12,13}.json` (+ dev responses), `selection.json`, `sealed.json` (totals only)
- `phase18/results/sft_v5_chosen/` (GSM8K with and without calculator, multiple-choice tasks)
- `phase17/runs/sft_v5_s{11,12,13}/metrics.json`; logs in `phase17/logs/`

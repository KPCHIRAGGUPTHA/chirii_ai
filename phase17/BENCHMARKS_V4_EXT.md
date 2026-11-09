# Phase 17D — Foundation v4 extended: evaluation

Runs: the college GPU PC (RTX A2000 12 GB), 9 November 2026, after the extended run finished (09:40). No fine-tuning.
Criteria: `phase17/CRITERIA_17D.md`, committed on 5 November before any 17D data was collected or GPU run started.

## In short

**The extended model again fails the pre-registered goal of beating GPT-2: all three conditions fail.** It is
clearly better than v4 — significantly better on ARC-Easy (+4.3), HellaSwag (+1.9) and LAMBADA (+7.3 points), and 5.7%
lower bits per byte — and it now matches GPT-2 on HellaSwag and beats it on ARC-Easy. But it is still significantly
behind GPT-2 on LAMBADA (27.0% vs 32.6%), better on only one of the five named tasks, and at chance on ARC-Challenge.

As the criteria say, this run changed two things at once (8B more tokens, and 30% general FineWeb), so it cannot tell
whether the gain came from the extra tokens or from the broader data.

## The run

- Started from Foundation v4 (98,425,344 parameters, step 5,722) with a fresh optimizer; warmup 300 steps to 3e-4,
  cosine to 3e-5; everything else as v4.
- 15,258 steps × 524,288 = 7,999,586,304 tokens, from a new 8.0B-token mix: 70% FineWeb-Edu and 30% general FineWeb,
  mixed at document level. Every document already used, every Phase 5D-overlap document and 8,484 documents sharing a
  13-word sequence with the 12 benchmarks were removed first (counts in `phase17/manifests17d/filter_manifest.json`).
- 80.5 hours of training at 27,618 tokens/s (80.7 hours on the clock; v4 trained for 30.0 hours); finished 9 Nov 09:40, about 18 hours before the deadline; no
  restarts. The launch gate passed (step-100 bits per byte 1.0616 ≤ 1.0896; max gradient norm after step 50: 0.32).
- Weights: private release `weights-2026-11-09` (verified by download).

## Bits per byte on the same texts

Identical scored tokens for every row; float32. Lower is better.

| Model | Context | Phase 10 validation set | Phase 5D validation text |
| :--- | ---: | ---: | ---: |
| MiniGPT base (34M) | 512 | 1.1335 | 1.1479 |
| v4 (98M, 3B tokens) | 512 | 1.0798 | 1.0960 |
| v4 (98M, 3B tokens) | 1,024 | 1.0596 | 1.0777 |
| v4 extended (98M, +8B tokens) | 512 | 1.0198 | 1.0357 |
| v4 extended (98M, +8B tokens) | 1,024 | 0.9992 | 1.0167 |

At its full 1,024-token context the extended model is 0.0604 bits per byte better than v4 on the Phase 10
validation set (5.7%), and 0.0609 better on the Phase 5D text.

## Standard benchmarks (Phase 18 harness, unchanged)

Same tasks, settings and subsets as Phase 18. GPT-2, v4 and MiniGPT 34M numbers are the committed Phase 18 / 17B results.
The 1,024-token window cut only 20 of 56,168 MMLU requests and 2 of 6,540 BoolQ requests.

| Task | Random | v4 extended (98M, 3B+8B tokens) | GPT-2 (124M) | v4 (98M, 3B tokens) | MiniGPT base (34M) | ext − GPT-2 | ext vs GPT-2 | ext − v4 | ext vs v4 | ext vs 34M |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- | :--- |
| ARC-Easy | 25.0 | 44.0 ± 1.0 | 39.5 ± 1.0 | 39.6 ± 1.0 | 38.6 ± 1.0 | +4.5 ± 1.4 | better | +4.3 ± 1.4 | better | better |
| ARC-Challenge | 25.0 | 23.8 ± 1.2† | 22.7 ± 1.2† | 24.1 ± 1.2† | 22.1 ± 1.2† | +1.1 ± 1.7 | not significant | -0.3 ± 1.8 | not significant | not significant |
| HellaSwag | 25.0 | 31.2 ± 0.5 | 31.1 ± 0.5 | 29.3 ± 0.5 | 27.7 ± 0.4 | +0.1 ± 0.7 | not significant | +1.9 ± 0.6 | better | better |
| PIQA | 50.0 | 61.4 ± 1.1 | 62.5 ± 1.1 | 59.0 ± 1.1 | 57.0 ± 1.2 | -1.1 ± 1.6 | not significant | +2.3 ± 1.6 | not significant | better |
| WinoGrande | 50.0 | 52.5 ± 1.4† | 51.6 ± 1.4† | 51.1 ± 1.4† | 49.7 ± 1.4† | +0.9 ± 2.0 | not significant | +1.3 ± 2.0 | not significant | not significant |
| SciQ | 25.0 | 67.8 ± 1.5 | 64.4 ± 1.5 | 63.9 ± 1.5 | 60.3 ± 1.5 | +3.4 ± 2.1 | not significant | +3.9 ± 2.1 | not significant | better |
| OpenBookQA | 25.0 | 29.2 ± 2.0 | 27.2 ± 2.0† | 27.4 ± 2.0† | 27.8 ± 2.0† | +2.0 ± 2.8 | not significant | +1.8 ± 2.9 | not significant | not significant |
| BoolQ | 50.0 | 59.3 ± 0.9 | 48.7 ± 0.9† | 60.8 ± 0.9 | 59.8 ± 0.9 | +10.6 ± 1.2 | better | -1.5 ± 1.2 | not significant | not significant |
| LAMBADA | 0.0 | 27.0 ± 0.6 | 32.6 ± 0.7 | 19.7 ± 0.6 | 17.3 ± 0.5 | -5.6 ± 0.9 | worse | +7.3 ± 0.8 | better | better |
| MMLU | 25.0 | 23.0 ± 0.4† | 22.9 ± 0.4† | 23.1 ± 0.4† | 23.2 ± 0.4† | +0.1 ± 0.5 | not significant | -0.0 ± 0.5 | not significant | not significant |
| GSM8K (500) | 0.0 | 1.0 ± 0.4 | 1.6 ± 0.6 | 1.0 ± 0.4 | 0.8 ± 0.4 | -0.6 ± 0.7 | not significant | +0.0 ± 0.6 | not significant | not significant |
| TriviaQA (1,000) | 0.0 | 0.1 ± 0.1† | 0.3 ± 0.2† | 0.3 ± 0.2† | 0.0 ± 0.0† | -0.2 ± 0.2 | not significant | -0.2 ± 0.2 | not significant | not significant |

## Pre-registered verdict (CRITERIA_17D.md: the 17B conditions applied to the extended model)

- Condition 1 (better than GPT-2 on ≥ 3 of HellaSwag, PIQA, LAMBADA, ARC-Easy, SciQ): significantly better on 1 (ARC-Easy) — **FAIL**
- Condition 2 (significantly worse than GPT-2 on none of 12): worse on 1 (LAMBADA) — **FAIL**
- Condition 3 (ARC-Challenge significantly above random): 23.8 ± 1.2 vs 25.02 (-0.97 standard errors above) — **FAIL**
- **Success: NO**

## Generations from the five fixed prompts

Greedy, 64 new tokens, the trainer's generation function, verbatim.

**Prompt:** `'The capital of France is'`

`'The capital of France is the city of Lorraine, which is located in the heart of the country. The city is located in the heart of the country, and is the capital of the French Republic. The city is located in the heart of the country, and is the capital of the French Republic. The city is located in the heart'`

**Prompt:** `'Water is made of hydrogen and'`

`'Water is made of hydrogen and oxygen.\nThe hydrogen is made of oxygen and hydrogen.\nThe oxygen is made of hydrogen and oxygen.\nThe hydrogen is made of oxygen and hydrogen.\nThe hydrogen is made of oxygen and hydrogen.\nThe hydrogen is made of oxygen and hydrogen.\nThe hydrogen is made of oxygen and hydrogen.\nThe'`

**Prompt:** `'2 + 2 ='`

`'2 + 2 = 100000000000000000000000000000000000000000000000000000000000000'`

**Prompt:** `'The main reason plants need sunlight is'`

`'The main reason plants need sunlight is because they are able to absorb the sun’s energy. Plants need to be able to absorb the sun’s energy. Plants need to be able to absorb the sun’s energy. Plants need to be able to absorb the sun’s energy. Plants need to be able to absorb the sun’s energy.'`

**Prompt:** `'def add(a, b):\n'`

`'def add(a, b):\nThe add() function returns the add() function.\nThe add() function returns the add() function.\nThe add() function returns the add() function.\nThe add() function returns the add() function.\nThe add() function returns the add() function.\nThe add() function returns the add() function'`

The extended model still repeats itself under greedy decoding, still gets the capital of France wrong ("Lorraine";
v4 said Luxembourg, the 34M model Paris), and cannot add 2 + 2 or write code.

## An honest reading

1. **More data helped, measurably.** Across the board the extended model is better than v4 at predicting text, and on
   the benchmarks closest to that skill (LAMBADA, HellaSwag, ARC-Easy) the gains are significant.
2. **It is not enough to beat GPT-2.** The LAMBADA gap shrank from 12.9 to 5.6 points but is still large and
   significant, and only ARC-Easy moved clearly past GPT-2.
3. **Reasoning and knowledge tests stay at chance** (ARC-Challenge, WinoGrande, MMLU, TriviaQA), for every model here.
4. **Cause not identified.** Tokens and data domain changed together, by design of this run.

## Files

- `phase17/CRITERIA_17D.md`; `phase17/runs/gate_17d.json`
- `phase17/eval_v4ext.sh` (all evaluation runs); `phase17/eval_bpb.py --include-ext` → `results/bpb_same_texts_ext.json`;
  `phase17/generate_fixed.py --include-ext` → `results/fixed_prompt_generations_ext.json`
- `phase17/report_17d.py` → `results/TABLES_17D.md`, `results/criteria_17d.json`; harness results in
  `phase18/results/minigpt_v4ext_base/`
- `phase17/runs/foundation_v4x/metrics.json`, `steps.jsonl`; `phase17/logs/foundation_v4x.log`

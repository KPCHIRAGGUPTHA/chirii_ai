# Phase 17B — Foundation v4 base: evaluation

Runs: the college GPU PC (RTX A2000 12 GB), 5 November 2026, after Foundation v4 finished pretraining (17:11). No
fine-tuning was done. Criteria: `phase17/CRITERIA_17B.md`, committed at 17:47 — after pretraining ended and before any
benchmark ran (see its note).

## In short

**Foundation v4 did not meet its pre-registered goal of beating GPT-2.** All three conditions in `CRITERIA_17B.md`
fail. On the 12 standard benchmarks it is level with GPT-2 on 8, significantly behind on 3 (LAMBADA, PIQA, HellaSwag)
and ahead only on BoolQ, where both models are below the 62.2% "always answer yes" baseline.

v4 is a better language model than the 34M MiniGPT — its bits per byte fell by about 6.5% on the same text — but that
gain shows up on only two benchmarks (HellaSwag +1.6, LAMBADA +2.4 points). On the knowledge and reasoning tests both
models stay at chance.

## The model

| | MiniGPT base (34M) | Foundation v4 base |
| :--- | ---: | ---: |
| Parameters | 33,870,848 | 98,425,344 |
| Layers × heads × width | 8 × 8 × 512 | 12 × 12 × 768 |
| Context | 512 tokens | 1,024 tokens |
| Pretraining tokens | 1.51B (FineWeb-Edu, 2 epochs of 755M) | 3.00B (FineWeb-Edu, about 1 pass over 2.99B) |
| Training time | 5 h 47 min | 30 h 05 min (27,774 tokens/s) |

v4 has GPT-2's shape (12 layers, 12 heads, 768 wide, 1,024 context) but our 16,384-token vocabulary, so it has
98.4M parameters rather than GPT-2's 124M. Weights: private release `weights-2026-11-05` (verified by download).

## Bits per byte on the same texts

Lower is better. Both models score exactly the same tokens and bytes: the 3,788,800 tokens of the
Phase 10 validation set and the 839,680 tokens of the Phase 5D validation text (LF form),
float32. The 512-token rows give both models the same context; the last row is v4 with its own 1,024-token context.

| Model | Context | Phase 10 validation set | Phase 5D validation text |
| :--- | ---: | ---: | ---: |
| MiniGPT base (34M) | 512 | 1.1335 | 1.1479 |
| Foundation v4 base (98M) | 512 | 1.0798 | 1.0960 |
| Foundation v4 base (98M) | 1,024 | 1.0596 | 1.0777 |

The 34M model's 1.1479 on the Phase 5D text is exactly its Phase 10 figure, so the measurement matches the earlier one.
v4 is 0.054 bits per byte better at equal context (about 4.7%) and 0.074 better with its full context (6.5%).

## Standard benchmarks (Phase 18 harness, unchanged)

Same tasks, settings and subsets as Phase 18: lm-evaluation-harness 0.4.13, 0-shot, float32, greedy, plain prompts;
the 10 multiple-choice tasks on full test sets; GSM8K and TriviaQA on the same fixed 500 and 1,000 questions. GPT-2 and
MiniGPT 34M numbers are the Phase 18 results. v4's 1,024-token window cut only 20 of 56,168 MMLU requests and 2 of 6,540
BoolQ requests, and nothing elsewhere.

| Task | Random | Foundation v4 base (98M) | GPT-2 (124M) | MiniGPT base (34M) | v4 − GPT-2 | v4 vs GPT-2 | v4 − 34M | v4 vs 34M |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| ARC-Easy | 25.0 | 39.6 ± 1.0 | 39.5 ± 1.0 | 38.6 ± 1.0 | +0.2 ± 1.4 | not significant | +1.0 ± 1.4 | not significant |
| ARC-Challenge | 25.0 | 24.1 ± 1.2† | 22.7 ± 1.2† | 22.1 ± 1.2† | +1.4 ± 1.7 | not significant | +2.0 ± 1.7 | not significant |
| HellaSwag | 25.0 | 29.3 ± 0.5 | 31.1 ± 0.5 | 27.7 ± 0.4 | -1.8 ± 0.6 | worse | +1.6 ± 0.6 | better |
| PIQA | 50.0 | 59.0 ± 1.1 | 62.5 ± 1.1 | 57.0 ± 1.2 | -3.5 ± 1.6 | worse | +2.0 ± 1.6 | not significant |
| WinoGrande | 50.0 | 51.1 ± 1.4† | 51.6 ± 1.4† | 49.7 ± 1.4† | -0.5 ± 2.0 | not significant | +1.4 ± 2.0 | not significant |
| SciQ | 25.0 | 63.9 ± 1.5 | 64.4 ± 1.5 | 60.3 ± 1.5 | -0.5 ± 2.1 | not significant | +3.6 ± 2.2 | not significant |
| OpenBookQA | 25.0 | 27.4 ± 2.0† | 27.2 ± 2.0† | 27.8 ± 2.0† | +0.2 ± 2.8 | not significant | -0.4 ± 2.8 | not significant |
| BoolQ | 50.0 | 60.8 ± 0.9 | 48.7 ± 0.9† | 59.8 ± 0.9 | +12.0 ± 1.2 | better | +0.9 ± 1.2 | not significant |
| LAMBADA | 0.0 | 19.7 ± 0.6 | 32.6 ± 0.7 | 17.3 ± 0.5 | -12.9 ± 0.9 | worse | +2.4 ± 0.8 | better |
| MMLU | 25.0 | 23.1 ± 0.4† | 22.9 ± 0.4† | 23.2 ± 0.4† | +0.1 ± 0.5 | not significant | -0.1 ± 0.5 | not significant |
| GSM8K (500) | 0.0 | 1.0 ± 0.4 | 1.6 ± 0.6 | 0.8 ± 0.4 | -0.6 ± 0.7 | not significant | +0.2 ± 0.6 | not significant |
| TriviaQA (1,000) | 0.0 | 0.3 ± 0.2† | 0.3 ± 0.2† | 0.0 ± 0.0† | +0.0 ± 0.2 | not significant | +0.3 ± 0.2 | not significant |

## Pre-registered verdict (CRITERIA_17B.md, applied exactly as committed)

- Condition 1 (better than GPT-2 on ≥ 3 of HellaSwag, PIQA, LAMBADA, ARC-Easy, SciQ): significantly better on 0 (none) — **FAIL**
- Condition 2 (significantly worse than GPT-2 on none of 12): worse on 3 (HellaSwag, PIQA, LAMBADA) — **FAIL**
- Condition 3 (ARC-Challenge significantly above random): 24.1 ± 1.2 vs 25.02 (-0.76 standard errors above) — **FAIL**
- **Success: NO**

## Generations from the five fixed prompts

Greedy, 64 new tokens, the trainer's own generation function (bf16), verbatim. The 34M base is shown for comparison.

**Prompt:** `'The capital of France is'`

- v4: `'The capital of France is the city of Luxembourg.\nThe city of Luxembourg is located in the heart of the city of Luxembourg.\nThe city of Luxembourg is located in the heart of the city of Luxembourg.\nThe city of Luxemb'`
- 34M: `'The capital of France is the city of Paris. The city is the capital of the French Republic. The capital of the French Republic is the city of Paris. The capital of the French Republic is the city of Paris. The capital of the French Republic is the city of Paris. The capital of the French Republic is the city of Paris. The'`

**Prompt:** `'Water is made of hydrogen and'`

- v4: `'Water is made of hydrogen and oxygen.\nThe hydrogen is a mixture of hydrogen and oxygen. The oxygen is the same as the oxygen in the air.\nThe oxygen is the same as the oxygen in the air.\nThe oxygen is the same as the oxygen in the air.\nThe oxygen is the same as the oxygen in the air.'`
- 34M: `'Water is made of hydrogen and oxygen.\nThe hydrogen is made of hydrogen and oxygen.\nThe hydrogen is made of hydrogen and oxygen.\nThe hydrogen is made of hydrogen and oxygen.\nThe hydrogen is made of hydrogen and oxygen.\nThe hydrogen is made of hydrogen and oxygen.\nThe hydrogen is made of hydrogen and oxygen.\nThe'`

**Prompt:** `'2 + 2 ='`

- v4: `'2 + 2 = 1000.\nThe first two numbers are the same, but the first two numbers are the same.\nThe second number is the same, but the first two numbers are the same.\nThe third number is the same, but the first two numbers are the same.\nThe third number is the'`
- 34M: `'2 + 2 = 100000000000000000000000000000000000000000000000000000000000000'`

**Prompt:** `'The main reason plants need sunlight is'`

- v4: `'The main reason plants need sunlight is because they are so sensitive to the sun’s rays.\nThe sun’s rays are absorbed by the leaves and the air in the plant. The sun’s rays are absorbed by the leaves and the air in the plant.\nThe sun’s rays are absorbed by the leaves and the air in the plant'`
- 34M: `'The main reason plants need sunlight is because they are able to absorb sunlight.\nThe plants that grow in the sun are called photosynthesis.\nThe plants that grow in the sun are called photosynthesis.\nThe plants that grow in the sun are called photosynthesis.\nThe plants that grow in the sun are called photosynthesis.\nThe plants that'`

**Prompt:** `'def add(a, b):\n'`

- v4: `'def add(a, b):\nThe first step is to add a new element to the element.\nThe second step is to add a new element to the element.\nThe third step is to add a new element to the element.\nThe third step is to add a new element to the element.\nThe third step is to add a new'`
- 34M: `'def add(a, b):\n- 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.'`

Both models repeat themselves, as small base models do with greedy decoding. v4 completes "The capital of France is"
with Luxembourg where the 34M model says Paris, and neither can add 2 + 2 or write code.

## An honest reading

1. **Better at predicting text, barely better at the tests.** v4's bits per byte improved clearly, but most benchmarks
   moved by less than their noise. The two that improved significantly, HellaSwag and LAMBADA, are the ones closest
   to plain next-word prediction.
2. **The LAMBADA gap to GPT-2 is still large (19.7% vs 32.6%).** v4 has GPT-2's shape and a 1,024-token window, so the
   window and depth alone are not the explanation. The remaining differences include the training data (FineWeb-Edu
   is filtered for educational web pages, while LAMBADA's passages come from novels), the amount of text seen, and the
   smaller vocabulary. This phase does not test which of these matters.
3. **At chance on knowledge and reasoning.** ARC-Challenge, WinoGrande, OpenBookQA, MMLU and TriviaQA stay at
   random for v4, GPT-2 and the 34M model alike. Condition 3 (ARC-Challenge above random) was the most demanding
   part of the criteria and is not met: 24.1% ± 1.2.
4. **What this does not show.** One training run and one seed; 0-shot only; no fine-tuning yet. It does not say how
   v4 will do after instruction tuning.

Fine-tuning (SFT) has not been started.

## Files

- `phase17/CRITERIA_17B.md` — the pre-registered conditions and how they are computed
- `phase17/eval_bpb.py` → `results/bpb_same_texts.json`; `phase17/generate_fixed.py` → `results/fixed_prompt_generations.json`
- `phase17/eval_v4.sh` — every 17B GPU run; harness results in `phase18/results/minigpt_v4_base/`
- `phase17/report_17b.py` → `results/TABLES_17B.md`, `results/criteria_17b.json`
- `phase17/runs/foundation_v4/metrics.json`, `steps.jsonl`, `generations.jsonl`; `phase17/logs/foundation_v4.log`

# Phase 17B — Foundation v4 base evaluation: pre-registered criteria

> Committed after pretraining ended (17:11, 5 Nov 2026) and before any benchmark was run. The conditions were set on 4 Nov 2026 in the planning conversation and are unchanged. At the time of commit, only the training validation loss (1.0621 bits per byte) had been seen.

## Conditions

v4 base vs GPT-2 on the Phase 18 harness (same tasks, settings and subsets). Success = significantly better (> 2 stderr) on at least 3 of {HellaSwag, PIQA, LAMBADA, ARC-Easy, SciQ}, significantly worse on none of the 12 tasks, and significantly above random on ARC-Challenge. Also report vs MiniGPT 34M base.

## How the conditions are computed (operational definitions; they do not change the conditions)

- **Harness**: exactly as in Phase 18 (`phase18/run_all.sh`, lm-evaluation-harness 0.4.13, MiniGPT adapter
  `phase18/minigpt_lm.py`): 0-shot, float32, greedy; the 10 multiple-choice / loglikelihood tasks on their full test
  sets with plain prompts; GSM8K and TriviaQA on the fixed subsets in `phase18/subsets.json` (500 and 1,000 items,
  seed 18) with plain prompts (v4 is a base model), capped at 256 and 64 generated tokens.
- **Metric per task**: as fixed in `phase18/report.py` — length-normalised accuracy for ARC-Easy, ARC-Challenge,
  HellaSwag, PIQA, SciQ and OpenBookQA; accuracy for WinoGrande, BoolQ, LAMBADA and MMLU; GSM8K exact match
  (flexible-extract); TriviaQA exact match.
- **GPT-2 and MiniGPT 34M base numbers**: the Phase 18 results already committed, not re-run.
- **"Significantly better / worse"** on a task: |v4 − GPT-2| > 2 × sqrt(se_v4² + se_GPT-2²), with the sign giving
  better or worse; standard errors as reported by lm-eval (or by `phase18/subset_scores.py` for the two subsets).
- **"Significantly above random" on ARC-Challenge**: (v4 − 25.02) > 2 × se_v4, where 25.02% is the random-guessing
  expectation in `phase18/results/random_baselines.json`.
- **Success** requires all three conditions; each is reported as PASS or FAIL.

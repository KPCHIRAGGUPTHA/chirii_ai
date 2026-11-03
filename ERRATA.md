# Errata

The release documents for v1.0 (`phase12/`) and v1.0.1 (`phase14/`) are frozen and are not edited. Corrections
to them are listed here. The top-level `README.md` already uses the corrected wording.

## 1. "The raw pretrained models score 0 on the goal evaluation"

**Where:** identical text in both release reports:

- `phase12/MINIGPT_V1_REPORT.md`, lines 50–52 (v1.0)
- `phase14/MINIGPT_V1_REPORT.md`, lines 50–52 (v1.0.1 copy)

**Original text:**

> **Instruction tuning took the model from 0 to useful: 0 → 102/180.** The raw pretrained models score 0 on
> the goal evaluation, as expected for a model with no instruction training (`phase9/GPU_RESULTS.md`). After
> tuning, the released model answers 102 of 180 held-out prompts correctly (`phase11/PHASE11C_REPORT.md`).

**What is wrong:**

- The cited source, `phase9/GPU_RESULTS.md`, scores the Phase 9 retrains (C_v2 and D_v2, seeds 42 and 43), not the
  34M model that was later instruction-tuned. The 34M base model (`model_step_11524.pt`) was never scored on the
  goal evaluation: `phase10/eval_results/eval_v3.json` holds only its bits-per-byte results, and no Phase 10 or
  Phase 11 report gives it a goal score.
- "0 → 102/180" compares two different test sets: the earlier 202-prompt goal evaluation and the 180-prompt
  held-out set.

**Corrected text:**

> Earlier models — Model C, the first fine-tuned model (Phase 6 SFT) and the Phase 9 retrains — scored 0/202 on
> the earlier goal evaluation (`phase9/goal_eval/results_v2/GOAL_EVAL_V2.md`, `phase9/GPU_RESULTS.md`); the 34M
> base model was not scored on it. After instruction tuning, the released model answers 102 of 180 held-out
> prompts correctly (`phase11/PHASE11C_REPORT.md`).

## 2. Other documents checked

These v1.0.1 documents were checked for anything else the README corrections changed (the goal-evaluation
baseline and the Phase 13 rebalanced-seed results):

- `phase14/MODEL_CARD.md`
- `phase14/DEMO_SCRIPT.md`
- `phase14/dist/minigpt_v1_demo/README.md`
- `phase12/MODEL_CARD.md` and `phase12/DEMO_SCRIPT.md` (the v1.0 versions)

**Result:** none of them states a goal-evaluation score for the base model or the Phase 13 harmful-request results.
No further errata.

## 3. Related clarifications in later, non-release documents

- **Phase 13 harmful-request results.** `POST_RELEASE_EXPERIMENTS.md` gives the sealed harmful score as "52/59 vs
  46/59", which covers only the first rebalanced (v4a) seed. Both seeds were scored: 46/59 (seed 1) and 52/59
  (seed 2), against 52/59 for the released model (`phase13/PHASE13_REPORT.md`, section 6). The reduced
  harmful-request refusal therefore showed up in one of the two seeds, not both.
- **Simple English Wikipedia licence.** `phase15/PHASE15A_REPORT.md` and `POST_RELEASE_EXPERIMENTS.md` give the
  corpus licence as CC BY-SA 4.0. The dataset's Hugging Face page (`wikimedia/wikipedia`) lists CC BY-SA 3.0 and
  GFDL, which the README uses.

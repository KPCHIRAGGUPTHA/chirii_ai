# Errata

The release documents for v1.0 (`phase12/`) and v1.0.1 (`phase14/`) are frozen and are not edited. Corrections
to them, and to later documents that newer work has superseded, are listed here. The top-level `README.md`
already uses the corrected wording.

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

## 4. Superseded by later work (added 10 November 2026)

These documents are records of their phase and are not edited; the corrections below apply from now on.

- **"Foundation v4 (124M)".** The title of `phase17/CRITERIA_17A.md` and a sentence in `phase18/BENCHMARKS.md`
  ("The next foundation model is GPT-2-sized (124M parameters)") give 124M. Foundation v4 has GPT-2's shape
  (12 layers, 12 heads, 768 wide, 1,024 context) but this project's 16,384-token vocabulary, so it has
  **98,425,344 parameters** (98.4M), not 124M (`phase17/BENCHMARKS_V4.md`).
- **"A larger foundation model" as an unbuilt next step.** The README roadmap and `POST_RELEASE_EXPERIMENTS.md`
  list a larger model as future work. It has since been built and evaluated: Foundation v4 (98.4M, 3.0B tokens) and
  its 8.0B-token extension (`phase17/BENCHMARKS_V4.md`, `phase17/BENCHMARKS_V4_EXT.md`). Neither met the
  pre-registered goal of beating GPT-2, and instruction tuning on the extended model did not pass its release bars
  (`phase17/PHASE17C_REPORT.md`). Retrieval has not been retried on the larger model.
- **"A reasonable v2.0 target"** in `phase18/BENCHMARKS.md` (match or beat GPT-2 on LAMBADA, HellaSwag, PIQA and
  SciQ): the extended base model reached it on HellaSwag, PIQA and SciQ (no significant difference) but not on
  LAMBADA (27.0 vs 32.6) (`phase17/BENCHMARKS_V4_EXT.md`). No v2.0 was released.
- **Late criteria commits.** `phase17/CRITERIA_17B.md` was committed after Foundation v4's pretraining ended (but
  before any benchmark), and `phase17/CRITERIA_17C.md` on 6 November; both files say so at the top. See
  `FINAL_REPORT.md`, "Mistakes made in planning".

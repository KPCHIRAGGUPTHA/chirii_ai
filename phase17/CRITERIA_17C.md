# Phase 17C — SFT v5 on the extended base: pre-registered criteria

> These conditions were set on 4 Nov 2026 in the planning conversation; the prep prompt carrying them wasn't sent, so this is their first commit, before any SFT run.

Committed on 6 November 2026, while the 17D extended pretraining run is still training and before any Phase 17C run.

## Conditions

**Base and training.** Base: the final model of the 17D extended run. Train SFT v5 with 3 seeds, same hyperparameters
as sft_v2.

**Selection on DEV only** (goal dev v2.1 + old over-refusal set, scored with scorer_v4). A seed qualifies if dev
arithmetic ≥ 90%, dev harmful ≥ 90%, old over-refusal ≥ 90% and dev unknown ≥ 70%. Pick the qualifier with the
highest dev total. If none qualifies, pick the highest dev total among seeds with dev harmful ≥ 90%; if none, the
highest dev total, and state plainly that the bar was not met.

**Held-out, once.** The chosen seed is scored ONCE with scorer_v4 on goal_heldout_v3, overrefusal_v3 and harmful_v3.
Bars: bare arithmetic ≥ 90%, harmful_v3 ≥ 90%, overrefusal_v3 ≥ 90%, unknown ≥ 70%.

**Multi-step word problems.** The 18 sealed v3 multi-step word problems are scored and reported SEPARATELY, labelled
"not independent of SFT v5 templates", and excluded from the bars. The independent multi-step measure is GSM8K (the
same 500-item subset as Phase 18), with and without the calculator.

**Also report.** MiniGPT v1.0.1 on the same sealed v3 sets with scorer_v4 (as a baseline, in the same run), and the
Phase 18 multiple-choice benchmarks of the chosen SFT model (reported, not required to improve).

## Operational definitions (how the conditions are computed; they do not change them)

- **Base weights**: the 17D run's weights at its last step (`phase17/runs/foundation_v4x/snapshots/` final snapshot),
  pinned by SHA-256 in the 17C configs.
- **SFT data**: SFT v5 as pinned in `phase17/manifests/sft_v5_manifest.json` (train SHA-256 `3b2740c0…`, val
  `2ed40e71…`), unchanged.
- **Hyperparameters** (from `phase11/configs/sft_v2.json`): 2 epochs, micro-batch 16 × accumulation 2, learning rate
  2e-4 with 50 warmup steps and cosine decay to 2e-5, weight decay 0, betas (0.9, 0.95), gradient clip 1.0, compile on.
  **Seeds 11, 12 and 13** (11 is sft_v2's seed). Trainer: `phase11/sft_train.py`.
- **Decoding** for every 17C generation: MiniGPT's chat runtime with the calculator, greedy with no-repeat-3-gram
  (the released v1.0.1 decoding).
- **Dev scores**: goal dev v2.1 without its 26 RAG items (176 items), categories as in Phase 11; goal items scored
  with scorer_v4 (`phase17/scorer_v4/scorer_v4.py`, frozen SHA-256 `b1f1219d…`). "Dev total" = correct goal dev items
  out of 176. **Over-refusal items** (the old 60-item set, and overrefusal_v3) are scored with
  `phase11/overrefusal/score_overrefusal.py`, unchanged, because scorer_v4 has no over-refusal rule; it is the scorer
  used for every over-refusal figure so far.
- **Held-out bars**: bare arithmetic = the 18 `arithmetic`/`bare` items of goal_heldout_v3; unknown = its 36
  `unknown` items; harmful_v3 = all 60 items (refusal and self-harm support, scorer_v4); overrefusal_v3 = all 100
  items. The other goal_heldout_v3 categories (definitions, facts, instructions, its 32 harmful items) are reported,
  not barred. Sealed sets are loaded only through the hash-checked loader and their prompts are never printed.
- **"Scored ONCE"**: one run per model on the sealed v3 sets; no re-runs, no changes after seeing results.
- **v1.0.1 baseline**: the released model (sft_v2) with its self-harm safety layer as shipped, and also without the
  layer, in the same run, same decoding, same scorers. The new SFT model has no safety layer.
- **GSM8K**: the Phase 18 harness and subset (`phase18/subsets.json`, 500 items), chat runtime, calculator on and
  calculator blocked, exact match (flexible-extract).
- **Phase 18 multiple-choice benchmarks**: the 10 tasks exactly as in Phase 18 (plain prompts, full test sets).

# Phase 17D — extended pretraining of Foundation v4: pre-registered criteria

Written and committed on 5 November 2026 **before any Phase 17D GPU run** (probe, gate or long run) and before any
Phase 17D data was collected. Nothing below may be changed after a GPU result is seen; any deviation will be reported.

## Honest note

17B failed: Foundation v4 (98.4M parameters, 3.0B FineWeb-Edu tokens) met none of the three conditions against GPT-2.
Two hypotheses: **too few tokens** (3B) and **data domain** (FineWeb-Edu has little narrative or everyday text). This
run changes **both** at once (more tokens, and 30% general FineWeb), so its result **cannot tell the two apart**.

## Evaluation of the final extended model

- **The 17B conditions vs GPT-2, verbatim:** "v4 base vs GPT-2 on the Phase 18 harness (same tasks, settings and
  subsets). Success = significantly better (> 2 stderr) on at least 3 of {HellaSwag, PIQA, LAMBADA, ARC-Easy, SciQ},
  significantly worse on none of the 12 tasks, and significantly above random on ARC-Challenge." — applied to the
  final extended model in place of v4 base, computed exactly as defined in `phase17/CRITERIA_17B.md`.
- **Also reported (not a condition):** v4 (3B tokens) vs the extended model on every task, difference ± sqrt(se1² +
  se2²), significant when |difference| > 2 × that.
- **"Final extended model"** = the weights at the last training step (not the best validation step).

## Launch gate (100-step probe of the continued run)

| Gate | Passes when |
| :--- | :--- |
| G1 finite | no NaN or Inf training loss or gradient norm at any of the 100 steps |
| G2 gradients | maximum gradient norm after step 50 ≤ 10 |
| G3 no damage | validation bits per byte at step 100 ≤ 1.0596 + 0.03 = 1.0896 |

G3 is measured exactly as the 1.0596 baseline was (`phase17/eval_bpb.py`): the full Phase 10 validation set
(`phase10/data/tokens/val.bin`, 3,788,800 tokens), non-overlapping 1,024-token windows, float32.
**All pass:** launch the long run automatically. **Any fails:** stop, launch nothing, report.

## Data (new documents only)

- New tokens: **70% more FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, sample-10BT, the Phase 10/17A revision
  `87f09149…`, continuing after the 2,684,592 stream documents already read) and **30% general FineWeb**
  (`HuggingFaceFW/fineweb`, sample-10BT, revision pinned when collection starts and recorded in the manifest).
- Excluded: every document already used (Phase 10 and Phase 17A, by text hash, for both sources), every document
  sharing a 40+ character line with the Phase 5D validation text, exact duplicates within the new data, empty
  documents and documents containing a literal special token.
- **Benchmark decontamination:** documents sharing any 13-word sequence with any of the 12 Phase 18 benchmark test
  sets (the exact evaluated text plus answer, as in `phase18/contamination.py`) are removed; counts are reported
  per source and per benchmark.
- Same train/val split rule as Phase 10 (SHA-256 of the document, first 8 bytes mod 1,000, below 5 is val).
- Tokenized with tokenizer v2, hashes pinned. Collected up to 8.0B train tokens in total (5.6B FineWeb-Edu, 2.4B
  FineWeb), then mixed **at document level** (documents interleaved in a seeded random order) at 70:30 by tokens;
  if one source falls short, the other is trimmed (whole documents) to keep 70:30.

## Run

- Start: the Foundation v4 step-5,722 weights (`phase17/runs/foundation_v4/snapshots/model_step_05722.pt`,
  SHA-256 `3c5d02dd…`), with a fresh optimizer state.
- Learning rate: linear warmup 300 steps to 3e-4, cosine decay to 3e-5 at the final step. Everything else as
  Foundation v4: 524,288 tokens per step (micro-batch 16 × accumulation 32 × 1,024), AdamW betas (0.9, 0.95), weight
  decay 0.1 on Linear weights only, gradient clip 1.0, bf16, compile, uniform random 1,024-token windows over the
  new training file, seed 1; validation on the same pinned Phase 10 subset; weights snapshots every 2,000 steps.
- **Token budget** = min(8.0B, measured tokens/s × hours until 10 November 2026 06:00 IST minus 2 hours), rounded
  down to whole steps.
  - measured tokens/s = training tokens of probe steps 21–100 ÷ their training time (the probe's first 20 steps
    include compilation);
  - hours are counted from the moment the budget is computed (right after the gate) to 10 Nov 2026 04:00 IST.
- The probe's 100 steps lie inside the 300-step warmup, so they do not depend on the final step count; the long run
  restarts from the step-5,722 weights with the final step count, rather than resuming the probe.
- Reboot safety: the `@reboot` crontab entry resumes only this run.

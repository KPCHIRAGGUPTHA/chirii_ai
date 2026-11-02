# Phase 13 decision rule (written before sft_v4a / sft_v4b were trained)

Committed before any v4 training. Decoding is fixed at no-repeat-3-gram for every model here. "DEV" for this
phase means the goal dev set (v2.1) and the OLD over-refusal set from Phase 11D (`phase11/overrefusal/`).
The sealed sets `overrefusal_v2` and `harmful_v2` are the FINAL test and are not used to choose anything.

Baselines (sft_v2, no-repeat-3-gram, from committed reports):
- dev goal total: 105/176 (`phase11/PHASE11C_REPORT.md`)
- dev harmful: 24/26 (`phase11/PHASE11C_REPORT.md`)
- old over-refusal: 42/60 (`phase11/PHASE11D_REPORT.md`)

A variant (v4a or v4b) **qualifies** if all three hold on DEV:
- old over-refusal >= 90% (>= 54/60), and
- dev harmful >= 90% (>= 24/26), and
- dev goal total >= 102 (that is v2's 105 minus 3).

Choice:
1. Among variants that qualify, pick the highest old over-refusal score. Tie: higher dev goal total; still tied, prefer v4a (simpler change).
2. If none qualify, pick the highest old over-refusal score among variants with dev harmful >= 90%, and state plainly that the bar was not met.
3. If still none (no variant has dev harmful >= 90%), keep sft_v2 and report that nothing beat it.

The chosen variant is then retrained with a second seed to check stability, and only then are the sealed
sets scored (once), alongside sft_v2 for comparison. The demo model for v1.0 remains sft_v2 regardless of this phase.

# Phase 14 Report: seed noise + self-harm safety layer (v1.0.1)

The released model weights stay **sft_v2 (seed 1)**; v1.0.1 adds a rule-based self-harm safety layer in front of
the model and documents seed noise. Branch `phase14-safety` from the `minigpt-v1.0` tag; `phase12/` and the
`minigpt-v1.0` tag are untouched. Every number is from this phase's runs or a cited report. Decoding is
no-repeat-3-gram throughout.

## A. Seed noise (dev only)

The exact v2 data and hyperparameters were retrained with seeds 2 and 3 (seed 1 = the released sft_v2, config
seed 11). All three scored on dev and the old over-refusal set.

| Metric | seed 1 | seed 2 | seed 3 | mean | min | max | range |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev goal total /176 | 105 | 105 | 105 | 105.0 | 105 | 105 | 0 |
| — arithmetic /32 | 31 | 30 | 30 | 30.3 | 30 | 31 | 1 |
| — definition /28 | 14 | 17 | 16 | 15.7 | 14 | 17 | 3 |
| — fact /34 | 10 | 9 | 8 | 9.0 | 8 | 10 | 2 |
| — instruction /30 | 7 | 7 | 7 | 7.0 | 7 | 7 | 0 |
| — unknown /26 | 19 | 17 | 19 | 18.3 | 17 | 19 | 2 |
| — harmful /26 | 24 | 25 | 25 | 24.7 | 24 | 25 | 1 |
| Old over-refusal /60 | 42 | 31 | 41 | 38.0 | 31 | 42 | 11 |

**Finding.** Goal accuracy is essentially seed-independent (total range 0, categories ≤3). But over-refusal
swings by **11 of 60 — 18 percentage points — between identical runs**. This is the noise floor: any
over-refusal difference smaller than that is not meaningful. In particular, Phase 13's v2/v4a/v4b over-refusal
scores (42/41/39) sit well inside this band, so that experiment could not have detected a real effect. The
seed 1 model is kept as the release regardless, as instructed.

## B. Self-harm safety layer

`phase14/safety_layer.py` runs on the user message before the model. On first-person self-harm or suicide
intent it returns a fixed supportive reply (warm, short, no judgement; Tele-MANAS 14416, emergency 112, and a
general fallback for people outside India; never any method information) and the model is not called.

- **Detection** is a conservative key-phrase list (documented in the code): fire on a direct self-harm request,
  or on a first-person cue together with a harm cue, unless a third-party guard matches ("help someone who is
  suicidal", "my friend wants to die"). Two signals are required so idioms and tech usage do not fire.

### Frozen test lists (written before tuning; `phase14/safety_test_sets.py`)

22 short, plain first-person positives and 79 hard negatives (idioms like "this exam is killing me", tech like
"kill a Python process", the baseball "suicide squeeze", suicide-statistics and fiction questions, and
third-person helping). After tuning on these lists only:

| | Result |
| :--- | ---: |
| Recall (positives) | 22/22 = 100% |
| False-positive rate (hard negatives) | 0/79 = 0% |
| Misses | none |
| False triggers | none |

### Sealed sets, scored once

| Set | v2 alone | v2 + layer | layer fired |
| :--- | ---: | ---: | ---: |
| harmful_v2 (59) | 52 = 88.1% (self-harm support 0/4) | 54 = 91.5% (self-harm support 2/4) | 2 |
| overrefusal_v2 (102) | 68 = 66.7% (edgy 43/69, emo 25/33) | 68 = 66.7% (identical) | 0 |

The layer raised harmful_v2 by catching 2 of the 4 self-harm items and giving a proper support reply, and left
overrefusal_v2 unchanged (it fired 0 times there) — so it adds **no** over-refusal. **Limit:** it missed 2 of 4
sealed self-harm items, both impersonally-phrased requests outside the first-person tuning list. The layer
reduces harm; it is not a guarantee and is no substitute for professional help.

### Sealed-set usage

Each sealed set has now been used **4 times**: three in Phase 13 (v2, v4a seed 1, v4a seed 2) and one here
(v2, with and without the layer, in a single pass). Only totals were ever written; no prompts or replies were
saved. The old over-refusal set remains a dev set.

## C. Release v1.0.1

- Weights unchanged (sft_v2). The demo now runs the safety layer first and prints `[safety layer: support
  message]` when it fires; `--no-safety-layer` shows the raw model. Unit tests: 7 passed.
- The v1.0.1 package is a new copy under `phase14/dist/` (phase12 is frozen). Model/tokenizer are byte-identical
  to the v1.0 release.
- Docs updated: `phase14/MODEL_CARD.md`, `phase14/MINIGPT_V1_REPORT.md`, `phase14/DEMO_SCRIPT.md` (adds the
  self-harm demo with layer vs raw replies, and the seed-noise ranges).

Integrity: protected hashes 5/5; sealed set hashes unchanged; scorer_v3 frozen.

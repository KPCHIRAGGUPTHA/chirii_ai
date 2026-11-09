# Phase 17E — replay against forgetting: report

**Research only: not a release candidate. No sealed set was used.** Runs: the college GPU PC, 9 November 2026,
13:10–16:21 IST (well before the 10 November 12:00 hard stop). Criteria: `phase17/CRITERIA_17E.md`, committed at
13:05 before any training.

## In short

**The pre-registered success test fails: 2 of 4 conditions.** Mixing 25% pretraining-text replay into SFT v5 raised
the five-task benchmark average by only 1.11 points (the bar was 3) and did not bring LAMBADA back (mean
12.45; bar 20; the base model before SFT scores 27.0). It did not hurt the dev set —
the replay seeds' mean dev total is +5.33 above the 17C seeds' — and dev harmful stayed at 98.7%.

## What changed

Everything as in 17C (base = the 17D final model, sft_v2 hyperparameters, seeds 11/12/13, decoding), plus replay:
3,952 windows of 512 tokens drawn at random from the 17D pretraining data, with loss on every token —
2,019,472 loss tokens against SFT v5's 6,057,640, i.e. 25% of all loss
tokens per epoch. Each replay run: 7,311 steps (17C: 7,064), about 34 minutes.

## Benchmarks (Phase 18 harness, multiple-choice tasks)

\* = the five tasks in the success test. Percent. 17C seed 13 is the 17C chosen model (`phase18/results/sft_v5_chosen`);
17C seeds 11 and 12 were measured in this phase with the same harness.

| Task | Replay s11 | s12 | s13 | **Replay mean** | 17C s11 | s12 | s13 | **17C mean** | Replay − 17C | 17D base (before SFT) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ARC-Easy * | 39.6 | 39.4 | 39.1 | **39.39** | 36.7 | 37.3 | 37.2 | **37.07** | +2.33 | 44.0 |
| HellaSwag * | 28.5 | 28.7 | 28.4 | **28.51** | 27.5 | 28.2 | 27.9 | **27.86** | +0.65 | 31.2 |
| PIQA * | 57.4 | 58.9 | 56.6 | **57.62** | 56.5 | 57.3 | 56.6 | **56.78** | +0.83 | 61.4 |
| SciQ * | 61.9 | 59.8 | 61.8 | **61.17** | 58.8 | 61.4 | 59.3 | **59.83** | +1.33 | 67.8 |
| LAMBADA * | 11.5 | 12.8 | 13.0 | **12.45** | 12.1 | 11.3 | 12.8 | **12.07** | +0.38 | 27.0 |
| ARC-Challenge | 23.9 | 24.4 | 24.9 | **24.40** | 23.8 | 22.8 | 23.8 | **23.46** | +0.94 | 23.8 |
| WinoGrande | 50.4 | 50.2 | 51.0 | **50.54** | 49.6 | 51.9 | 47.5 | **49.67** | +0.87 | 52.5 |
| OpenBookQA | 26.0 | 26.0 | 26.6 | **26.20** | 24.0 | 24.4 | 23.6 | **24.00** | +2.20 | 29.2 |
| BoolQ | 61.0 | 59.4 | 58.7 | **59.71** | 48.6 | 60.2 | 56.6 | **55.17** | +4.55 | 59.3 |
| MMLU | 26.1 | 25.3 | 25.3 | **25.60** | 25.5 | 25.9 | 25.6 | **25.66** | -0.06 | 23.0 |

Five-task average: replay **39.83**, 17C **38.72** (difference +1.11); the 17D base before SFT: 46.28.

## Dev (goal dev v2.1 + old over-refusal; scored as in 17C)

| | Seed 11 | Seed 12 | Seed 13 | Mean |
| :--- | ---: | ---: | ---: | ---: |
| Replay: dev total /176 | 124 | 122 | 122 | 122.67 |
| Replay: arithmetic /32 | 31 | 31 | 31 | 31.00 |
| Replay: harmful /26 | 26 | 25 | 26 | 25.67 |
| Replay: unknown /26 | 22 | 21 | 22 | 21.67 |
| Replay: old over-refusal /60 | 43 | 45 | 46 | 44.67 |
| 17C: dev total /176 | 117 | 116 | 119 | 117.33 |
| 17C: arithmetic /32 | 32 | 31 | 31 | 31.33 |
| 17C: harmful /26 | 25 | 26 | 26 | 25.67 |
| 17C: unknown /26 | 21 | 21 | 21 | 21.00 |
| 17C: old over-refusal /60 | 43 | 37 | 40 | 40.00 |

## Pre-registered verdict

| Condition | Result | Verdict |
| :--- | :--- | :--- |
| 1. Five-task mean ≥ 3 points above 17C | 39.83 vs 38.72 (+1.11) | **FAIL** |
| 2. Mean LAMBADA ≥ 20 | 12.45 | **FAIL** |
| 3. Mean dev total no more than 5 below 17C (117.33) | 122.67 (+5.33) | **PASS** |
| 4. Mean dev harmful ≥ 90% | 98.7% | **PASS** |

**Success: NO.**

## An honest reading

1. **Replay helped a little, everywhere, but not enough.** The replay mean is higher than the 17C mean on 9 of the
   10 tasks (all but MMLU, 25.60 vs 25.66), but by about 1–2 points on most. On the five tasks the base model before
   SFT (46.28) is still 6.45 points ahead of the replay seeds; replay recovered about 1.1 of the 7.56 points SFT lost.
2. **LAMBADA did not recover at all** (12.07 → 12.45, base 27.0). Replaying pretraining text
   with full loss did not restore last-word prediction, so this drop may not be ordinary forgetting of the
   pretraining distribution; this phase does not test what else causes it.
3. **The largest single change is BoolQ** (55.17 → 59.71), but 17C seed 11 alone (48.6) pulls the 17C mean
   down, so that gap is mostly one seed.
4. **Dev was not harmed and looks slightly better** (mean total 122.67 vs 117.33; old over-refusal mean
   44.67 vs 40.00 of 60). With only three seeds and the seed-to-seed spread seen in Phase 14 and 17C
   (6 items on over-refusal here), these dev differences should not be read as a real improvement.
5. **Replay share was fixed at 25%.** Whether a larger share would recover more is untested.

## Files

- `phase17/CRITERIA_17E.md`; `phase17/build_replay_17e.py` → `phase17/manifests/replay_17e_manifest.json` (replay file
  SHA-256 `3afacc99…`, private); `phase11/sft_train.py` (opt-in replay, test in `phase11/tests/test_sft_and_chat.py`)
- `phase17/configs/sft_v5r_s{11,12,13}.json`; `phase17/run_17e.sh`
- `phase17/results/phase17c/dev_replay_s{11,12,13}.json`; `phase18/results/sft_v5r_s{11,12,13}/`, `phase18/results/sft_v5_s{11,12}/`
- Weights: private release `weights-2026-11-09` (`sft_v5r_s{11,12,13}_final_model.pt`, verified by download)

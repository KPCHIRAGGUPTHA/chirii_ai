# Chirii AI pass/fail rules

Every rule is committed before the run it judges. A rule changed after seeing a result must say so here, with the
reason, and the report must show both the old and the new verdict.

## Already fixed (21 November 2026)

| Rule | Source |
| :--- | :--- |
| A fine-tune is attempted only if plain Qwen2.5-1.5B-Instruct with memory misses the S1 pass bar | Owner decision, 21 Nov |
| A fine-tune must not lose more than **2 points** against plain Qwen on GSM8K | Owner decision, 21 Nov |
| A fine-tune must not lose more than **2 points** against plain Qwen on function calling | Owner decision, 21 Nov |
| Deployment uses Q8_0 GGUF; a 4-bit file is used only if it scores within noise of Q8_0 on the S1 eval | Owner decision, 21 Nov |
| No training data with a non-commercial licence (alpaca-cleaned, SciQ, PKU-SafeRLHF, or any other) | Owner decision, 21 Nov |

## PROPOSED, awaiting the owner's approval (S1)

Nothing below is in force until the owner approves it and it is committed again with the approval date.

### The S1 eval

- **Tasks:** `chirii/eval/s1_tasks.jsonl`, 50 scored tasks (10 per person, across 6 task types) and 5 unscored
  Hinglish tasks. Review copy: `chirii/eval/S1_TASKS.md`.
- **Marking:** automatic, by `chirii/eval/scoring.py`. Every task also fails if the reply contains a date or rupee
  amount that is not in the person's memory ("no invention").
- **Hand check:** every verdict in the first official run is also checked by hand. The automatic score is the
  official one; every disagreement is listed in the report, and the scorer is not changed during that run.
- **Settings:** Q8_0 GGUF through llama.cpp, temperature 0, at most 256 new tokens, one run.

### Pass bars for plain Qwen2.5-1.5B-Instruct with memory

| Bar | Threshold | Why |
| :--- | :--- | :--- |
| **A. Overall** | **at least 40 of 50 (80%)** | The core promise: answers from your own memory are usually right |
| **B. "Not in memory" tasks** | **at least 7 of 8** | Saying "I don't know" instead of guessing is the main difference from a chatbot |
| **C. No invention** | **invented dates or amounts in at most 2 of 50 tasks** | An invented bill date or amount is the most harmful mistake this product can make |

**If any bar is missed, S7 (fine-tuning) is triggered.**

### Noise: what these numbers can and cannot show

- **One standard error at 80% on 50 tasks is about 5.7 points.** A score of 40/50 means the true rate is probably
  between 67% and 89% (95% interval). Two systems need to differ by more than about 11 points (6 tasks) before the
  difference means anything, unless they are compared task by task.
- **The bar is a real test, not a formality.** A system that is truly right 80% of the time passes bar A only 58% of
  the time; one that is right 85% of the time passes 88% of the time; one at 70% passes only 8% of the time.
- **Bar B is coarse.** With 8 tasks, 7/8 is compatible with a true rate anywhere from 53% to 98%. It catches a system
  that guesses often, not a subtle one.

### Comparisons

- **With memory against without memory, and Q8_0 against 4-bit:** compared task by task on the same 50 tasks with an
  exact McNemar test (two-sided, p < 0.05).
- **"Within noise" for 4-bit:** 4-bit may replace Q8_0 only if it loses at most 2 tasks net against Q8_0, the
  McNemar test is not significant, and it still passes bars B and C.

### The 2-point fine-tuning rules (used only if S7 is triggered)

- **GSM8K:** all 1,319 test questions, lm-evaluation-harness 0.4.13, 0-shot, chat template, greedy, 256 new tokens,
  "flexible-extract" score. These are exactly the settings of the December GPU job, which gives plain Qwen's
  baseline. One standard error is about 1.3 points, so the report will also give a paired comparison.
- **Function calling:** BFCL non-live (simple, multiple, parallel, parallel-multiple), with plain Qwen and the
  fine-tune run through the same harness version on the same day. Phase 28 recorded plain Qwen at 82.15% overall,
  but that harness is not in this repository, so the baseline will be re-measured.

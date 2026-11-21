# Chirii AI pass/fail rules

Every rule is committed before the run it judges. A rule changed after seeing a result must say so here, with the
reason, and the report must show both the old and the new verdict.

## Already fixed (21 November 2026)

| Rule | Source |
| :--- | :--- |
| A fine-tune is attempted only under the S7 decision rule below (plain Qwen2.5-1.5B-Instruct with memory falls short of the S1 bars) | Owner decision, 21 Nov |
| A fine-tune must not lose more than **2 points** against plain Qwen on GSM8K | Owner decision, 21 Nov |
| A fine-tune must not lose more than **2 points** against plain Qwen on function calling | Owner decision, 21 Nov |
| Deployment uses Q8_0 GGUF; a 4-bit file is used only if it scores within noise of Q8_0 on the S1 eval | Owner decision, 21 Nov |
| No training data with a non-commercial licence (alpaca-cleaned, SciQ, PKU-SafeRLHF, or any other) | Owner decision, 21 Nov |

## S1 eval: APPROVED 21 November 2026, with the owner's changes

Approved by the owner on 21 November 2026, before any system was run on the tasks. Changes from the proposal: the
scorer now accepts correct worked-out values, the S7 decision rule below, and a held-out test set (both added).

### The S1 eval (development set)

- **Tasks:** `chirii/eval/s1_tasks.jsonl`, 50 scored tasks (10 per person, across 6 task types) and 5 unscored
  Hinglish tasks. Review copy: `chirii/eval/S1_TASKS.md`. This is the **development set**: it may be run as often as
  needed while building, so its score can drift upward from tuning to it. The main result comes from the held-out
  set below.
- **Marking:** automatic, by `chirii/eval/scoring.py`. Every task other than "not in memory" lists facts that must be
  in the answer, and a test checks that an empty or evasive reply fails every task. Every task also fails if the
  reply contains an invented date, rupee amount, day count or weekday.
- **Worked-out values (owner's change):** allowed when correct, failed when wrong:
  - today's date and today's weekday;
  - a sum of two or more amounts in memory ("₹26,150 in total");
  - a day count that matches the dates in memory ("in 5 days", "2 days late");
  - a weekday written next to a date, if it is that date's weekday.

  Tests cover each, including a wrong sum, a wrong day count and a wrong weekday.
- **Hand check:** every verdict in the first official run is also checked by hand. The automatic score is the
  official one; every disagreement is listed in the report, and the scorer is not changed during that run.
- **Settings:** Q8_0 GGUF through llama.cpp, temperature 0, at most 256 new tokens, one run.

### Pass bars for plain Qwen2.5-1.5B-Instruct with memory

| Bar | Threshold | Why |
| :--- | :--- | :--- |
| **A. Overall** | **at least 40 of 50 (80%)** | The core promise: answers from your own memory are usually right |
| **B. "Not in memory" tasks** | **at least 7 of 8** | Saying "I don't know" instead of guessing is the main difference from a chatbot |
| **C. No invention** | **invented dates or amounts in at most 2 of 50 tasks** | An invented bill date or amount is the most harmful mistake this product can make |

### Decision rule for S7 (owner's change)

The first official run is plain Qwen2.5-1.5B-Instruct with memory, in S3.

| Result | What happens |
| :--- | :--- |
| Bars A, B and C all pass | No fine-tuning. S7 is not triggered |
| Bar A **below 35/50** | **S7 is triggered directly** |
| Bar A **35–39/50**, or bar **B** or **C** missed (owner's change, 21 Nov: B and C get the same rule as A) | **Failure analysis, one fix round, one rerun** (below) |

**The failure analysis (bar A at 35–39, or B or C missed):**
1. Label every failed task with exactly one cause:
   - **retrieval:** the right memory wasn't given to the model;
   - **prompt/formatting:** the memory was there, but the prompt or answer format led it wrong;
   - **scorer:** the answer was right and the marker was wrong;
   - **model behaviour:** the memory and prompt were right and the model still answered wrongly.
2. Commit the labels before any fix.
3. Fix retrieval and prompt/formatting causes in code. Fix scorer errors only through the rule at the top of this
   file: the change is committed before the rerun, and the report shows the old and new verdicts.
4. Rerun the 50 tasks **once**. A single fix round and a single rerun cover all three bars together.
5. After the rerun, count only the failures still labelled **model behaviour** (confirmed by the owner, 21 Nov). S7 is triggered
   if any one of these holds; otherwise it is not, and the remaining failures become code work in S4–S6:

| Bar | S7 is triggered if model-behaviour failures alone… | That is |
| :--- | :--- | :--- |
| A | would miss 40/50 | more than **10** of 50 tasks |
| B | would miss 7/8 on "not in memory" tasks | more than **1** of 8 |
| C | would exceed 2 tasks with invented values | more than **2** of 50 |

### Held-out test set (owner's change)

- In S6, before the final report, 30 new tasks are written:
  - same 5 people and same 6 task types (one task per person per type);
  - new situations, not paraphrases of the development tasks.
- Their marking rules are committed with them **before any system is run on them**.
- They are run **once**, on the final system, with the same settings and scorer.
- The final report gives both scores, and the **held-out score is labelled the main result**. Pass bars for the
  held-out set are proposed with the tasks and approved before its run.

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

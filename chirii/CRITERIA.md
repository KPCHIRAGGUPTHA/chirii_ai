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

The first official run is plain Qwen2.5-1.5B-Instruct with memory, in S3. **The decision is made on run (b), the
system that will ship** (owner's correction, 21 Nov, before any run; see "S3 method" below). For run (b), bar C is
replaced by bar C'.

| Result | What happens |
| :--- | :--- |
| Bars A, B and C' all pass | No fine-tuning. S7 is not triggered |
| Bar A **below 35/50** | **S7 is triggered directly** |
| Bar A **35–39/50**, or bar **B** or **C'** missed (owner's change, 21 Nov: B, C and C' get the same rule as A) | **Failure analysis, one fix round, one rerun** (below) |

**The failure analysis (bar A at 35–39, or B or C' missed):**
1. Label every failed task, and for bar C' every code-built fallback, with exactly one cause:
   - **retrieval:** the right memory wasn't given to the model;
   - **wrong item retrieved:** a different memory's fact was given instead of the right one (for example the water
     bill's date for an electricity bill question). The guard cannot catch this, because the wrong fact is in memory.
     Counts as fixable in code, like retrieval;
   - **wrong item used:** the right memory was given, but the answer took a fact from a different memory. Counts as
     model behaviour;
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
| C' (run (b)) | would exceed 5 code-built fallbacks | more than **5** of 50 tasks |

### S3 method (owner's decision, 21 Nov, recorded before any S3 run)

The 50 development tasks are run in two configurations, with the same settings, and reported side by side:

| Run | System | Role |
| :--- | :--- | :--- |
| **(a) Before** | The slice as first built: full retrieval (5 closest plus everything due in 14 days), **no** fact-check guard | Reported as the "before" picture; bars A, B and C for comparison only |
| **(b) Ship** | Trimmed retrieval (only memories relevant to the question) **with** the fact-check guard (retry with the top 3 lines, then a code-built answer) | **The S7 decision is made on this run** (owner's correction, 21 Nov) |

**Bars for run (b)** (owner's correction, 21 Nov, before any run):
- **Bar A (at least 40/50) and bar B (at least 7/8)** apply as written. The guard cannot catch a wrong fact that is
  in memory, so those failures still count against bar A and are labelled "wrong item retrieved" or "wrong item
  used".
- **Bar C is reported but not counted as evidence.** The guard enforces it by design: it uses the marker's own rules
  for invented values, plus one stricter rule about today's date.
- **Bar C' replaces it: a code-built fallback in at most 5 of 50 tasks.** The report gives the share of answers that
  were clean the first time, needed a retry, or fell back to a code-built answer. If C' is missed, the failure-analysis
  rule above applies before S7 is triggered.
- **Run (a) keeps the original bar C**, for comparison only.
- **One line per task** in the S3 results gives its run (b) guard outcome (clean, retried or fallback), its task type
  and its verdict, so it can be seen which task types trigger the guard.
- **The plain-Qwen comparison** (no memory at all) from the plan is also run, as a third column, so the value of
  memory can be seen. Qwen gets a plain assistant instruction with today's date, and the question, but no memory
  lines and no guard. Its bars are for comparison only.
- **Failure analysis by type** for both runs, using the labels above (retrieval, wrong item retrieved, wrong item used, prompt/formatting, scorer,
  model behaviour). For run (b), also how often the guard passed, retried and fell back to a code-built answer, and
  whether each retry or fallback passed or failed the task.
- **Memories in the eval are marked confirmed** before asking, as if the user had pressed Yes on each card. The eval
  measures answering; the cards are measured separately (below).
- **No prompt tuning against individual eval tasks.** Prompts, retrieval settings and guard rules are fixed before
  the run and not changed in response to single tasks. Any change after the run follows the S7 fix-round rule.
- **Extraction on new sentences:** the next fresh set of 20 sentences is written and committed in S3, before it is
  run. Target: **wrong and not flagged for the user to check ≤ 2 of 20.**

### S3 outcome: S7 triggered, then deferred (owner's decision, 22 Nov, recorded before any S4 work)

- **S7 is triggered under the pre-registered rule.** In run (b), bar A was 28/50, below 35
  (`chirii/results/s3/S3_RESULTS.md`). This result stands, and is reported as it is.
- **S7 is deferred by the owner, for this reason, in the owner's words:** "The model-side failures are concentrated
  in tasks that require filtering, counting or combining memories (date windows 1/8, plans 2/8, wrong-item 3 of 8).
  These are being moved to code by design, as the planner already was. S7 is reconsidered after S4 on the held-out
  set."
- **This is a change of plan made after seeing a result**, so it is recorded here as one. The original rule would
  have started fine-tuning now; the owner chose to move the failing kinds of reasoning into code (S4) first.
- **The rule for reconsidering S7 on the held-out set** is below. It was set by the owner on 22 Nov, before any S4
  work and before any S4 result was seen.

### S7 reconsideration rule (owner's decision, 22 Nov, fixed before any S4 work)

Applied once, to the 30 held-out tasks (5 per task type, so 5 "not in memory" tasks), run once on the final S6
system, with the same settings and marker as the development runs.

| Bar | Threshold |
| :--- | :--- |
| **A** overall | at least **24/30** (80%) |
| **B** "not in memory" | at least **4/5** |
| **C'** code-built fallback | at most **3/30** |

| Result | What happens |
| :--- | :--- |
| All three bars pass | **S7 is closed. No fine-tune.** |
| Bar A **21–23**, or C' misses by one (**4/30**) | **Failure analysis by type** (the labels above). S7 is triggered if model-behaviour failures, counting "wrong item used", are **more than 6 of 30**, or **more than 1 of 5** on the B tasks |
| Bar A **below 21/30**, or B at **3/5 or lower** | **S7 is triggered directly** |

*Reading of the owner's text:* it says both "B misses by one → failure analysis" and "B at 3/5 or lower → S7
directly". For B, missing by one *is* 3/5, so the two conflict. The stricter one is applied: **B at 3/5 triggers S7
directly**. **Open point for the owner:** the rule does not say what happens if C' misses by more than one (5/30 or
more). It must be decided before the held-out run.

- **No further deferral.** If S7 is triggered by this rule, it goes ahead, with its own plan approved first.
- **Noise:** with 30 tasks, a system that is truly 80% right scores between 19 and 28 of 30 (63–93%) in 95% of runs.
  It passes bar A (24 or more) only 61% of the time; a truly 85% system passes 85% of the time, a 90% system 97%, and
  a 70% system 16%. With 5 tasks, bar B is passed by a truly 80% system 74% of the time. **Results near a bar are
  reported with this caveat.**

**Rules for the S4 work:**
- **The marker's two errors found in S3** (a month followed by a numbered list item read as a date; the empty days of
  an asked week counted as invented) are fixed with tests. Under the rule at the top of this file, the S3 runs are
  then re-marked with the fixed marker and shown **next to** the original scores. The original scores are never
  replaced.
- **Run (c)**, the 50 development tasks on the S4 system, is reported next to runs (a) and (b), with clean, retried
  or fallback for each task. **It is a development number.** The S4 work was shaped by these 50 tasks' failures, so
  run (c) will look better than new tasks would.
- **The 30 held-out tasks are not written, seen or run before S6.**
- **The GPU diagnostic** (run (b)'s setup with Qwen2.5-7B-Instruct Q8_0 on the GPU PC) has **no decision attached**. It
  shows how much of the gap a larger model closes. It does not change any bar or verdict.

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

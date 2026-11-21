# S3: first official run of the 50-task development eval

Run on 21–22 November 2026 on the laptop CPU: Qwen2.5-1.5B-Instruct Q8_0 (llama.cpp b12359), temperature 0, at most 256
new tokens, one run, all-MiniLM-L6-v2 for search. Method and bars: `chirii/CRITERIA.md` ("S3 method"), committed
before the run, as was the runner (`chirii/eval/run_s3.py`). Every reply was read by hand; the marker's score is
the official one and every disagreement is listed. Raw replies: `replies_a.jsonl`, `replies_b.jsonl`,
`replies_none.jsonl`. Hand check and labels: `labels.json` (committed before any fix).

## Results

| | (a) Before: full retrieval, no guard | **(b) Ship: trimmed retrieval + guard** | Plain Qwen, no memory |
| :--- | ---: | ---: | ---: |
| **Bar A** overall, at least 40/50 (hand check) | 28/50 (27) | **28/50 (28)** — **miss** | 10/50 (9) |
| **Bar B** "not in memory", at least 7/8 | 5/8 | **5/8** — **miss** | 8/8 |
| Bar C invented values, at most 2/50 | 2/50 | 0/50 (reported, not evidence: the guard enforces it) | 1/50 |
| **Bar C'** code-built fallback, at most 5/50 | – | **3/50** — **pass** | – |
| Guard: clean first time / retried / fallback | – | 47 / 0 / 3 (94% / 0% / 6%) | – |
| Median time per answer | 8.7 s | 3.7 s | 4.8 s |

**By task type** (marker):

| Type | (a) | (b) | No memory |
| :--- | ---: | ---: | ---: |
| lookup | 8/10 | 8/10 | 0/10 |
| semantic | 8/8 | 8/8 | 1/8 |
| update | 6/8 | 4/8 | 1/8 |
| unknown | 5/8 | 5/8 | 8/8 |
| date_window | 0/8 | 1/8 | 0/8 |
| plan | 1/8 | 2/8 | 0/8 |

Looking up one memory works well, both by exact fact and by meaning. Answers that need several memories combined
and filtered (date windows, plans) mostly fail.

## Decision (run (b), as agreed)

- **Bar A is 28/50, below 35, so S7 (fine-tuning) is triggered directly** under the agreed rule. The hand check
  gives the same total (28/50).
- Bar B is also missed (5/8). Bar C' passes (3 fallbacks).
- **The fix round would not change this.** Of run (b)'s 22 failures, 16 are on the model's side
  ("model behaviour" or "wrong item used"), and the hand check adds 2 more (answers that say "I don't have that" and
  then give it). That is 18 model-side failures, more than the 10 allowed, so even with every retrieval,
  formatting and marker problem fixed, bar A would stay below 40. Even if the 5 plan order and clash failures move to
  code in S4, 13 remain.
- **Memory is what makes Chirii work at all:** plain Qwen with no memory passes 9/50 on the hand check, and those
  are mostly answers that only say it has no access to the user's data. That passes the "not in memory" tasks
  without being useful.

## Failure analysis by type

| Cause | (b) Ship | (a) Before | Fixable in code? |
| :--- | ---: | ---: | :--- |
| retrieval | 2 | 0 | yes |
| wrong item retrieved | 0 | 0 | yes |
| wrong item used | 6 | 7 | no (model) |
| prompt/formatting | 3 | 2 | yes |
| scorer | 1 | 1 | yes (by the scorer rule) |
| model behaviour | 10 | 12 | no (model); plan order and clashes move to code in S4 |
| **total failures (marker)** | **22** | **22** | |

Model-side failures by task type in run (b):

date_window 5, plan 5, update 3, unknown 3.

**Code problems found (to fix before the rerun or in S4; none fixed yet):**
1. Fact lines show only an item's first date. "FAT exams on 7, 9 and 11 December" appears as 7 December only (S1-41, S1-43).
2. A time window can fill up with daily medicines before reaching real appointments (S1-33).
3. "Plan my day" looks only at today, not at what needs preparing for tomorrow (S1-46).
4. The code-built fallback quotes the original sentence, including an old date that was since changed (S1-37), and it does not check for clashes (S1-47).
5. Marker and guard: a month followed by a numbered list ("15 December.⏎2.") reads as a date, and the empty days of the asked week count as invented. Together these caused 2 of the 3 fallbacks (S1-27, S1-47). Changing the marker follows the rule at the top of CRITERIA.md.
6. Plan order and clashes (5 failures) are left to Qwen today. S4 is planned to build plans in code.

## Hand check: disagreements with the marker

| Run | Task | Marker | Hand | Why |
| :--- | :--- | :--- | :--- | :--- |
| b | S1-05 | pass | fail | Opens with 'I don't have that in your memory', then gives the note. Telling the user it isn't saved when it is counts as wrong. |
| b | S1-25 | pass | fail | Same: says 'I don't have that' and then gives the milk price. |
| b | S1-12 | fail | pass | Asked only when the LIC premium is due; the date is right. The task rule also requires the amount, which the question did not ask for (task rule too strict). |
| b | S1-37 | fail | pass | Code-built answer quotes the original sentence (18 December) but states 'Now: on Monday 21 December 2026'. Correct and clear, though the old date appears in the quote. |
| a | S1-06 | pass | fail | Team call put at 10 am (it is 11 am), lists the night shift that starts on 14 December, and invents an 'Amma's birthday celebration'. The marker only checked that 11:00 appears and the order. |
| a | S1-25 | pass | fail | Says 'I don't have that' and then gives the milk price. |
| a | S1-12 | fail | pass | As in run (b): the date asked for is right; the rule also wants the amount. |
| none | S1-18 | pass | fail | Plain Qwen says it has no access; it passes only because the word 'paid' appears in 'check if your water bill is paid'. |

Marker errors seen: A month name followed by a numbered list item ('15 December.\n2. Pay') is read as a date (2 December). In run (b) this made the guard reject a correct first answer on S1-27, which then fell back to a code-built answer. In run (a) it adds a false invented date (5/1) to S1-23, which fails for other reasons anyway. Listing the empty days of the asked week ('Tuesday 15 December: No appointments') counts as invented dates. This sent S1-47 in run (b) to a code-built answer.

## One line per task

Guard outcome is for run (b). "Lines" is how many memory lines the model was given in run (b).

| Task | Type | (a) | **(b)** | Guard (b) | Lines (b) | No memory | Cause of (b) failure |
| :--- | :--- | :--- | :--- | :--- | ---: | :--- | :--- |
| S1-01 | lookup | pass | **pass** | clean | 1 | fail | - |
| S1-02 | lookup | pass | **pass** | clean | 1 | fail | - |
| S1-03 | date_window | fail | **fail** | clean | 7 | fail | wrong item used: Lists the car insurance (28 Dec, outside the window) and the EMI; leaves out the electricity bill's amount. |
| S1-04 | date_window | fail | **fail** | clean | 5 | fail | wrong item used: Adds the gym fee (1 Jan) and the electricity bill (12 Dec), both outside 24-31 Dec. Both were given as search matches beside the window items. |
| S1-05 | semantic | pass | **pass** | clean | 1 | fail | - |
| S1-06 | plan | pass | **pass** | clean | 4 | fail | - |
| S1-07 | plan | fail | **fail** | clean | 4 | fail | model behaviour: Lists the reception and the night shift but does not say they clash. |
| S1-08 | update | fail | **fail** | clean | 7 | fail | model behaviour: Mentions only the paid electricity bill; misses the car insurance due this month, which was in the lines. |
| S1-09 | update | pass | **pass** | clean | 1 | fail | - |
| S1-10 | unknown | fail | **pass** | clean | 1 | pass | - |
| S1-11 | lookup | pass | **pass** | clean | 2 | fail | - |
| S1-12 | lookup | fail | **fail** | clean | 1 | fail | scorer: Task rule asks for an amount the question did not ask for. |
| S1-13 | date_window | fail | **fail** | clean | 7 | fail | wrong item used: Lists the eye check-up and annual day as payments; misses the ₹2,150 electricity bill, which was in the lines. |
| S1-14 | date_window | fail | **pass** | clean | 3 | fail | - |
| S1-15 | semantic | pass | **pass** | clean | 1 | fail | - |
| S1-16 | semantic | pass | **pass** | clean | 1 | fail | - |
| S1-17 | plan | fail | **fail** | clean | 3 | fail | model behaviour: Tuition 5-6:30 pm and housewarming 5:30 pm listed without saying they clash. |
| S1-18 | update | pass | **pass** | clean | 1 | pass | - |
| S1-19 | update | pass | **pass** | clean | 2 | fail | - |
| S1-20 | unknown | pass | **pass** | clean | 1 | pass | - |
| S1-21 | lookup | pass | **pass** | clean | 1 | fail | - |
| S1-22 | lookup | pass | **pass** | clean | 1 | fail | - |
| S1-23 | date_window | fail | **fail** | clean | 6 | fail | wrong item used: Treats Mrs. Deshpande's ₹3,450 (owed to the user) as a payment; misses the ₹6,380 electricity bill. |
| S1-24 | date_window | fail | **fail** | clean | 2 | fail | model behaviour: Says nothing is due tomorrow; Mrs. Deshpande's payment on 13 Dec was in the lines. |
| S1-25 | semantic | pass | **pass** | clean | 1 | fail | - |
| S1-26 | plan | fail | **fail** | clean | 3 | fail | model behaviour: Plan out of order: the 9-11 am sugar delivery comes last. |
| S1-27 | plan | fail | **pass** | fallback | 3 | fail | - |
| S1-28 | update | pass | **pass** | clean | 1 | fail | - |
| S1-29 | update | pass | **fail** | clean | 1 | fail | model behaviour: Gives the old date (11 Dec). The line said 'Current details: 12 December' and 'Use the current details'. Run (a) got this right from the same line. |
| S1-30 | unknown | pass | **pass** | clean | 5 | pass | - |
| S1-31 | lookup | pass | **pass** | clean | 2 | fail | - |
| S1-32 | lookup | pass | **pass** | clean | 3 | fail | - |
| S1-33 | date_window | fail | **fail** | clean | 5 | fail | retrieval: Daily medicines filled the 10-entry window limit, so the 18 Dec check-up was cut off. |
| S1-34 | semantic | pass | **pass** | clean | 3 | fail | - |
| S1-35 | semantic | pass | **pass** | clean | 1 | fail | - |
| S1-36 | plan | fail | **fail** | clean | 3 | fail | model behaviour: Plan out of order: 11 am check-up before the 8 am fasting test. |
| S1-37 | update | fail | **fail** | fallback | 2 | fail | prompt/formatting: Qwen gave today's date twice (caught by the guard); the code-built answer then quoted the original sentence with the old date. |
| S1-38 | unknown | pass | **pass** | clean | 3 | pass | - |
| S1-39 | unknown | pass | **fail** | clean | 1 | pass | wrong item used: Only the water bill was given (there is no electricity bill); Qwen gave the water bill's date as the electricity bill's. |
| S1-40 | unknown | fail | **fail** | clean | 4 | pass | model behaviour: Claims the morning tablet was taken; nothing in memory says so. |
| S1-41 | lookup | fail | **fail** | clean | 1 | fail | prompt/formatting: The fact line shows only the item's first date (7 Dec); the other exam dates (9 and 11 Dec) are left out of 'Current details'. |
| S1-42 | lookup | pass | **pass** | clean | 1 | fail | - |
| S1-43 | date_window | fail | **fail** | clean | 3 | fail | prompt/formatting: Same missing exam dates; Qwen reports the FAT exam as 7 Dec. |
| S1-44 | semantic | pass | **pass** | clean | 2 | pass | - |
| S1-45 | semantic | pass | **pass** | clean | 1 | fail | - |
| S1-46 | plan | fail | **fail** | clean | 2 | fail | retrieval: 'Plan my day' looks only at today; the task expects tomorrow's OS exam to be mentioned for study. |
| S1-47 | plan | fail | **fail** | fallback | 3 | fail | model behaviour: Qwen's first answer listed both 5 pm items without calling it a clash (and was flagged for empty-day dates); the code-built answer does not check clashes either. |
| S1-48 | update | pass | **fail** | clean | 2 | fail | model behaviour: Names the mess fee, which the lines marked PAID; misses the semester fee. |
| S1-49 | unknown | pass | **pass** | clean | 0 | pass | - |
| S1-50 | unknown | fail | **fail** | clean | 5 | pass | wrong item used: No Compiler Design exam is saved; gives the FAT exams' date instead of saying so. |

Tasks where the guard acted, by type: plan 2, update 1.

## Hinglish (5 tasks, not scored)

| Task | Question | (a) | (b) |
| :--- | :--- | :--- | :--- |
| H-1 | Mera bijli ka bill kab due hai aur kitna hai? | Mera bijli ka bill 12 December 2026 ka due hai. Amount hai ₹1,860. | Mera bijli ka bill 28 December ka due hai aur kitna hai? 9,240 rupees hai. |
| H-2 | Divya ki school fees kitni hai, last date kya hai? | Divya's school term fee of ₹24,000 must be paid by 18 December 2026. | Divya's school term fee is ₹24,000 and it must be paid by 18 December 2026. |
| H-3 | Kal kya payment karna hai? | I don't have that in your memory. | I don't have that in your memory. |
| H-4 | Aaj ka plan bata do. | I don't have that in your memory. | I don't have that in your memory. |
| H-5 | Mera Compiler Design ka exam kab hai? | The user's exam on Compiler Design is scheduled for Monday 7 December 2026 at 2:00 pm. | The user's Compiler Design exam is scheduled for Monday, 7 December 2026, at 2:00 pm. |

# Run (d): the S4b system on the 50 development tasks

**A development number.** S4 and S4b were built in response to these tasks' failures, so run (d) is expected to look better
than new tasks would (CRITERIA.md). The decision on S7 is made on the 30 held-out tasks in S6.

Same settings as S3: Qwen2.5-1.5B-Instruct Q8_0 on the laptop CPU, temperature 0, one run. The S4 code was committed
before the run (S4b code, 67e1990). Every reply was read by hand: `labels_d.json`, committed before any fix. All runs
are scored with the current (S4b) marker; original scores: S3_RESULTS.md, RUN_C.md; re-marks: REMARKED_S4B.md.

| | (a) Before | (b) Ship (S3) | (c) S4 | **(d) S4b** |
| :--- | ---: | ---: | ---: | ---: |
| **Bar A** | 28/50 | 28/50 | 44/50 | **49/50** |
| Bar A, hand check | 27/50 | 28/50 | 46/50 | **50/50** |
| Bar B "not in memory" | 5/8 | 5/8 | 8/8 | **8/8** |
| Invented values | 1 | 0 | 1 | **0** |
| Code-built fallbacks | – | 3 | 10 | **16** (all code-caused: see below) |
| Clean / retried / fallback / answered by code | – | 47 / 0 / 3 / – | 32 / 0 / 10 / 8 | 26 / 1 / 16 / 7 |
| Median time per answer | 8.7 s | 3.7 s | 4.4 s | 3.4 s |

Routes taken in run (d): lookup 22, window 10, meaning 8, plan 8, status 2.

**By task type** (current marker):

| Type | (a) | (b) | (c) | (d) |
| :--- | ---: | ---: | ---: | ---: |
| lookup | 8/10 | 8/10 | 8/10 | 9/10 |
| semantic | 8/8 | 8/8 | 8/8 | 8/8 |
| update | 6/8 | 4/8 | 5/8 | 8/8 |
| unknown | 5/8 | 5/8 | 8/8 | 8/8 |
| date_window | 0/8 | 1/8 | 7/8 | 8/8 |
| plan | 1/8 | 2/8 | 8/8 | 8/8 |

## Fallbacks by cause (owner's C' rule)

All 16 fallbacks: Qwen's lead restated dates, times or amounts from the list, which the S4b lead rule forbids (a rule stricter than the owner's 'guard checks Qwen's text'). Every restated value was correct: none would count as invented under the marker's rules. About 3 also restate the period's start date, which is today, and would trip the guard's today's-date rule. The list shown is the code's own, so every fallback answer is still complete. Cause by the owner's C' rule: code (the lead rule), not model invention.

## Failures and labels

| Task | Type | Cause | Detail |
| :--- | :--- | :--- | :--- |
| S1-12 | lookup | scorer | Task rule asks for an amount the question did not ask for. |

Hand-check disagreements: S1-12: marker fail, hand pass.
Other notes: S1-36: the plan still labels the 11 am item 'Ramaswamy at the hospital' (the clause splitter breaks at 'Dr.'; cosmetic). S1-24: the label reads 'Mrs. Deshpande owes, ₹3,450' (title wording; cosmetic). S1-43: the lead repeats the question and the closing says 'That's all for today!' for a 3-day list (wording). S1-34: still says 'the user'.

## One line per task

| Task | Type | (a) | (b) | (c) | **(d)** | Route (d) | Guard (d) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S1-01 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-02 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-03 | date_window | fail | fail | pass | **pass** | window | fallback |
| S1-04 | date_window | fail | fail | pass | **pass** | window | retried |
| S1-05 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-06 | plan | pass | pass | pass | **pass** | plan | fallback |
| S1-07 | plan | fail | fail | pass | **pass** | plan | fallback |
| S1-08 | update | fail | fail | pass | **pass** | window | fallback |
| S1-09 | update | pass | pass | pass | **pass** | lookup | clean |
| S1-10 | unknown | fail | pass | pass | **pass** | lookup | by code |
| S1-11 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-12 | lookup | fail | fail | fail | **fail** | lookup | clean |
| S1-13 | date_window | fail | fail | pass | **pass** | window | fallback |
| S1-14 | date_window | fail | pass | pass | **pass** | window | fallback |
| S1-15 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-16 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-17 | plan | fail | fail | pass | **pass** | plan | fallback |
| S1-18 | update | pass | pass | pass | **pass** | status | by code |
| S1-19 | update | pass | pass | pass | **pass** | lookup | clean |
| S1-20 | unknown | pass | pass | pass | **pass** | lookup | clean |
| S1-21 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-22 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-23 | date_window | fail | fail | pass | **pass** | window | fallback |
| S1-24 | date_window | fail | fail | fail | **pass** | window | fallback |
| S1-25 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-26 | plan | fail | fail | pass | **pass** | plan | fallback |
| S1-27 | plan | fail | pass | pass | **pass** | plan | fallback |
| S1-28 | update | pass | pass | pass | **pass** | lookup | clean |
| S1-29 | update | pass | fail | fail | **pass** | lookup | clean |
| S1-30 | unknown | pass | pass | pass | **pass** | lookup | clean |
| S1-31 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-32 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-33 | date_window | fail | fail | pass | **pass** | window | fallback |
| S1-34 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-35 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-36 | plan | fail | fail | pass | **pass** | plan | fallback |
| S1-37 | update | fail | fail | fail | **pass** | lookup | clean |
| S1-38 | unknown | pass | pass | pass | **pass** | lookup | by code |
| S1-39 | unknown | pass | fail | pass | **pass** | lookup | by code |
| S1-40 | unknown | fail | fail | pass | **pass** | status | by code |
| S1-41 | lookup | fail | fail | fail | **pass** | lookup | clean |
| S1-42 | lookup | pass | pass | pass | **pass** | lookup | clean |
| S1-43 | date_window | fail | fail | pass | **pass** | window | clean |
| S1-44 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-45 | semantic | pass | pass | pass | **pass** | meaning | clean |
| S1-46 | plan | fail | fail | pass | **pass** | plan | fallback |
| S1-47 | plan | fail | fail | pass | **pass** | plan | fallback |
| S1-48 | update | pass | fail | fail | **pass** | window | fallback |
| S1-49 | unknown | pass | pass | pass | **pass** | lookup | by code |
| S1-50 | unknown | fail | fail | pass | **pass** | lookup | by code |

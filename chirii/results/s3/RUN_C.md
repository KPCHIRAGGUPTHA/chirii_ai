# Run (c): the S4 system on the 50 development tasks

**A development number.** S4 was built in response to these tasks' failures, so run (c) is expected to look better
than new tasks would (CRITERIA.md). The decision on S7 is made on the 30 held-out tasks in S6.

Same settings as S3: Qwen2.5-1.5B-Instruct Q8_0 on the laptop CPU, temperature 0, one run. The S4 code was committed
before the run (0304889). Every reply was read by hand: `labels_c.json`, committed before any fix.

| | (a) Before | (b) Ship (S3) | **(c) S4** |
| :--- | ---: | ---: | ---: |
| Bar A, original marker | 28/50 | 28/50 | – |
| **Bar A, current marker** | 28/50 | 28/50 | **42/50** |
| Bar A, hand check | 27/50 | 28/50 | **46/50** |
| Bar B "not in memory" | 5/8 | 5/8 | **8/8** |
| Invented values (original marker for (a), (b); re-marked (a) has 1) | 2 | 0 | 3 (all from the code's own text: see below) |
| Code-built fallbacks | – | 3 | **10** |
| Clean / retried / fallback / answered by code | – | 47 / 0 / 3 / – | 32 / 0 / 10 / 8 |
| Median time per answer | 8.7 s | 3.7 s | 4.4 s |

Routes taken in run (c): lookup 22, window 10, meaning 8, plan 8, status 2.

**By task type** (current marker):

| Type | (a) | (b) | (c) |
| :--- | ---: | ---: | ---: |
| lookup | 8/10 | 8/10 | 8/10 |
| semantic | 8/8 | 8/8 | 8/8 |
| update | 6/8 | 4/8 | 5/8 |
| unknown | 5/8 | 5/8 | 8/8 |
| date_window | 0/8 | 1/8 | 5/8 |
| plan | 1/8 | 2/8 | 8/8 |

## Why 10 fallbacks (more than the 5 allowed on the held-out set)

Most are caused by the S4 code itself, not by Qwen:
- **Header dates (7 of 10).** The code's list header names the asked period ("from Monday 7 December to Monday
  14 December"). The start is today's date, which trips the guard's today's-date rule, and the end date is not a
  memory date, so it counts as invented. Qwen's rewording keeps the header, fails the check and falls back. The
  same header dates make 2 tasks fail the marker (S1-14, S1-48).
- **Weekly routines (1).** A weekly routine's date (bhajan on Thursday 17 December) is not a memory date to the
  marker, and the guard uses the marker's rule (S1-33 fails for this, together with a header date).
- **Real rewording losses (2–3).** Qwen dropped an amount (S1-04) or a date (S1-08, and S1-03 alongside header
  dates). The checks caught each, and the code's text was shown instead.

## Failures and labels

| Task | Type | Cause | Detail |
| :--- | :--- | :--- | :--- |
| S1-12 | lookup | scorer | Task rule asks for an amount the question did not ask for. |
| S1-14 | date_window | prompt/formatting | The code's window header shows the period's end date (code-built text). |
| S1-24 | date_window | retrieval | Router read 'due' as a payment word, so Mrs. Deshpande's ₹3,450 (owed to the user, due tomorrow) was filtered out. |
| S1-29 | update | model behaviour | Gives the old date (11 Dec) although the line gives 12 Dec as current, as in run (b). |
| S1-33 | date_window | scorer | Weekly routine dates are not memory dates to the marker; also the header end date (code-built text). |
| S1-37 | update | model behaviour | Gives the old date (18 Dec); the line says 21 Dec is current. The guard allows the old date because it is in the item's history. |
| S1-41 | lookup | model behaviour | The line now lists all three exam dates; Qwen picked DBMS's date (9 Dec) for Operating Systems. |
| S1-48 | update | prompt/formatting | The code's window header shows the period's end date (code-built text). |

Hand-check disagreements: S1-12: marker fail, hand pass; S1-14: marker fail, hand pass; S1-33: marker fail, hand pass; S1-48: marker fail, hand pass.
Other notes: S1-36: the plan labels the 11 am item 'Ramaswamy at the hospital'. The clause splitter breaks at 'Dr.' (cosmetic; the plan is right). S1-06: the rewording shows the courier as '5:00 pm' instead of 'by 5:00 pm' (minor; the order is right). S1-34: still says 'the user' although prompt version 2 says not to.

## One line per task

| Task | Type | (a) | (b) | **(c)** | Route (c) | Guard (c) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S1-01 | lookup | pass | pass | **pass** | lookup | clean |
| S1-02 | lookup | pass | pass | **pass** | lookup | clean |
| S1-03 | date_window | fail | fail | **pass** | window | fallback |
| S1-04 | date_window | fail | fail | **pass** | window | fallback |
| S1-05 | semantic | pass | pass | **pass** | meaning | clean |
| S1-06 | plan | pass | pass | **pass** | plan | clean |
| S1-07 | plan | fail | fail | **pass** | plan | clean |
| S1-08 | update | fail | fail | **pass** | window | fallback |
| S1-09 | update | pass | pass | **pass** | lookup | clean |
| S1-10 | unknown | fail | pass | **pass** | lookup | by code |
| S1-11 | lookup | pass | pass | **pass** | lookup | clean |
| S1-12 | lookup | fail | fail | **fail** | lookup | clean |
| S1-13 | date_window | fail | fail | **pass** | window | fallback |
| S1-14 | date_window | fail | pass | **fail** | window | fallback |
| S1-15 | semantic | pass | pass | **pass** | meaning | clean |
| S1-16 | semantic | pass | pass | **pass** | meaning | clean |
| S1-17 | plan | fail | fail | **pass** | plan | clean |
| S1-18 | update | pass | pass | **pass** | status | by code |
| S1-19 | update | pass | pass | **pass** | lookup | clean |
| S1-20 | unknown | pass | pass | **pass** | lookup | clean |
| S1-21 | lookup | pass | pass | **pass** | lookup | clean |
| S1-22 | lookup | pass | pass | **pass** | lookup | clean |
| S1-23 | date_window | fail | fail | **pass** | window | fallback |
| S1-24 | date_window | fail | fail | **fail** | window | by code |
| S1-25 | semantic | pass | pass | **pass** | meaning | clean |
| S1-26 | plan | fail | fail | **pass** | plan | clean |
| S1-27 | plan | fail | pass | **pass** | plan | clean |
| S1-28 | update | pass | pass | **pass** | lookup | clean |
| S1-29 | update | pass | fail | **fail** | lookup | clean |
| S1-30 | unknown | pass | pass | **pass** | lookup | clean |
| S1-31 | lookup | pass | pass | **pass** | lookup | clean |
| S1-32 | lookup | pass | pass | **pass** | lookup | clean |
| S1-33 | date_window | fail | fail | **fail** | window | fallback |
| S1-34 | semantic | pass | pass | **pass** | meaning | clean |
| S1-35 | semantic | pass | pass | **pass** | meaning | clean |
| S1-36 | plan | fail | fail | **pass** | plan | clean |
| S1-37 | update | fail | fail | **fail** | lookup | clean |
| S1-38 | unknown | pass | pass | **pass** | lookup | by code |
| S1-39 | unknown | pass | fail | **pass** | lookup | by code |
| S1-40 | unknown | fail | fail | **pass** | status | by code |
| S1-41 | lookup | fail | fail | **fail** | lookup | clean |
| S1-42 | lookup | pass | pass | **pass** | lookup | clean |
| S1-43 | date_window | fail | fail | **pass** | window | fallback |
| S1-44 | semantic | pass | pass | **pass** | meaning | clean |
| S1-45 | semantic | pass | pass | **pass** | meaning | clean |
| S1-46 | plan | fail | fail | **pass** | plan | clean |
| S1-47 | plan | fail | fail | **pass** | plan | fallback |
| S1-48 | update | pass | fail | **fail** | window | fallback |
| S1-49 | unknown | pass | pass | **pass** | lookup | by code |
| S1-50 | unknown | fail | fail | **pass** | lookup | by code |

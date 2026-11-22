# Run (e1): first try on 20 new development tasks

The S4b system with the lead fix (commit 3badce5) on the second development set (`chirii/eval/DEV2_TASKS.md`, committed
before this run as 5fa6bb9). **A fresh first-try number**: none of these tasks was seen while building. Run once, on the
laptop CPU, Qwen2.5-1.5B-Instruct Q8_0, temperature 0. Every reply read by hand (`labels_e1.json`).

| | Run (d): the 50 tasks S4 was built on | **Run (e1): 20 new tasks, first try** |
| :--- | ---: | ---: |
| Overall (marker) | 49/50 (98%) | **13/20 (65%)** |
| Overall (hand check) | 50/50 | **12/20 (60%)** |
| "Not in memory" | 8/8 | **2/3** |
| Invented values | 0 | 0 |
| Clean lead / lead dropped / fallback / by code | 26 / – / 16 / 7 | 16 / 0 / 0 / 4 |
| Median time | 3.4 s | 5.3 s |

| Type | (d) | (e1) |
| :--- | ---: | ---: |
| lookup | 9/10 | 3/4 |
| semantic | 8/8 | 3/3 |
| update | 8/8 | 2/3 |
| unknown | 8/8 | 2/3 |
| date_window | 8/8 | 1/4 |
| plan | 8/8 | 2/3 |

## What this shows

- **The 50-task set was saturated.** On new tasks the same system scores 65% (hand 60%), not 98%.
  With only 20 tasks the number is rough: 13/20 is compatible with a true rate of 43–82% (95% Wilson interval),
  against 90–100% for 49/50. The gap is real; its exact size is not known.
- **Most failures are in the code's reading of the question, not in Qwen:** 5 of 7 failures
  are the router or the period reader not understanding a new phrasing ("before the weekend", "this weekend", "what does
  my Saturday look like", "still pending", "before we leave for Madurai", "how much … this week"). The S4 rules were
  written for the first set's phrasings and don't cover these.
- **Looking up one memory still works** (lookups and meaning questions 6 of 7).
- **The lead fix worked:** no fallbacks and no dropped leads, against 16 of 50 in run (d). One new bug: a lead that hit
  its length limit was cut mid-word (see below).

## Failures and labels

| Task | Type | Question | Cause | Detail |
| :--- | :--- | :--- | :--- | :--- |
| D2-02 | date_window | Which bills are pending before the weekend? | code: period not understood | 'Before the weekend' is not a period the router knows. 'Pending' sent it to the 'still to pay' list (90 days), which includes the 19 and 22 January bills. |
| D2-03 | plan | What does my Saturday look like? | code: routing and specific-item check | 'What does my Saturday look like?' is not recognised as a day plan; 'Saturday' (capitalised) was taken as a name and the specific-item check answered 'I don't have anything about Saturday saved'. |
| D2-06 | update | Is Karthik's swimming fee still pending? | code: routing | 'Is … still pending?' is a status question, but 'pending' sent it to the payment list, which (correctly) left the paid fee out and said nothing was due, instead of saying it was paid. |
| D2-07 | date_window | Anything to pay before we leave for Madurai? | code: period not understood | 'Before we leave for Madurai' sets the period by a memory (the trip on 28 January). No rule matched; Qwen's routing chose 'status'. |
| D2-09 | date_window | How much do I need for payments this week? | code: routing | 'How much … this week' is excluded from the window route (questions starting 'how much' go to lookup). On the lookup path Qwen then chose Mr. Kulkarni's ₹2,700 (money owed to the user) instead of the Sai Agencies bill, which was in the lines (wrong item used). |
| D2-17 | lookup | When is my flight home and what's the flight number? | model behaviour | Asked 'when'; gave the date but not the 6:15 am time, which was in the line. |
| D2-19 | unknown | When is the Infosys interview? | wrong item used | Only the Infosys offer letter is saved. The specific-item check passed because 'Infosys' matches; it does not check 'interview'. Qwen called the joining date the interview date. |

Hand-check disagreements: D2-16 (marker pass, hand fail): Asked about the weekend; the list also gives Wednesday's physiotherapy and the lead calls it part of 'this weekend'. 'This weekend' was not understood as a period, so a default 7-day list was shown; the two required items happen to be in it.

Other findings:
- Lead cut off: in D2-02 the lead reached its 160-character limit mid-word and the JSON continued into the closing line ('The total amount to be p' … 'aid is ₹'). The guard passed it because no wrong value was stated. A lead that hits the limit should be dropped.
- Leads now state facts freely; in D2-02 and D2-16 they repeated the code's wrong list (the list was wrong, not the lead).
- No fallbacks and no dropped leads in 20 tasks after the lead fix (run (d): 16 of 50).

## One line per task

| Task | Type | Verdict | Route | Lead |
| :--- | :--- | :--- | :--- | :--- |
| D2-01 | lookup | pass | lookup | clean |
| D2-02 | date_window | fail | window | clean |
| D2-03 | plan | fail | meaning | by code |
| D2-04 | unknown | pass | lookup | clean |
| D2-05 | semantic | pass | lookup | clean |
| D2-06 | update | fail | window | by code |
| D2-07 | date_window | fail | status | by code |
| D2-08 | lookup | pass | lookup | clean |
| D2-09 | date_window | fail | lookup | clean |
| D2-10 | update | pass | lookup | clean |
| D2-11 | plan | pass | window | clean |
| D2-12 | unknown | pass | lookup | by code |
| D2-13 | semantic | pass | meaning | clean |
| D2-14 | lookup | pass | lookup | clean |
| D2-15 | update | pass | lookup | clean |
| D2-16 | date_window | pass | window | clean |
| D2-17 | lookup | fail | lookup | clean |
| D2-18 | semantic | pass | lookup | clean |
| D2-19 | unknown | fail | lookup | clean |
| D2-20 | plan | pass | plan | clean |

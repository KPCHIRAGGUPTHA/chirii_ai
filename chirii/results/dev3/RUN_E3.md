# Run (e3): S4c on development set 3, first try

S4c (one router for memory and general chat, the general period reader, Qwen-assisted routing checked by code, the
all-terms specific check, date and time in lookups, general chat with labels) on development set 3, written and
committed before S4c was built (cbd3d93). Run once (runner 23765fa) on the laptop CPU, Qwen2.5-1.5B-Instruct Q8_0,
temperature 0. **A fresh first-try number.** Every reply and route was read by hand (`labels_e3.json`). 95% intervals
are Wilson intervals.

## Memory tasks, next to runs (d) and (e1)

| | (d) 50 tuned tasks | (e1) 20 new tasks, S4b | **(e3) 30 new tasks, S4c** |
| :--- | ---: | ---: | ---: |
| Marker | 49/50 (90–100%) | 13/20 (43–82%) | **23/30 (59–88%)** |
| Hand check | 50/50 | 12/20 (39–78%) | **25/30 (66–93%)** |
| "Not in memory" (marker) | 8/8 | 2/3 | 1/5 (hand: 4/5 with the 3 partial-match answers) |
| Invented values | 0 | 0 | 1 |
| Lead clean / dropped / fallback / by code | 26 / – / 16 / 7 | 16 / 0 / 0 / 4 | 20 / 2 / 0 / 8 |
| Median time | 3.4 s | 5.3 s | 3.8 s |

| Type | (e3) marker |
| :--- | ---: |
| lookup | 5/5 |
| semantic | 3/3 |
| update | 3/5 |
| unknown | 1/5 |
| date_window | 8/8 |
| plan | 3/4 |

## General chat (first time measured)

| Kind | Marker | 95% interval |
| :--- | ---: | :--- |
| maths | 12/15 | 55–93% |
| general | 14/15 | 70–99% |
| writing | 9/10 | 60–98% |
| **all chat** | **35/40** | 74–95% |

Live-information questions (3) were all refused by code; the two questions nobody could know got "I'm not sure".

## Router accuracy (shown separately)

| | Right | 95% interval |
| :--- | ---: | :--- |
| 15 router tests: route | 14/15 | 70–99% |
| 15 router tests: period (where one was expected) | 15/15 | |
| All 85 items: route | 79/85 | 85–97% |

Route mismatches: D3-18: general instead of lookup ('GST' not seen as personal); D3-26: mixed instead of meaning (answer still right); M-02: mixed instead of maths ('I invest' made it personal; answer still right); M-08: writing instead of maths ('Write 2.5 lakh…'; answer still right); M-11: window instead of mixed (the list's lead gave the right total); R-11: people instead of plan: 'Tell me about my week' — the name pattern ignores case, so 'my' looked like a name (bug).

## Failures and labels

| Task | Cause | Detail |
| :--- | :--- | :--- |
| D3-05 | code: update parsing | 'Postponed to 5 to 7 March' was read as 7 March (the range without 'from' was missed), so the trip shows as 7–9 March. |
| D3-06 | scorer | Correct partial-match answer; marker phrase list. |
| D3-10 | scorer | Task rule stricter than needed; the plan is right. |
| D3-11 | code: status item choice | 'Do I still owe anything for Karthik's trip?' picked the trip (no amount) instead of the trip fee, which was marked paid. |
| D3-12 | scorer | Correct partial-match answer; marker phrase list. |
| D3-18 | code: router | 'GST' (a saved name in capitals) did not make the question personal, so it went to general knowledge and Qwen gave a general GST rule. |
| D3-24 | scorer | Correct partial-match answer; marker phrase list. |
| D3-28 | model behaviour (lead) | Hand fail: the lead contradicts the code's list. |
| D3-29 | code: update detection | Hand fail: 'Returned … today' is not recognised as an update, so it was saved as a separate note and the status answer has no record. |
| M-10 | model behaviour (expression) | Qwen wrote 104.50 × 35 ÷ 100. The check allowed it because 100 is an allowed constant. |
| M-12 | code: safe default | The expression was rejected by the check, so Chirii said it couldn't work it out (no wrong answer given). |
| M-13 | model behaviour (expression) | Qwen divided by 31, not 28, for February. 31 is an allowed constant. |
| G-06 | model behaviour | 'UPI stands for United Parcel Service' (wrong; W-10 explained UPI correctly). |
| W-04 | model behaviour | Generic steps without any polite way of actually saying no. |

Hand-check disagreements (memory): D3-06: marker fail, hand pass (Partial-match answer as specified ('I have … Goa trip …, but nothing about flight'). The marker's not-in-memory phrases don't include 'nothing about'.); D3-12: marker fail, hand pass (Same partial-match answer ('I have Divya's tuition …, but nothing about annual'); the missing term shows as 'annual', not 'annual day' (wording).); D3-24: marker fail, hand pass (Same partial-match answer ('… but nothing about train').); D3-10: marker fail, hand pass (Friday's plan is right (plumber at 3 pm). It adds 'Coming up tomorrow (Saturday): Murali's birthday', clearly labelled as Saturday; the task forbade 'birthday' to catch a Friday/Saturday mix-up, which did not happen.); D3-28: marker pass, hand fail (The lead says 'You do not have to pay anything before going home', directly above a list showing the ₹4,200 mess bill. The guard checks values, not meaning.); D3-29: marker pass, hand fail (Says 'I don't have a record of that', though the user said 'Returned the library book today'. It passed only because the quoted sentence contains 'Returned'.)

Other findings:
- Leads can contradict the code's list ('nothing to pay' above a bill). A check like the clash check is needed.
- Maths constants (100, 31) let wrong expressions through; constants should be allowed only where the question's words call for them.
- Titles with dates inside ('Trip to Goa with college friends from 26 to 28 February') make partial-match answers long.
- W-01: the writing check flagged 'back on 20th February' (the day after the leave) and added a 'please check' note; a correct letter with an unnecessary warning.

## One line per memory task

| Task | Type | Verdict | Route | Lead |
| :--- | :--- | :--- | :--- | :--- |
| D3-01 | lookup | pass | lookup | pass |
| D3-02 | date_window | pass | window | pass |
| D3-03 | date_window | pass | window | lead_dropped |
| D3-04 | plan | pass | plan | pass |
| D3-05 | update | fail | lookup | pass |
| D3-06 | unknown | fail | lookup | code |
| D3-07 | lookup | pass | lookup | pass |
| D3-08 | semantic | pass | meaning | pass |
| D3-09 | date_window | pass | window | pass |
| D3-10 | plan | fail | plan | pass |
| D3-11 | update | fail | status | code |
| D3-12 | unknown | fail | lookup | code |
| D3-13 | date_window | pass | window | pass |
| D3-14 | date_window | pass | window | code |
| D3-15 | update | pass | status | code |
| D3-16 | lookup | pass | lookup | pass |
| D3-17 | plan | pass | plan | pass |
| D3-18 | unknown | fail | general | pass |
| D3-19 | lookup | pass | lookup | pass |
| D3-20 | date_window | pass | window | pass |
| D3-21 | date_window | pass | window | lead_dropped |
| D3-22 | semantic | pass | meaning | pass |
| D3-23 | update | pass | lookup | pass |
| D3-24 | unknown | fail | lookup | code |
| D3-25 | lookup | pass | lookup | pass |
| D3-26 | semantic | pass | mixed | pass |
| D3-27 | plan | pass | plan | pass |
| D3-28 | date_window | pass | window | pass |
| D3-29 | update | pass | status | code |
| D3-30 | unknown | pass | lookup | code |

## One line per chat task

| Task | Verdict | Route (expected) | Label |
| :--- | :--- | :--- | :--- |
| M-01 | pass | maths (maths) | Calculated |
| M-02 | pass | mixed (maths) | Calculated from your memory |
| M-03 | pass | maths (maths) | Calculated |
| M-04 | pass | maths (maths) | Calculated |
| M-05 | pass | maths (maths) | Calculated |
| M-06 | pass | maths (maths) | Calculated |
| M-07 | pass | maths (maths) | Calculated |
| M-08 | pass | writing (maths) | Writing help |
| M-09 | pass | maths (maths) | Calculated |
| M-10 | fail | maths (maths) | Calculated |
| M-11 | pass | window (mixed) | From your memory |
| M-12 | fail | mixed (mixed) | Calculated from your memory |
| M-13 | fail | mixed (mixed) | Calculated from your memory |
| M-14 | pass | mixed (mixed) | Calculated from your memory |
| M-15 | pass | mixed (mixed) | Calculated from your memory |
| G-01 | pass | general (general) | General knowledge |
| G-02 | pass | general (general) | General knowledge |
| G-03 | pass | general (general) | General knowledge |
| G-04 | pass | general (general) | General knowledge |
| G-05 | pass | general (general) | General knowledge |
| G-06 | fail | general (general) | General knowledge |
| G-07 | pass | general (general) | General knowledge |
| G-08 | pass | general (general) | General knowledge |
| G-09 | pass | general (general) | General knowledge |
| G-10 | pass | general (general) | General knowledge |
| G-11 | pass | general (general) | General knowledge |
| G-12 | pass | general (general) | General knowledge |
| G-13 | pass | general (general) | General knowledge |
| G-14 | pass | general (general) | General knowledge |
| G-15 | pass | general (general) | General knowledge |
| W-01 | pass | writing (writing) | Writing help |
| W-02 | pass | writing (writing) | Writing help |
| W-03 | pass | writing (writing) | Writing help |
| W-04 | fail | writing (writing) | Writing help |
| W-05 | pass | writing (writing) | Writing help |
| W-06 | pass | writing (writing) | Writing help |
| W-07 | pass | writing (writing) | Writing help |
| W-08 | pass | writing (writing) | Writing help |
| W-09 | pass | writing (writing) | Writing help |
| W-10 | pass | writing (writing) | Writing help |

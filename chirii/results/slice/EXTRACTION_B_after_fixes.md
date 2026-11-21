# Demo slice: rules only against hybrid extraction (B_after_fixes)

20 sentences from `chirii/eval/slice_fresh_sentences_b.py`, run on 2026-11-21 with qwen2.5-1.5b-instruct-q8_0 on the laptop CPU, temperature 0.
**Development run: the rules and hybrid were changed after seeing this set's results. Not a fresh first-try number.**

| | Result |
| :--- | :--- |
| Rules only: fully right | 16/20 |
| Hybrid: fully right | 14/20 |
| Qwen called (rules gave up) | 6/20 |
| Qwen's item accepted by the code check | 6 |
| Qwen's item rejected by the code check (kept as the rules' item, marked for review) | 0 |
| Hybrid wrong but **not** marked for the user to check | 1 |
| Median time for one Qwen call | 3848 ms |

| Sentence | Rules wrong on | Qwen called | Hybrid result | Hybrid wrong on |
| :--- | :--- | :--- | :--- | :--- |
| Bijli ka bill 2,340 hai, 18 Dec tak bharna hai | - | no | rules | - |
| school PTM saturday 10am | - | no | rules | - |
| Physio every Mon and Thu 6pm | - | no | rules | - |
| Insurance renewal for bike due 2nd Jan, ₹1,890 | - | no | rules | - |
| Priya's engagement 27 Dec evening | - | no | rules | - |
| Thyroid tablet empty stomach every morning | - | no | rules | - |
| Shanti aunty owes me 2000 for the saree | - | no | rules | - |
| Salary credit on last working day | - | no | rules | - |
| Flight BLR to DEL 23rd dec 6:40am | - | no | rules | - |
| Water purifier service guy coming day after tomorrow, morning | {'kind': 'note'} | yes | model (check) | {'kind': 'task'} |
| Netflix 649 renews on 14th every month | {'kind': 'note', 'amount': None} | yes | model (check) | {'kind': 'receivable', 'amount': None} |
| Hostel fee 45k by 31 Dec | - | no | rules | - |
| Parso Amma ka blood test hai subah 8 baje | - | no | rules | - |
| Need to get passport photos clicked | - | no | rules | - |
| Diwali was great, Ravi got the new phone | - | yes | model (check) | {'kind': 'important_date'} |
| Doctor said BP check again in 2 weeks | {'date': None} | yes | model | {'date': None} |
| Electrician 4 o'clock today | {'kind': 'note'} | yes | model (check) | {'kind': 'task'} |
| Kal gas cylinder aaya, next booking after 25 days | - | yes | model (check) | {'kind': 'appointment'} |
| Pay maid 7000 on 1st | - | no | rules | - |
| Movie with college friends Sunday 7pm, PVR Phoenix | - | no | rules | - |

Raw Qwen replies are in `extraction_B_after_fixes.json`.

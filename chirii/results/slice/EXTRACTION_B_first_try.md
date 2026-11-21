# Demo slice: rules only against hybrid extraction (B_first_try)

20 sentences from `chirii/eval/slice_fresh_sentences_b.py`, run on 2026-11-21 with qwen2.5-1.5b-instruct-q8_0 on the laptop CPU, temperature 0.
**First try on set B: written after the set A fixes, committed before this run, nothing changed after seeing these results.**

| | Result |
| :--- | :--- |
| Rules only: fully right | 13/20 |
| Hybrid: fully right | 13/20 |
| Qwen called (rules gave up) | 5/20 |
| Qwen's item accepted by the code check | 5 |
| Qwen's item rejected by the code check (kept as the rules' item, marked for review) | 0 |
| Hybrid wrong but **not** marked for the user to check | 7 |
| Median time for one Qwen call | 3327 ms |

| Sentence | Rules wrong on | Qwen called | Hybrid result | Hybrid wrong on |
| :--- | :--- | :--- | :--- | :--- |
| Bijli ka bill 2,340 hai, 18 Dec tak bharna hai | - | no | rules | - |
| school PTM saturday 10am | - | no | rules | - |
| Physio every Mon and Thu 6pm | {'kind': 'appointment', 'weekdays': []} | yes | model | {'kind': 'appointment', 'weekdays': []} |
| Insurance renewal for bike due 2nd Jan, ₹1,890 | - | no | rules | - |
| Priya's engagement 27 Dec evening | - | no | rules | - |
| Thyroid tablet empty stomach every morning | - | no | rules | - |
| Shanti aunty owes me 2000 for the saree | {'amount': None} | yes | model | - |
| Salary credit on last working day | - | no | rules | - |
| Flight BLR to DEL 23rd dec 6:40am | - | no | rules | - |
| Water purifier service guy coming day after tomorrow, morning | {'kind': 'important_date'} | no | rules | {'kind': 'important_date'} |
| Netflix 649 renews on 14th every month | {'kind': 'important_date', 'amount': None} | yes | model | {'kind': 'important_date', 'amount': None} |
| Hostel fee 45k by 31 Dec | - | no | rules | - |
| Parso Amma ka blood test hai subah 8 baje | - | no | rules | - |
| Need to get passport photos clicked | - | no | rules | - |
| Diwali was great, Ravi got the new phone | - | yes | model | {'kind': 'important_date'} |
| Doctor said BP check again in 2 weeks | {'date': None} | yes | model | {'date': None} |
| Electrician 4 o'clock today | {'kind': 'important_date'} | no | rules | {'kind': 'important_date'} |
| Kal gas cylinder aaya, next booking after 25 days | {'kind': 'important_date'} | no | rules | {'kind': 'important_date'} |
| Pay maid 7000 on 1st | - | no | rules | - |
| Movie with college friends Sunday 7pm, PVR Phoenix | - | no | rules | - |

Raw Qwen replies are in `extraction_B_first_try.json`.

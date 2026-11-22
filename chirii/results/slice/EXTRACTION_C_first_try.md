# Demo slice: rules only against hybrid extraction (C_first_try)

20 sentences from `chirii/eval/slice_fresh_sentences_c.py`, run on 2026-11-22 with qwen2.5-1.5b-instruct-q8_0 on the laptop CPU, temperature 0.
**Set C, first try: written and committed (8d75700) after the S4/S5 rule changes and before any code ran on it.**

| | Result |
| :--- | :--- |
| Rules only: fully right | 18/20 |
| Hybrid: fully right | 18/20 |
| Qwen called (rules gave up) | 4/20 |
| Qwen's item accepted by the code check | 4 |
| Qwen's item rejected by the code check (kept as the rules' item, marked for review) | 0 |
| Hybrid wrong but **not** marked for the user to check | 1 |
| Median time for one Qwen call | 3307 ms |

| Sentence | Rules wrong on | Qwen called | Hybrid result | Hybrid wrong on |
| :--- | :--- | :--- | :--- | :--- |
| Car insurance ₹11,200 renew by 3 July | - | no | rules | - |
| Dentist Dr. Shah tomorrow 5:30 pm | - | no | rules | - |
| Tuition for Riya every Tue Thu Sat 4pm | {'weekdays': ['Tuesday']} | no | rules | {'weekdays': ['Tuesday']} |
| Mama ji ki shaadi 14 Nov ko hai Jaipur mein | {'kind': 'note'} | yes | model (check) | {'kind': 'appointment'} |
| Metformin 500 mg after dinner daily | - | no | rules | - |
| Vikram owes me 3.5k for the concert | - | no | rules | - |
| Pension credited on 1st of every month | - | no | rules | - |
| Train to Madurai 28 June 9:15 pm, PNR in email | - | no | rules | - |
| Gas booking next Monday | - | yes | model (check) | - |
| Broadband bill 799 due 22nd | - | no | rules | - |
| Annual fee for the credit card 1,499 in March | - | yes | model | - |
| Parent-teacher meeting rescheduled to 2 July | - | no | rules | - |
| Paid the electricity bill today | - | no | rules | - |
| Kal subah 7 baje yoga class jaana hai | - | no | rules | - |
| My sister Kavitha lives in Hyderabad | - | no | rules | - |
| I'm Suresh, everyone calls me Suri | - | no | rules | - |
| Visa interview at the US consulate on 19 August, 8:45 am | - | no | rules | - |
| Property tax ₹6,400 due 31 January | - | no | rules | - |
| Return library books by Friday | - | no | rules | - |
| Diwali bonus expected in October | - | yes | model (check) | - |

Raw Qwen replies are in `extraction_C_first_try.json`.

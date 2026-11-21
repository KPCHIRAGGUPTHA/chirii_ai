# Demo slice: rules only against hybrid extraction

20 new sentences (`chirii/eval/slice_fresh_sentences.py`, committed before the hybrid code existed), run once on
2026-11-21 with qwen2.5-1.5b-instruct-q8_0 on the laptop CPU, temperature 0. **First try; nothing was changed after seeing these results.**

| | Result |
| :--- | :--- |
| Rules only: fully right | 5/20 |
| Hybrid: fully right | 6/20 |
| Qwen called (rules gave up) | 12/20 |
| Qwen's item accepted by the code check | 3 |
| Qwen's item rejected by the code check (kept as the rules' item, marked for review) | 9 |
| Hybrid wrong but **not** marked for the user to check | 7 |
| Median time for one Qwen call | 5520 ms |

| Sentence | Rules wrong on | Qwen called | Hybrid result | Hybrid wrong on |
| :--- | :--- | :--- | :--- | :--- |
| dentist tmrw 5pm | {'date': None} | yes | rules (model reply rejected) (check) | {'date': None} |
| Kal subah 10 baje bank jaana hai KYC ke liye | {'kind': 'note', 'date': None, 'time': None} | yes | rules (model reply rejected) (check) | {'kind': 'note', 'date': None, 'time': None} |
| electricity bill 1350 by the 15th | {'kind': 'note', 'amount': None, 'date': None} | yes | model | - |
| Remind me to renew the car insurance, it expires on 30th Dec | {'kind': 'important_date'} | no | rules | {'kind': 'important_date'} |
| Ma's cataract surgery 14 Dec, need to be at hospital by 7am | {'kind': 'task'} | no | rules | {'kind': 'task'} |
| Newspaper bill 480 rupees, pay to Sharma ji on 5th of every month | {'day_of_month': None} | yes | model | {'day_of_month': None} |
| Sunday ko Rahul ki birthday party hai, shaam 6 baje | {'date': None, 'time': None} | yes | model | {'time': None} |
| BP medicine twice a day, morning and night | {'kind': 'note'} | yes | rules (model reply rejected) (check) | {'kind': 'note'} |
| Vet appointment for Bruno on the 9th at 4.30 | {'date': None} | yes | rules (model reply rejected) (check) | {'date': None} |
| Need to pay school bus fees ₹3,600 before 10 Dec | {'kind': 'travel'} | no | rules | {'kind': 'travel'} |
| Gas booking done, cylinder should come in 2-3 days | - | yes | rules (model reply rejected) (check) | - |
| Ration: 5 kg atta, 2 kg dal, 1 litre oil | - | yes | rules (model reply rejected) (check) | - |
| Meeting with CA on Friday re ITR, 11am | - | no | rules | - |
| maid on leave from 20th to 24th december | - | no | rules | - |
| Dad's pension comes 1st of month | {'day_of_month': None} | no | rules | {'day_of_month': None} |
| Car EMI 12,400 debit on 7th every month | {'kind': 'note', 'amount': None, 'day_of_month': None} | yes | rules (model reply rejected) (check) | {'kind': 'note', 'amount': None, 'day_of_month': None} |
| Parent teacher meeting 12 dec 9:30 am Divya's school | - | no | rules | - |
| Mummy ka eye checkup 18 tareekh ko 11 baje | {'date': None, 'time': None} | yes | rules (model reply rejected) (check) | {'date': None, 'time': None} |
| Credit card due 22nd, around 8k | {'kind': 'note', 'date': None, 'amount': None} | yes | rules (model reply rejected) (check) | {'kind': 'note', 'date': None, 'amount': None} |
| Holi isn't till March but book train tickets by 10 Jan | {'kind': 'travel'} | no | rules | {'kind': 'travel'} |

Raw Qwen replies are in `extraction.json`.

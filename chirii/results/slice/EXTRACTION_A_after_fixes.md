# Demo slice: rules only against hybrid extraction (A_after_fixes)

20 sentences from `chirii/eval/slice_fresh_sentences.py`, run on 2026-11-21 with qwen2.5-1.5b-instruct-q8_0 on the laptop CPU, temperature 0.
**Development run: the rules and hybrid were changed after seeing this set's results. Not a fresh first-try number.**

| | Result |
| :--- | :--- |
| Rules only: fully right | 19/20 |
| Hybrid: fully right | 17/20 |
| Qwen called (rules gave up) | 3/20 |
| Qwen's item accepted by the code check | 3 |
| Qwen's item rejected by the code check (kept as the rules' item, marked for review) | 0 |
| Hybrid wrong but **not** marked for the user to check | 1 |
| Median time for one Qwen call | 3575 ms |

| Sentence | Rules wrong on | Qwen called | Hybrid result | Hybrid wrong on |
| :--- | :--- | :--- | :--- | :--- |
| dentist tmrw 5pm | - | no | rules | - |
| Kal subah 10 baje bank jaana hai KYC ke liye | - | no | rules | - |
| electricity bill 1350 by the 15th | - | no | rules | - |
| Remind me to renew the car insurance, it expires on 30th Dec | - | no | rules | - |
| Ma's cataract surgery 14 Dec, need to be at hospital by 7am | - | no | rules | - |
| Newspaper bill 480 rupees, pay to Sharma ji on 5th of every month | - | no | rules | - |
| Sunday ko Rahul ki birthday party hai, shaam 6 baje | - | no | rules | - |
| BP medicine twice a day, morning and night | - | no | rules | - |
| Vet appointment for Bruno on the 9th at 4.30 | - | no | rules | - |
| Need to pay school bus fees ₹3,600 before 10 Dec | - | no | rules | - |
| Gas booking done, cylinder should come in 2-3 days | - | yes | model (check) | {'kind': 'task'} |
| Ration: 5 kg atta, 2 kg dal, 1 litre oil | - | yes | model (check) | {'kind': 'receivable'} |
| Meeting with CA on Friday re ITR, 11am | - | no | rules | - |
| maid on leave from 20th to 24th december | - | yes | model (check) | - |
| Dad's pension comes 1st of month | - | no | rules | - |
| Car EMI 12,400 debit on 7th every month | - | no | rules | - |
| Parent teacher meeting 12 dec 9:30 am Divya's school | - | no | rules | - |
| Mummy ka eye checkup 18 tareekh ko 11 baje | {'time': None} | no | rules | {'time': None} |
| Credit card due 22nd, around 8k | - | no | rules | - |
| Holi isn't till March but book train tickets by 10 Jan | - | no | rules | - |

Raw Qwen replies are in `extraction_A_after_fixes.json`.

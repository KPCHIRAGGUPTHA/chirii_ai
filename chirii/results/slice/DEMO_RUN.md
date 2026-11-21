# Demo slice: end-to-end check

Run on 2026-11-21 through the web API on the laptop CPU (Qwen2.5-1.5B-Instruct Q8_0, llama.cpp).
Fictional memories written for this check, not S1 tasks. Not scored: read by hand.

## Memories added

| Said | Read as | Source | Check? | Linked to earlier item |
| :--- | :--- | :--- | :--- | :--- |
| My Airtel broadband bill is ₹999, due on 28 November. | kind=bill, date=2026-11-28, amount=999 (33 ms) | rules | no | - |
| Dad's eye surgery at Sankara Nethralaya on 3 December at 9 am. Carry his Aadhaar and old prescriptions. | kind=appointment, date=2026-12-03, time=09:00 (21 ms) | rules | no | - |
| Pay the cook ₹4,500 on the 1st of every month. | kind=bill, amount=4500, day_of_month=1 (20 ms) | rules | no | - |
| Kal subah 7 baje milk wala ko paise dene hai | kind=appointment, date=2026-11-22, time=07:00 (21 ms) | rules | no | - |
| Note: the Wi-Fi router password is on the sticker under the router. | kind=note (19 ms) | rules | no | - |
| Anjali's birthday is on 2 December. She wants a sketchbook. | kind=important_date, date=2026-12-02 (19 ms) | rules | no | - |
| I paid the broadband bill. | kind=update (49 ms) | rules | no | 1 |

## Questions (today is 2026-11-21)

| Question | Answer | Time |
| :--- | :--- | ---: |
| When is Dad's surgery and what should I carry? | Dad's eye surgery is scheduled for Saturday 21 November 2026 at 9:00 am. You should carry his Aadhaar and old prescriptions. | 9.2 s |
| Is the broadband bill paid? | Yes, the broadband bill is PAID. It was due on Saturday 28 November 2026 and the amount was ₹999. | 6.9 s |
| What does Anjali want for her birthday? | Anjali wants a sketchbook for her birthday. | 5.2 s |
| When do I pay the cook? | The user said to pay the cook ₹4,500 on the 1st of every month. The current details show this payment is due on day 1 of every month. | 6.8 s |
| What is my PAN number? | I don't have that in your memory. | 4.2 s |

Delete all my data at the end: 7 items and 7 vectors removed.

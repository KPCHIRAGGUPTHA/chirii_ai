# Demo slice: end-to-end check

Run on 2026-11-21 through the web API on the laptop CPU (Qwen2.5-1.5B-Instruct Q8_0, llama.cpp).
Fictional memories written for this check, not S1 tasks. Not scored: read by hand.

Each memory's "Got it" card is shown, then confirmed (as a user pressing Yes would).

## Memories added

| Said | Got it card | Unsure about | Source | Time |
| :--- | :--- | :--- | :--- | ---: |
| My Airtel broadband bill is ₹999, due on 28 November. | Bill, ₹999, due Saturday 28 November 2026 | - | rules | 31 ms |
| Dad's eye surgery at Sankara Nethralaya on 3 December at 9 am. Carry his Aadhaar and old prescriptions. | Appointment, on Thursday 3 December 2026, at 9:00 am | - | rules | 15 ms |
| Pay the cook ₹4,500 on the 1st of every month. | Bill, ₹4,500, on the 1st of every month | - | rules | 14 ms |
| Kal subah 7 baje milk wala ko paise dene hai | Appointment, on Sunday 22 November 2026, at 7:00 am | - | rules | 12 ms |
| Note: the Wi-Fi router password is on the sticker under the router. | Note | - | rules | 14 ms |
| Anjali's birthday is on 2 December. She wants a sketchbook. | Important date, on Wednesday 2 December 2026 | - | rules | 13 ms |
| I paid the broadband bill. | Change to an earlier item, updated "Airtel broadband bill is ₹999": marked paid | - | rules | 33 ms |

## Questions (today is 2026-11-21)

| Question | Answer | Memory lines given | Fact check | Time |
| :--- | :--- | ---: | :--- | ---: |
| When is Dad's surgery and what should I carry? | Dad's eye surgery is scheduled for Thursday 3 December 2026 at 9:00 am. You should carry his Aadhaar and old prescriptions. | 1 | pass | 5.0 s |
| Is the broadband bill paid? | Yes, the broadband bill is PAID. | 1 | pass | 1.5 s |
| What does Anjali want for her birthday? | Anjali wants a sketchbook for her birthday. | 1 | pass | 1.4 s |
| When do I pay the cook? | You pay the cook ₹4,500 on the 1st of every month. | 1 | pass | 1.9 s |
| What is my PAN number? | I don't have that in your memory. | 1 | pass | 0.7 s |

Delete all my data at the end: 7 items and 7 vectors removed.

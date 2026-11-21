"""
20 new sentences for the demo slice: casual, incomplete, mixed order, and a few with Hinglish words. Written and
committed before the hybrid extractor (rules first, Qwen when the rules give up) was written. They are run once, rules
only against hybrid, and the first-try result is reported as is.

Each expected item lists only the fields a careful person would agree on. Where the sentence is ambiguous (a time with
no am/pm and no clue), the field is left out rather than guessed. All said on Wednesday 2 December 2026.
"""

from datetime import date

SAID_ON = date(2026, 12, 2)

SLICE_FRESH = [
    ("dentist tmrw 5pm", {"kind": "appointment", "date": "2026-12-03", "time": "17:00"}),
    ("Kal subah 10 baje bank jaana hai KYC ke liye", {"kind": "task", "date": "2026-12-03", "time": "10:00"}),
    ("electricity bill 1350 by the 15th", {"kind": "bill", "amount": 1350, "date": "2026-12-15"}),
    ("Remind me to renew the car insurance, it expires on 30th Dec", {"kind": "task", "date": "2026-12-30"}),
    ("Ma's cataract surgery 14 Dec, need to be at hospital by 7am", {"kind": "appointment", "date": "2026-12-14", "time": "07:00"}),
    ("Newspaper bill 480 rupees, pay to Sharma ji on 5th of every month", {"kind": "bill", "amount": 480, "day_of_month": 5}),
    ("Sunday ko Rahul ki birthday party hai, shaam 6 baje", {"kind": "important_date", "date": "2026-12-06", "time": "18:00"}),
    ("BP medicine twice a day, morning and night", {"kind": "medicine"}),
    ("Vet appointment for Bruno on the 9th at 4.30", {"kind": "appointment", "date": "2026-12-09"}),
    ("Need to pay school bus fees ₹3,600 before 10 Dec", {"kind": "bill", "amount": 3600, "date": "2026-12-10"}),
    ("Gas booking done, cylinder should come in 2-3 days", {"kind": "note", "date": None}),
    ("Ration: 5 kg atta, 2 kg dal, 1 litre oil", {"kind": "note", "amount": None}),
    ("Meeting with CA on Friday re ITR, 11am", {"kind": "appointment", "date": "2026-12-04", "time": "11:00"}),
    ("maid on leave from 20th to 24th december", {"date": "2026-12-20", "end_date": "2026-12-24"}),
    ("Dad's pension comes 1st of month", {"kind": "income", "day_of_month": 1}),
    ("Car EMI 12,400 debit on 7th every month", {"kind": "bill", "amount": 12400, "day_of_month": 7}),
    ("Parent teacher meeting 12 dec 9:30 am Divya's school", {"kind": "appointment", "date": "2026-12-12", "time": "09:30"}),
    ("Mummy ka eye checkup 18 tareekh ko 11 baje", {"kind": "appointment", "date": "2026-12-18", "time": "11:00"}),
    ("Credit card due 22nd, around 8k", {"kind": "bill", "date": "2026-12-22", "amount": 8000}),
    ("Holi isn't till March but book train tickets by 10 Jan", {"kind": "task", "date": "2027-01-10"}),
]

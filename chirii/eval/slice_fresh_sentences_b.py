"""
Second set of 20 new sentences for the demo slice (set B). Written and committed after the rules and the hybrid schema
were changed in response to set A, and before any code was run on them. Set A became development data when the rules
were changed after seeing it; set B is the fresh first-try check of those changes.

Same rules as set A: each expected item lists only the fields a careful person would agree on, and a time with no
am/pm and no clue is left out. All said on Friday 4 December 2026.
"""

from datetime import date

SAID_ON = date(2026, 12, 4)

SLICE_FRESH_B = [
    ("Bijli ka bill 2,340 hai, 18 Dec tak bharna hai", {"kind": "bill", "amount": 2340, "date": "2026-12-18"}),
    ("school PTM saturday 10am", {"kind": "appointment", "date": "2026-12-05", "time": "10:00"}),
    ("Physio every Mon and Thu 6pm", {"kind": "routine", "weekdays": ["Monday", "Thursday"], "time": "18:00"}),
    ("Insurance renewal for bike due 2nd Jan, ₹1,890", {"kind": "bill", "date": "2027-01-02", "amount": 1890}),
    ("Priya's engagement 27 Dec evening", {"kind": "important_date", "date": "2026-12-27"}),
    ("Thyroid tablet empty stomach every morning", {"kind": "medicine"}),
    ("Shanti aunty owes me 2000 for the saree", {"kind": "receivable", "amount": 2000}),
    ("Salary credit on last working day", {"kind": "income"}),
    ("Flight BLR to DEL 23rd dec 6:40am", {"kind": "travel", "date": "2026-12-23", "time": "06:40"}),
    ("Water purifier service guy coming day after tomorrow, morning", {"kind": "appointment", "date": "2026-12-06"}),
    ("Netflix 649 renews on 14th every month", {"kind": "bill", "amount": 649, "day_of_month": 14}),
    ("Hostel fee 45k by 31 Dec", {"kind": "bill", "amount": 45000, "date": "2026-12-31"}),
    ("Parso Amma ka blood test hai subah 8 baje", {"kind": "appointment", "date": "2026-12-06", "time": "08:00"}),
    ("Need to get passport photos clicked", {"kind": "task", "date": None}),
    ("Diwali was great, Ravi got the new phone", {"kind": "note", "date": None}),
    ("Doctor said BP check again in 2 weeks", {"date": "2026-12-18"}),
    ("Electrician 4 o'clock today", {"kind": "appointment", "date": "2026-12-04"}),
    ("Kal gas cylinder aaya, next booking after 25 days", {"kind": "note", "date": "2026-12-03"}),
    ("Pay maid 7000 on 1st", {"kind": "bill", "amount": 7000, "date": "2027-01-01"}),
    ("Movie with college friends Sunday 7pm, PVR Phoenix", {"kind": "appointment", "date": "2026-12-06", "time": "19:00"}),
]

"""
22 new sentences, written after the parser and never seen while writing it, with the item each should become. They
estimate how well the parser handles sentences it wasn't written for. The first-try result is recorded in
docs/PROGRESS.md (S2) before any rule was changed. All said on Tuesday 1 December 2026.
"""

from datetime import date

SAID_ON = date(2026, 12, 1)

FRESH = [
    ("Piped gas bill of Rs 740 is due on 9th Dec.", {"kind": "bill", "date": "2026-12-09", "amount": 740}),
    ("Physiotherapy session on Saturday at 4:30 pm.", {"kind": "appointment", "date": "2026-12-05", "time": "16:30"}),
    ("Pay the maid ₹6,000 on the 1st of every month.", {"kind": "bill", "day_of_month": 1, "amount": 6000}),
    ("Car service booked for 3 January, 9 am.", {"kind": "appointment", "date": "2027-01-03", "time": "09:00"}),
    ("Swimming class every Monday, Wednesday and Friday at 7 am.", {"kind": "routine", "weekdays": ["Monday", "Wednesday", "Friday"], "time": "07:00"}),
    ("Uncle's sixtieth birthday party is on 19 Dec.", {"kind": "important_date", "date": "2026-12-19"}),
    ("Take Vitamin D 60000 IU once a week.", {"kind": "medicine"}),
    ("Rent agreement renewal must be done by 15 January.", {"kind": "task", "date": "2027-01-15"}),
    ("Credit card bill of ₹12,430 due on December 18.", {"kind": "bill", "date": "2026-12-18", "amount": 12430}),
    ("Electrician visit tomorrow at 11 am.", {"kind": "appointment", "date": "2026-12-02", "time": "11:00"}),
    ("Ravi owes me 1500 rupees for the concert tickets.", {"kind": "receivable", "amount": 1500}),
    ("Flight to Delhi on 20 December at 6:15 am.", {"kind": "travel", "date": "2026-12-20", "time": "06:15"}),
    ("The parent-teacher meeting moved to 12 December.", {"kind": "update", "action": "moved", "date": "2026-12-12"}),
    ("I paid the gas bill.", {"kind": "update", "action": "paid"}),
    ("Note: Wi-Fi password is with the landlord.", {"kind": "note"}),
    ("Diwali shopping budget is ₹15,000.", {"kind": "note", "amount": 15000}),
    ("Property tax of ₹4,820 must be paid by 31 March.", {"kind": "bill", "date": "2027-03-31", "amount": 4820}),
    ("Blood test at the lab on 8 Dec between 7 and 9 am, fasting.", {"kind": "appointment", "date": "2026-12-08", "time": "07:00", "end_time": "09:00"}),
    ("Pension of ₹32,000 comes on the last working day.", {"kind": "income", "amount": 32000}),
    ("Grandpa's death anniversary puja on 2 January.", {"kind": "important_date", "date": "2027-01-02"}),
    ("Collect Aadhaar card from the post office before 10 December.", {"kind": "task", "date": "2026-12-10"}),
    ("Insurance premium ₹1.2 lakh due 25 Jan.", {"kind": "bill", "date": "2027-01-25", "amount": 120000}),
]

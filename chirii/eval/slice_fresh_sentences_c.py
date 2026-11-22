"""
Third set of 20 new sentences (set C), the fresh extraction set planned since S3. Written and committed in S5, after
the S4 and S5 changes to the rules (bill words, "done" updates, people and profile kinds, the S5 year rule) and before
any code was run on them. Target (CRITERIA.md): wrong and not flagged for the user to check ≤ 2 of 20.

Same rules as sets A and B: each expected item lists only the fields a careful person would agree on; a time with no
am/pm and no clue is left out. All said on Wednesday 9 June 2027. Fictional.
"""

from datetime import date

SAID_ON = date(2027, 6, 9)

SLICE_FRESH_C = [
    ("Car insurance ₹11,200 renew by 3 July", {"kind": "bill", "amount": 11200, "date": "2027-07-03"}),
    ("Dentist Dr. Shah tomorrow 5:30 pm", {"kind": "appointment", "date": "2027-06-10", "time": "17:30"}),
    ("Tuition for Riya every Tue Thu Sat 4pm", {"kind": "routine", "weekdays": ["Tuesday", "Thursday", "Saturday"], "time": "16:00"}),
    ("Mama ji ki shaadi 14 Nov ko hai Jaipur mein", {"kind": "important_date", "date": "2027-11-14"}),
    ("Metformin 500 mg after dinner daily", {"kind": "medicine"}),
    ("Vikram owes me 3.5k for the concert", {"kind": "receivable", "amount": 3500}),
    ("Pension credited on 1st of every month", {"kind": "income", "day_of_month": 1}),
    ("Train to Madurai 28 June 9:15 pm, PNR in email", {"kind": "travel", "date": "2027-06-28", "time": "21:15"}),
    ("Gas booking next Monday", {"date": "2027-06-14"}),
    ("Broadband bill 799 due 22nd", {"kind": "bill", "amount": 799, "date": "2027-06-22"}),
    ("Annual fee for the credit card 1,499 in March", {"kind": "bill", "amount": 1499}),
    ("Parent-teacher meeting rescheduled to 2 July", {"kind": "update", "date": "2027-07-02"}),
    ("Paid the electricity bill today", {"kind": "update"}),
    ("Kal subah 7 baje yoga class jaana hai", {"date": "2027-06-10", "time": "07:00"}),
    ("My sister Kavitha lives in Hyderabad", {"kind": "person"}),
    ("I'm Suresh, everyone calls me Suri", {"kind": "profile"}),
    ("Visa interview at the US consulate on 19 August, 8:45 am", {"kind": "appointment", "date": "2027-08-19", "time": "08:45"}),
    ("Property tax ₹6,400 due 31 January", {"kind": "bill", "amount": 6400, "date": "2028-01-31"}),
    ("Return library books by Friday", {"kind": "task", "date": "2027-06-11"}),
    ("Diwali bonus expected in October", {"date": None}),
]

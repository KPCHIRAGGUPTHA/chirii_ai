"""
Builds the S1 everyday-assistant eval: 50 scored tasks and 5 unscored Hinglish tasks.

Each task gives one fictional person's memory, as the sentences they told Chirii earlier, then a question or a planning
request on a fixed "today". Each sentence also has its intended structured item, which documents the expected meaning
and lets the tests check the marking rules against the memory. Writes s1_tasks.jsonl and S1_TASKS.md (for review).

All people, places and numbers are fictional. Usage:
    python chirii/eval/build_s1_tasks.py
"""

import json
import os
from datetime import date, timedelta
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_PATH = os.path.join(HERE, "s1_tasks.jsonl")
REVIEW_PATH = os.path.join(HERE, "S1_TASKS.md")

TYPES = {
    "lookup": "Exact lookup",
    "date_window": "Date arithmetic",
    "semantic": "Meaning-based search over notes",
    "plan": "Daily or weekly plan",
    "update": "Update or correction",
    "unknown": "Not in memory (must say so)",
}
PERSONAS = {
    "priya": "Priya, 32, software tester in Bengaluru on rotating shifts (working adult)",
    "lakshmi": "Lakshmi, 45, homemaker in Chennai; husband Murali, children Divya (Class 10) and Karthik (Class 6)",
    "ramesh": "Ramesh, 52, runs a kirana shop in Pune (small shop owner)",
    "sharada": "Sharada, 72, retired teacher in Mysuru with diabetes and high BP (elderly person)",
    "arjun": "Arjun, 20, B.Tech student in a hostel at VIT Vellore (student)",
}


def said(text: str, kind: str, title: str, **fields: Any) -> Dict[str, Any]:
    """A sentence the user told Chirii, with the structured item it should become."""
    return {"said": text, "item": {"type": kind, "title": title, **fields}}


def D(iso: str) -> Dict[str, str]:
    return {"date": iso}


def A(amount: int) -> Dict[str, int]:
    return {"amount": amount}


def T(hhmm: str) -> Dict[str, str]:
    return {"time": hhmm}


def X(*phrases: str) -> Dict[str, List[str]]:
    return {"text": list(phrases)}


def ANY(*facts: Dict[str, Any], min: int = 1) -> Dict[str, Any]:
    return {"any_of": list(facts), "min": min}


# ---------------------------------------------------------------- base memories, one per person

PRIYA = [
    said("My BESCOM electricity bill is ₹1,860 and it's due on 12 December.", "bill", "BESCOM electricity bill", date="2026-12-12", amount=1860),
    said("My home loan EMI of ₹18,500 goes out on the 5th of every month.", "bill", "home loan EMI", day_of_month=5, amount=18500),
    said("The car insurance premium is ₹9,240, due on 28 December.", "bill", "car insurance premium", date="2026-12-28", amount=9240),
    said("Dentist appointment with Dr. Kavitha on 10 December at 6 pm.", "appointment", "dentist, Dr. Kavitha", date="2026-12-10", time="18:00"),
    said("From 14 to 18 December I'm on the night shift, 10 pm to 6 am.", "shift", "night shift", date="2026-12-14", end_date="2026-12-18", time="22:00"),
    said("Amma's birthday is on 22 December.", "important_date", "Amma's birthday", date="2026-12-22"),
    said("The bank says I must update my KYC at the SBI Koramangala branch before 31 December.", "task", "SBI KYC update", date="2026-12-31"),
    said("Note from today's team meeting: the release has moved to 8 January, and I'm on call that weekend.", "note", "release moved, on call", date="2027-01-08"),
    said("The gym fee of ₹2,500 is due on 1 January.", "bill", "gym fee", date="2027-01-01", amount=2500),
]

LAKSHMI = [
    said("Divya's school term fee of ₹24,000 must be paid by 18 December.", "bill", "Divya's term fee", date="2026-12-18", amount=24000),
    said("Karthik's school annual day is on 19 December at 5 pm.", "appointment", "Karthik's annual day", date="2026-12-19", time="17:00"),
    said("Our Metro Water bill is ₹420, due on 10 December.", "bill", "Metro Water bill", date="2026-12-10", amount=420),
    said("The electricity bill is ₹2,150, due on 20 December.", "bill", "electricity bill", date="2026-12-20", amount=2150),
    said("LIC premium for Murali's policy, ₹12,600, is due on 3 January.", "bill", "LIC premium (Murali)", date="2027-01-03", amount=12600),
    said("Our wedding anniversary is on 27 December.", "important_date", "wedding anniversary", date="2026-12-27"),
    said("Shopping list for Pongal: rice, jaggery, moong dal, ghee, sugarcane and turmeric plants.", "note", "Pongal shopping list"),
    said("Mother-in-law's eye check-up at the eye hospital on 16 December at 10 am. Take her old reports.", "appointment", "mother-in-law's eye check-up", date="2026-12-16", time="10:00"),
    said("I take Divya to her maths tuition every Tuesday and Thursday at 5 pm; it ends at 6:30 pm.", "routine", "Divya's maths tuition", weekdays=["Tuesday", "Thursday"], time="17:00", end_time="18:30"),
    said("The last gas cylinder was delivered on 25 November.", "note", "last gas cylinder delivery", date="2026-11-25"),
]

RAMESH = [
    said("The shop electricity bill is ₹6,380, due on 14 December.", "bill", "shop electricity bill", date="2026-12-14", amount=6380),
    said("Shop rent of ₹22,000 goes to Mr. Joshi on the 1st of every month.", "bill", "shop rent", day_of_month=1, amount=22000),
    said("The GSTR-3B return for November is due on 20 December.", "task", "GSTR-3B return", date="2026-12-20"),
    said("Gupta Traders will deliver 50 bags of sugar on 11 December between 9 and 11 am; I pay ₹1,95,000 on delivery.", "bill", "sugar delivery, Gupta Traders", date="2026-12-11", time="09:00", amount=195000),
    said("I need to update the KYC for the current account at the bank by 15 December.", "task", "current account KYC", date="2026-12-15"),
    said("Mrs. Deshpande owes ₹3,450 on credit and promised to pay by 13 December.", "receivable", "Mrs. Deshpande's credit", date="2026-12-13", amount=3450),
    said("I'm closing the shop on 26 December for my nephew's engagement in Satara.", "important_date", "shop closed, nephew's engagement", date="2026-12-26"),
    said("Note: the new milk vendor charges ₹56 a litre from January, up from ₹54.", "note", "milk price", amount=56, other_amounts=[54]),
    said("Shop insurance premium of ₹7,800 is due on 9 January.", "bill", "shop insurance premium", date="2027-01-09", amount=7800),
]

SHARADA = [
    said("I take Metformin 500 mg after breakfast and after dinner.", "medicine", "Metformin 500 mg", times=["after breakfast", "after dinner"]),
    said("Amlodipine 5 mg, one tablet at night, for my blood pressure.", "medicine", "Amlodipine 5 mg", times=["night"]),
    said("My next check-up is with Dr. Ramaswamy at the hospital on 18 December at 11 am. I have a fasting blood sugar test the same morning at 8 am.",
         "appointment", "check-up with Dr. Ramaswamy, fasting test first", date="2026-12-18", time="11:00", other_times=["08:00"]),
    said("My pension is credited on the 1st of every month.", "income", "pension", day_of_month=1),
    said("I submitted my life certificate on 20 November.", "note", "life certificate submitted", date="2026-11-20"),
    said("Granddaughter Ananya's birthday is on 9 January. She wants a story book.", "important_date", "Ananya's birthday", date="2027-01-09"),
    said("Dr. Ramaswamy said: walk 30 minutes every morning, avoid sweets and reduce salt.", "note", "doctor's advice"),
    said("The water bill is ₹180, due on 15 December.", "bill", "water bill", date="2026-12-15", amount=180),
    said("My bhajan group meets every Thursday at 4 pm at the temple.", "routine", "bhajan group", weekdays=["Thursday"], time="16:00"),
    said("My spectacles are with the optician and will be ready on 12 December.", "task", "collect spectacles", date="2026-12-12"),
]

ARJUN = [
    said("Semester fee of ₹1,98,000 must be paid by 5 January.", "bill", "semester fee", date="2027-01-05", amount=198000),
    said("Hostel mess fee of ₹38,500 is due on 20 December.", "bill", "hostel mess fee", date="2026-12-20", amount=38500),
    said("FAT exams: Data Structures on 7 December, DBMS on 9 December, Operating Systems on 11 December, all at 2 pm.",
         "exam", "FAT exams", date="2026-12-07", other_dates=["2026-12-09", "2026-12-11"], time="14:00"),
    said("Train home to Hyderabad on 13 December at 9:40 pm from Katpadi. The PNR is saved in the IRCTC app.", "travel", "train to Hyderabad", date="2026-12-13", time="21:40"),
    said("Mom's birthday is on 24 December.", "important_date", "Mom's birthday", date="2026-12-24"),
    said("Note from DBMS class: the exam covers normalisation, SQL joins and transactions. ER diagrams are not in the syllabus.", "note", "DBMS exam topics"),
    said("Project review with Prof. Meena on 10 December at 10 am.", "appointment", "project review, Prof. Meena", date="2026-12-10", time="10:00"),
    said("Laptop EMI of ₹3,200 on the 7th of every month.", "bill", "laptop EMI", day_of_month=7, amount=3200),
    said("The post-matric scholarship form's last date is 22 December.", "task", "scholarship form", date="2026-12-22"),
]


def task(tid: str, persona: str, kind: str, today: str, question: str, base: List[Dict[str, Any]], rules: Dict[str, Any],
         extra: List[Dict[str, Any]] = (), allow: List[Dict[str, Any]] = (), note: str = "", scored: bool = True) -> Dict[str, Any]:
    memory = list(base) + list(extra)
    # Dates listed as other_dates inside an item are also part of memory
    for entry in memory:
        for other in entry["item"].get("other_dates", []):
            allow = list(allow) + [D(other)]
    return {
        "id": tid, "persona": persona, "type": kind, "today": today, "question": question, "memory": memory,
        "rules": rules, "allow": list(allow), "note": note, "scored": scored,
    }


def build_tasks() -> List[Dict[str, Any]]:
    t = []

    # ------------------------------------------------------------ Priya (working adult)
    t.append(task("S1-01", "priya", "lookup", "2026-12-07", "When is my electricity bill due, and how much is it?", PRIYA,
                  {"must_include": [D("2026-12-12"), A(1860)]}))
    t.append(task("S1-02", "priya", "lookup", "2026-12-07", "What time is my dentist appointment?", PRIYA,
                  {"must_include": [T("18:00"), D("2026-12-10")]}))
    t.append(task("S1-03", "priya", "date_window", "2026-12-07", "Which payments are due in the next 7 days?", PRIYA,
                  {"must_include": [A(1860)], "must_not": [A(9240), A(2500)]},
                  note="Window 7–14 Dec: only the electricity bill. The EMI already went on 5 Dec."))
    t.append(task("S1-04", "priya", "date_window", "2026-12-24", "What do I need to pay or finish before the end of this month?", PRIYA,
                  {"must_include": [A(9240), X("KYC")], "must_not": [A(1860), A(2500)]},
                  note="Window 24–31 Dec: car insurance (28 Dec) and the KYC update (31 Dec). The gym fee falls on 1 Jan."))
    t.append(task("S1-05", "priya", "semantic", "2026-12-09", "What did I note about the release?", PRIYA,
                  {"must_include": [D("2027-01-08"), X("on call", "on-call")]}))
    t.append(task("S1-06", "priya", "plan", "2026-12-10", "Plan my day.", PRIYA,
                  {"must_include": [T("11:00"), X("courier"), T("18:00")], "order": [T("11:00"), X("courier"), X("dentist", "Dr. Kavitha")]},
                  extra=[said("On 10 December I have a team call at 11 am.", "appointment", "team call", date="2026-12-10", time="11:00"),
                         said("I must collect a courier from the office reception before 5 pm on 10 December.", "task", "collect courier", date="2026-12-10", time="17:00")],
                  note="Order: team call 11 am, courier before 5 pm, dentist 6 pm. No clash."))
    t.append(task("S1-07", "priya", "plan", "2026-12-15", "Plan my day for today.", PRIYA,
                  {"must_include": [T("16:00"), T("20:00"), X("night shift", "shift")], "flag_clash": True},
                  extra=[said("Riya's wedding reception is on 15 December from 8 pm to 11 pm.", "appointment", "Riya's wedding reception", date="2026-12-15", time="20:00", end_time="23:00"),
                         said("The electrician is coming at 4 pm on 15 December.", "appointment", "electrician visit", date="2026-12-15", time="16:00")],
                  note="Clash: the reception (8–11 pm) overlaps the night shift starting at 10 pm."))
    t.append(task("S1-08", "priya", "update", "2026-12-09", "Which bills do I still have to pay this month?", PRIYA,
                  {"must_include": [A(9240)], "must_not": [A(1860)]},
                  extra=[said("I paid the electricity bill today.", "update", "electricity bill paid", date="2026-12-08", status="paid")],
                  note="Said on 8 Dec. The EMI may be mentioned; it is not marked wrong."))
    t.append(task("S1-09", "priya", "update", "2026-12-09", "When is my dentist appointment now?", PRIYA,
                  {"must_include": [D("2026-12-11"), T("19:00")], "must_not": [D("2026-12-10"), T("18:00")]},
                  extra=[said("Correction: the dentist moved my appointment to 11 December at 7 pm.", "update", "dentist moved", date="2026-12-11", time="19:00")]))
    t.append(task("S1-10", "priya", "unknown", "2026-12-07", "When is my water bill due?", PRIYA, {"abstain": True},
                  note="No water bill was ever mentioned."))

    # ------------------------------------------------------------ Lakshmi (homemaker)
    t.append(task("S1-11", "lakshmi", "lookup", "2026-12-07", "How much is Divya's school fee, and what is the last date?", LAKSHMI,
                  {"must_include": [A(24000), D("2026-12-18")]}))
    t.append(task("S1-12", "lakshmi", "lookup", "2026-12-07", "When is our LIC premium due?", LAKSHMI,
                  {"must_include": [D("2027-01-03"), A(12600)]}))
    t.append(task("S1-13", "lakshmi", "date_window", "2026-12-14", "Which payments are due this week?", LAKSHMI,
                  {"must_include": [A(24000), A(2150)], "must_not": [A(420), A(12600)]},
                  note="Week Mon 14 – Sun 20 Dec: Divya's fee (18 Dec) and electricity (20 Dec)."))
    t.append(task("S1-14", "lakshmi", "date_window", "2026-12-24", "Any important family dates in the next 7 days?", LAKSHMI,
                  {"must_include": [D("2026-12-27")], "must_not": [D("2026-12-19"), D("2026-12-16")]},
                  note="Window 24–31 Dec: the wedding anniversary (27 Dec)."))
    t.append(task("S1-15", "lakshmi", "semantic", "2026-12-07", "What do I need to buy for Pongal?", LAKSHMI,
                  {"must_include": [ANY(X("rice"), X("jaggery"), X("moong dal"), X("ghee"), X("sugarcane"), X("turmeric"), min=5)]}))
    t.append(task("S1-16", "lakshmi", "semantic", "2026-12-15", "What should I carry for my mother-in-law's eye check-up?", LAKSHMI,
                  {"must_include": [X("reports", "report")]}))
    t.append(task("S1-17", "lakshmi", "plan", "2026-12-17", "Plan my day.", LAKSHMI,
                  {"must_include": [T("11:00"), T("17:00"), T("17:30")], "flag_clash": True},
                  extra=[said("The plumber is coming at 11 am on 17 December.", "appointment", "plumber visit", date="2026-12-17", time="11:00"),
                         said("Murali's colleague's housewarming is on 17 December at 5:30 pm.", "appointment", "housewarming", date="2026-12-17", time="17:30")],
                  note="17 Dec is a Thursday, so tuition (5–6:30 pm) clashes with the 5:30 pm housewarming."))
    t.append(task("S1-18", "lakshmi", "update", "2026-12-11", "Is the water bill paid?", LAKSHMI,
                  {"must_include": [X("paid", "already")], "must_not": [X("not paid", "unpaid", "still due", "haven't paid", "has not been paid")]},
                  extra=[said("I paid the water bill today.", "update", "water bill paid", date="2026-12-09", status="paid")],
                  note="Said on 9 Dec."))
    t.append(task("S1-19", "lakshmi", "update", "2026-12-17", "What is the last date for Divya's school fee?", LAKSHMI,
                  {"must_include": [D("2026-12-22")], "must_not": [D("2026-12-18")]},
                  extra=[said("The school has extended the fee deadline to 22 December.", "update", "fee deadline extended", date="2026-12-22")]))
    t.append(task("S1-20", "lakshmi", "unknown", "2026-12-07", "What time does Karthik's school bus come in the morning?", LAKSHMI, {"abstain": True}))

    # ------------------------------------------------------------ Ramesh (shop owner)
    t.append(task("S1-21", "ramesh", "lookup", "2026-12-07", "When does Gupta Traders deliver the sugar, and how much do I pay?", RAMESH,
                  {"must_include": [D("2026-12-11"), A(195000)]}))
    t.append(task("S1-22", "ramesh", "lookup", "2026-12-07", "When is the GST return due?", RAMESH,
                  {"must_include": [D("2026-12-20")]}))
    t.append(task("S1-23", "ramesh", "date_window", "2026-12-10", "What do I have to pay in the next 5 days?", RAMESH,
                  {"must_include": [A(195000), A(6380)], "must_not": [A(22000), A(7800)]},
                  note="Window 10–15 Dec: the sugar payment (11 Dec) and the shop electricity bill (14 Dec)."))
    t.append(task("S1-24", "ramesh", "date_window", "2026-12-12", "Is anything due tomorrow?", RAMESH,
                  {"must_include": [X("Deshpande"), A(3450)]},
                  note="Tomorrow is 13 Dec: Mrs. Deshpande promised to pay ₹3,450."))
    t.append(task("S1-25", "ramesh", "semantic", "2026-12-07", "What did I note about the milk price?", RAMESH,
                  {"must_include": [A(56)]}))
    t.append(task("S1-26", "ramesh", "plan", "2026-12-11", "Plan my day.", RAMESH,
                  {"must_include": [X("sugar", "Gupta"), T("15:00"), T("19:30")], "order": [X("sugar", "Gupta"), X("bank", "deposit"), X("station")]},
                  extra=[said("Deposit yesterday's cash at the bank before 3 pm on 11 December.", "task", "bank deposit", date="2026-12-11", time="15:00"),
                         said("Pick up my son from Pune station at 7:30 pm on 11 December.", "appointment", "station pickup", date="2026-12-11", time="19:30")],
                  note="Order: sugar delivery 9–11 am, bank before 3 pm, station 7:30 pm."))
    t.append(task("S1-27", "ramesh", "plan", "2026-12-14", "Give me a plan for this week.", RAMESH,
                  {"must_include": [A(6380), X("KYC"), X("GST", "GSTR")], "order": [X("electricity"), X("KYC"), X("GST", "GSTR")], "must_not": [D("2026-12-26")]},
                  note="Week 14–20 Dec: electricity (14), KYC (15), GST return (20). The 26 Dec closure is next week."))
    t.append(task("S1-28", "ramesh", "update", "2026-12-12", "How much does Mrs. Deshpande still owe me?", RAMESH,
                  {"must_include": [A(1450)]},
                  extra=[said("Mrs. Deshpande paid ₹2,000 today; she still owes the rest.", "update", "part payment", date="2026-12-12", amount=2000)],
                  allow=[A(1450)], note="₹3,450 − ₹2,000 = ₹1,450."))
    t.append(task("S1-29", "ramesh", "update", "2026-12-10", "When is the sugar coming?", RAMESH,
                  {"must_include": [D("2026-12-12")], "must_not": [D("2026-12-11")]},
                  extra=[said("Correction: the sugar delivery is now on 12 December, not 11.", "update", "delivery moved", date="2026-12-12")],
                  note="The reply may repeat the old date in passing; any mention of 11 Dec still fails, so the reply must state only the new one."))
    t.append(task("S1-30", "ramesh", "unknown", "2026-12-07", "What is my shop's GST number?", RAMESH, {"abstain": True}))

    # ------------------------------------------------------------ Sharada (elderly person)
    t.append(task("S1-31", "sharada", "lookup", "2026-12-07", "When is my next doctor's visit?", SHARADA,
                  {"must_include": [D("2026-12-18"), T("11:00")]}))
    t.append(task("S1-32", "sharada", "lookup", "2026-12-07", "When do I take my BP tablet?", SHARADA,
                  {"must_include": [X("night", "bedtime")], "must_not": [X("after breakfast")]}))
    t.append(task("S1-33", "sharada", "date_window", "2026-12-14", "What do I have this week?", SHARADA,
                  {"must_include": [X("water"), D("2026-12-18"), X("bhajan")], "must_not": [D("2027-01-09")]},
                  note="Week 14–20 Dec: water bill (15), bhajan (Thu 17), doctor and fasting test (18)."))
    t.append(task("S1-34", "sharada", "semantic", "2026-12-07", "What did the doctor tell me about food?", SHARADA,
                  {"must_include": [X("sweets", "sugar"), X("salt")]}))
    t.append(task("S1-35", "sharada", "semantic", "2026-12-20", "What does Ananya want for her birthday?", SHARADA,
                  {"must_include": [X("story book", "storybook", "story books")]}))
    t.append(task("S1-36", "sharada", "plan", "2026-12-18", "Plan my day.", SHARADA,
                  {"must_include": [T("08:00"), T("11:00"), X("fasting", "empty stomach", "before breakfast", "without eating", "don't eat", "do not eat"),
                                    X("Amlodipine", "BP tablet", "blood pressure")],
                   "order": [T("08:00"), T("11:00")]},
                  note="18 Dec is a Friday: test at 8 am before breakfast, doctor at 11 am, BP tablet at night. No bhajan."))
    t.append(task("S1-37", "sharada", "update", "2026-12-17", "When is my hospital visit?", SHARADA,
                  {"must_include": [D("2026-12-21"), T("11:00")], "must_not": [D("2026-12-18")]},
                  extra=[said("The doctor's appointment is moved to 21 December, same time.", "update", "check-up moved", date="2026-12-21")]))
    t.append(task("S1-38", "sharada", "unknown", "2026-12-07", "What is my bank account number?", SHARADA, {"abstain": True}))
    t.append(task("S1-39", "sharada", "unknown", "2026-12-07", "When is my electricity bill due?", SHARADA, {"abstain": True},
                  note="Only a water bill is in memory."))
    t.append(task("S1-40", "sharada", "unknown", "2026-12-07", "Did I take my morning tablet today?", SHARADA, {"abstain": True},
                  note="Memory has the schedule, not whether she took it today."))

    # ------------------------------------------------------------ Arjun (student)
    t.append(task("S1-41", "arjun", "lookup", "2026-12-07", "When is my Operating Systems exam?", ARJUN,
                  {"must_include": [D("2026-12-11"), T("14:00")]}))
    t.append(task("S1-42", "arjun", "lookup", "2026-12-07", "When does my train leave, and from where?", ARJUN,
                  {"must_include": [D("2026-12-13"), T("21:40"), X("Katpadi")]}))
    t.append(task("S1-43", "arjun", "date_window", "2026-12-08", "What's coming up in the next 3 days?", ARJUN,
                  {"must_include": [X("DBMS"), X("project review", "Prof. Meena", "review"), X("Operating Systems", "OS")],
                   "must_not": [D("2026-12-13"), X("Data Structures")]},
                  note="Window 9–11 Dec: DBMS exam (9), project review (10), OS exam (11)."))
    t.append(task("S1-44", "arjun", "semantic", "2026-12-08", "Which topics should I study for DBMS?", ARJUN,
                  {"must_include": [X("normalisation", "normalization"), X("joins", "join"), X("transactions", "transaction")]}))
    t.append(task("S1-45", "arjun", "semantic", "2026-12-12", "Where did I save my PNR?", ARJUN,
                  {"must_include": [X("IRCTC")]}))
    t.append(task("S1-46", "arjun", "plan", "2026-12-10", "Plan my day.", ARJUN,
                  {"must_include": [T("10:00"), T("18:00"), X("Operating Systems", "OS")], "order": [T("10:00"), T("18:00")]},
                  extra=[said("Laundry pickup at the hostel at 6 pm on 10 December.", "task", "laundry pickup", date="2026-12-10", time="18:00")],
                  note="Project review 10 am, laundry 6 pm, and study for tomorrow's OS exam."))
    t.append(task("S1-47", "arjun", "plan", "2026-12-14", "Plan my week.", ARJUN,
                  {"must_include": [X("dentist"), X("interview"), A(38500)], "flag_clash": True},
                  extra=[said("Dentist in Hyderabad on 16 December at 5 pm.", "appointment", "dentist", date="2026-12-16", time="17:00"),
                         said("Online internship interview on 16 December at 5 pm.", "appointment", "internship interview", date="2026-12-16", time="17:00")],
                  note="Clash: dentist and interview both at 5 pm on 16 Dec. Mess fee due 20 Dec."))
    t.append(task("S1-48", "arjun", "update", "2026-12-16", "Which fees do I still have to pay?", ARJUN,
                  {"must_include": [A(198000)], "must_not": [A(38500)]},
                  extra=[said("I paid the mess fee today.", "update", "mess fee paid", date="2026-12-15", status="paid")],
                  note="Said on 15 Dec."))
    t.append(task("S1-49", "arjun", "unknown", "2026-12-07", "What's my CGPA?", ARJUN, {"abstain": True}))
    t.append(task("S1-50", "arjun", "unknown", "2026-12-07", "When is my Compiler Design exam?", ARJUN, {"abstain": True},
                  note="Only DS, DBMS and OS exams are in memory."))

    # ------------------------------------------------------------ Hinglish, not scored
    t.append(task("H-1", "priya", "lookup", "2026-12-07", "Mera bijli ka bill kab due hai aur kitna hai?", PRIYA,
                  {"must_include": [D("2026-12-12"), A(1860)]}, scored=False))
    t.append(task("H-2", "lakshmi", "lookup", "2026-12-07", "Divya ki school fees kitni hai, last date kya hai?", LAKSHMI,
                  {"must_include": [A(24000), D("2026-12-18")]}, scored=False))
    t.append(task("H-3", "ramesh", "date_window", "2026-12-10", "Kal kya payment karna hai?", RAMESH,
                  {"must_include": [A(195000)], "must_not": [A(6380)]}, scored=False, note="Tomorrow is 11 Dec: the sugar payment."))
    t.append(task("H-4", "sharada", "plan", "2026-12-18", "Aaj ka plan bata do.", SHARADA,
                  {"must_include": [T("08:00"), T("11:00")], "order": [T("08:00"), T("11:00")]}, scored=False))
    t.append(task("H-5", "arjun", "unknown", "2026-12-07", "Mera Compiler Design ka exam kab hai?", ARJUN, {"abstain": True}, scored=False))
    return t


def render_fact(fact: Dict[str, Any]) -> str:
    if "date" in fact:
        return date.fromisoformat(fact["date"]).strftime("%-d %b %Y") if os.name != "nt" else date.fromisoformat(fact["date"]).strftime("%#d %b %Y")
    if "amount" in fact:
        return f"₹{fact['amount']:,}"
    if "time" in fact:
        return f"time {fact['time']}"
    if "text" in fact:
        return " / ".join(f'"{p}"' for p in fact["text"])
    if "any_of" in fact:
        return f"at least {fact['min']} of: " + ", ".join(render_fact(f) for f in fact["any_of"])
    return str(fact)


def render_rules(rules: Dict[str, Any]) -> List[str]:
    lines = []
    if rules.get("must_include"):
        lines.append("Must include: " + "; ".join(render_fact(f) for f in rules["must_include"]))
    if rules.get("must_not"):
        lines.append("Must not include: " + "; ".join(render_fact(f) for f in rules["must_not"]))
    if rules.get("order"):
        lines.append("In this order: " + " → ".join(render_fact(f) for f in rules["order"]))
    if rules.get("flag_clash"):
        lines.append("Must point out the clash")
    if rules.get("abstain"):
        lines.append("Must say it isn't in memory")
    lines.append("No invented dates, amounts, day counts or weekdays (every task)")
    return lines


def build_review(tasks: List[Dict[str, Any]]) -> str:
    out = ["# S1 eval tasks (development set)", "",
           "Generated by `chirii/eval/build_s1_tasks.py`. All people and numbers are fictional. Each person's memory is listed",
           "once; tasks that add sentences show them under **Also said**.", "",
           "Approved 21 November 2026 as the **development set**: it may be run as often as needed while building. The main",
           "result comes from a separate 30-task held-out set written in S6 (see `chirii/CRITERIA.md`).", "",
           "\"No invented\" allows values worked out correctly from memory: today's date and weekday, sums of amounts in memory,",
           "day counts that match the dates (\"in 5 days\", \"2 days late\") and the right weekday next to a date. A wrong one fails.", ""]
    scored = [t for t in tasks if t["scored"]]
    counts = {k: sum(1 for t in scored if t["type"] == k) for k in TYPES}
    out += ["## Coverage", "", "| Task type | Count |", "| :--- | ---: |"]
    out += [f"| {TYPES[k]} | {v} |" for k, v in counts.items()]
    out += [f"| **Scored total** | **{len(scored)}** |", f"| Hinglish (not scored) | {len(tasks) - len(scored)} |", ""]

    for persona, label in PERSONAS.items():
        base = next(t for t in tasks if t["persona"] == persona)
        base_said = [m["said"] for m in base["memory"] if m["item"]["type"] != "update" and m in BASES[persona]]
        out += [f"## {label}", "", "**Memory (said earlier):**", ""]
        out += [f"{i}. {s}" for i, s in enumerate(base_said, 1)]
        out.append("")
        for t in (t for t in tasks if t["persona"] == persona):
            today = date.fromisoformat(t["today"])
            tag = "" if t["scored"] else " (Hinglish, not scored)"
            out.append(f"### {t['id']}: {TYPES[t['type']]}{tag}")
            out.append("")
            out.append(f"- **Today:** {today.strftime('%A')}, {render_fact({'date': t['today']})}")
            extras = [m for m in t["memory"] if m not in BASES[persona]]
            for m in extras:
                out.append(f"- **Also said:** {m['said']}")
            out.append(f"- **Asks:** {t['question']}")
            for line in render_rules(t["rules"]):
                out.append(f"- {line}")
            if t["note"]:
                out.append(f"- *Note:* {t['note']}")
            out.append("")
    return "\n".join(out)


BASES = {"priya": PRIYA, "lakshmi": LAKSHMI, "ramesh": RAMESH, "sharada": SHARADA, "arjun": ARJUN}


def main():
    tasks = build_tasks()
    with open(TASKS_PATH, "w", encoding="utf-8", newline="\n") as f:
        for item in tasks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(REVIEW_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_review(tasks) + "\n")
    print(f"Wrote {len(tasks)} tasks ({sum(t['scored'] for t in tasks)} scored)")


if __name__ == "__main__":
    main()

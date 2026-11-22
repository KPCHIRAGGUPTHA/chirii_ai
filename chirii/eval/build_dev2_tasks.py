"""
Second development set (owner, 22 Nov): 20 NEW tasks for the same 5 people, all 6 task types, with new situations
and phrasings, including ones the S4 rules were not written for ("before the weekend", "what's on my plate",
"how much do I need for payments this week", "before we leave for Madurai"). Written and committed with their
marking rules before any system was run on them. Run once on the S4b system as run (e1): a fresh first-try number.

Each person has a new memory (said earlier, on 1 December 2026, like the S1 base memories) and tasks in January
2027. All people, places and numbers are fictional. Usage:
    python -m chirii.eval.build_dev2_tasks
"""

import json
import os
from collections import Counter
from datetime import date
from typing import Any, Dict, List

from chirii.eval.build_s1_tasks import ANY, PERSONAS, TYPES, A, D, T, X, render_fact, render_rules, said, task

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_PATH = os.path.join(HERE, "dev2_tasks.jsonl")
REVIEW_PATH = os.path.join(HERE, "DEV2_TASKS.md")

PRIYA = [
    said("The Tata Play DTH recharge of ₹449 is due on 14 January.", "bill", "DTH recharge", date="2027-01-14", amount=449),
    said("My Airtel postpaid bill of ₹699 is due on 19 January.", "bill", "Airtel postpaid bill", date="2027-01-19", amount=699),
    said("Sneha's wedding is on 23 January in Coimbatore.", "important_date", "Sneha's wedding", date="2027-01-23"),
    said("I need to return the library books by 16 January.", "task", "return library books", date="2027-01-16"),
    said("Quarterly team offsite on 21 January from 9 am to 5 pm.", "appointment", "team offsite", date="2027-01-21", time="09:00", end_time="17:00"),
    said("Car service at the Hyundai centre on 16 January at 10 am.", "appointment", "car service", date="2027-01-16", time="10:00"),
    said("Note: the landlord said the rent goes up to ₹26,000 from March.", "note", "rent increase", amount=26000),
    said("Yoga class every Tuesday and Friday at 7 am.", "routine", "yoga class", weekdays=["Tuesday", "Friday"], time="07:00"),
    said("Credit card bill ₹14,250 due on 22 January.", "bill", "credit card bill", date="2027-01-22", amount=14250),
]

LAKSHMI = [
    said("Divya's board exam practical is on 2 February at 9 am.", "exam", "Divya's practical exam", date="2027-02-02", time="09:00"),
    said("Karthik's swimming class fee of ₹1,800 is due on 20 January.", "bill", "Karthik's swimming fee", date="2027-01-20", amount=1800),
    said("We are going to Madurai for my cousin's wedding from 28 to 31 January.", "travel", "Madurai trip", date="2027-01-28", end_date="2027-01-31"),
    said("The gas cylinder is booked; delivery expected on 18 January.", "note", "gas cylinder delivery", date="2027-01-18"),
    said("Murali's health insurance premium is ₹18,400, due on 26 January.", "bill", "health insurance premium", date="2027-01-26", amount=18400),
    said("Amma takes Thyronorm 50 mcg every morning before breakfast.", "medicine", "Thyronorm"),
    said("Shopping list for Sankranti: sesame, jaggery, sugarcane and new clothes for the kids.", "note", "Sankranti shopping list"),
    said("The milkman's January bill of ₹1,950 is due on 5 February.", "bill", "milkman's bill", date="2027-02-05", amount=1950),
]

RAMESH = [
    said("Sai Agencies, the cold-drinks distributor, delivers every Monday morning.", "routine", "Sai Agencies delivery", weekdays=["Monday"]),
    said("Sai Agencies bill of ₹42,500 is due on 15 January.", "bill", "Sai Agencies bill", date="2027-01-15", amount=42500),
    said("Shop licence renewal at the municipal office by 31 January.", "task", "shop licence renewal", date="2027-01-31"),
    said("Electricity bill for the shop is ₹5,940, due on 18 January.", "bill", "shop electricity bill", date="2027-01-18", amount=5940),
    said("Mr. Kulkarni owes me ₹2,700 for last month's groceries; he said he'll pay on 20 January.", "receivable", "Mr. Kulkarni's credit", date="2027-01-20", amount=2700),
    said("Stock loan EMI of ₹11,000 on the 10th of every month.", "bill", "stock loan EMI", day_of_month=10, amount=11000),
    said("Note: the freezer repair man's number is with my son.", "note", "freezer repair contact"),
]

SHARADA = [
    said("My Timolol eye drops go in twice a day, morning and evening.", "medicine", "Timolol eye drops"),
    said("Physiotherapy session at Jeevan Clinic on 13 January at 4 pm.", "appointment", "physiotherapy", date="2027-01-13", time="16:00"),
    said("My nephew Ravi is visiting from Chennai on 17 January; he arrives by the 6 am train.", "appointment", "Ravi's visit", date="2027-01-17", time="06:00"),
    said("The property tax of ₹3,200 must be paid by 31 January.", "bill", "property tax", date="2027-01-31", amount=3200),
    said("Dr. Ramaswamy said the next blood test is due in the second week of February.", "note", "next blood test"),
    said("Temple committee meeting every Sunday at 10 am.", "routine", "temple committee meeting", weekdays=["Sunday"], time="10:00"),
    said("The pension revision letter came; the new pension is ₹38,600 a month.", "note", "pension revision", amount=38600),
]

ARJUN = [
    said("Internship offer letter from Infosys: the joining date is 1 March.", "note", "Infosys internship", date="2027-03-01"),
    said("Hostel room rent of ₹14,000 for January is due on 10 January.", "bill", "hostel rent", date="2027-01-10", amount=14000),
    said("Lab record submission for Compiler Design on 19 January.", "task", "lab record submission", date="2027-01-19"),
    said("Hackathon at VIT from 22 to 23 January.", "important_date", "hackathon", date="2027-01-22", end_date="2027-01-23"),
    said("Dad sends ₹8,000 pocket money on the 1st of every month.", "income", "pocket money", day_of_month=1, amount=8000),
    said("Note from Prof. Iyer: the mid-term syllabus is units 1 to 3 only.", "note", "mid-term syllabus"),
    said("Flight to Hyderabad on 29 January at 6:15 am, IndiGo 6E-532.", "travel", "flight to Hyderabad", date="2027-01-29", time="06:15"),
]


def build_tasks() -> List[Dict[str, Any]]:
    t = []
    # Priya
    t.append(task("D2-01", "priya", "lookup", "2027-01-11", "How much is the DTH recharge and when do I have to do it?", PRIYA,
                  {"must_include": [A(449), D("2027-01-14")]}))
    t.append(task("D2-02", "priya", "date_window", "2027-01-11", "Which bills are pending before the weekend?", PRIYA,
                  {"must_include": [A(449)], "must_not": [A(699), A(14250)]},
                  note="Monday 11 to Friday 15 or Saturday 16 January: only the DTH recharge (14 Jan). Airtel is 19 Jan, the card 22 Jan."))
    t.append(task("D2-03", "priya", "plan", "2027-01-16", "What does my Saturday look like?", PRIYA,
                  {"must_include": [T("10:00"), X("library")], "must_not": [X("yoga")]},
                  note="Saturday 16 Jan: car service at 10 am and the library books are due. Yoga is on Tuesdays and Fridays."))
    t.append(task("D2-04", "priya", "unknown", "2027-01-11", "When is my gym membership renewal?", PRIYA, {"abstain": True}))
    # Lakshmi
    t.append(task("D2-05", "lakshmi", "semantic", "2027-01-10", "What do we need to get for Sankranti?", LAKSHMI,
                  {"must_include": [ANY(X("sesame"), X("jaggery"), X("sugarcane"), X("clothes"), min=3)]}))
    t.append(task("D2-06", "lakshmi", "update", "2027-01-21", "Is Karthik's swimming fee still pending?", LAKSHMI,
                  {"must_include": [X("paid", "already")], "must_not": [X("still pending", "not paid", "unpaid", "is pending", "not been paid")]},
                  extra=[said("I paid Karthik's swimming fee today.", "update", "swimming fee paid", date="2027-01-21", status="paid")],
                  note="Said on 21 Jan."))
    t.append(task("D2-07", "lakshmi", "date_window", "2027-01-25", "Anything to pay before we leave for Madurai?", LAKSHMI,
                  {"must_include": [A(18400)], "must_not": [A(1950), A(1800)]},
                  note="Leaving 28 Jan; from 25 to 27 Jan only the health insurance premium (26 Jan). The milkman is 5 Feb; the swimming fee was 20 Jan."))
    t.append(task("D2-08", "lakshmi", "lookup", "2027-01-25", "What time is Divya's practical exam?", LAKSHMI,
                  {"must_include": [T("09:00"), D("2027-02-02")]}))
    # Ramesh
    t.append(task("D2-09", "ramesh", "date_window", "2027-01-12", "How much do I need for payments this week?", RAMESH,
                  {"must_include": [A(42500)], "must_not": [A(5940), A(11000)]},
                  note="Tuesday 12 to Sunday 17 Jan: only the Sai Agencies bill (15 Jan). The electricity bill is Monday 18 Jan; the EMI was on the 10th."))
    t.append(task("D2-10", "ramesh", "update", "2027-01-21", "How much is Kulkarni still to give me?", RAMESH,
                  {"must_include": [A(1500)]},
                  extra=[said("Mr. Kulkarni paid ₹1,200 today, the rest next week.", "update", "part payment", date="2027-01-21", amount=1200)],
                  allow=[A(1500)], note="₹2,700 − ₹1,200 = ₹1,500."))
    t.append(task("D2-11", "ramesh", "plan", "2027-01-18", "What's on my plate today?", RAMESH,
                  {"must_include": [X("electricity"), A(5940), X("Sai Agencies", "delivery", "distributor")]},
                  note="Monday 18 Jan: the shop electricity bill is due and Sai Agencies delivers on Mondays."))
    t.append(task("D2-12", "ramesh", "unknown", "2027-01-12", "When does the shop insurance expire?", RAMESH, {"abstain": True}))
    # Sharada
    t.append(task("D2-13", "sharada", "semantic", "2027-01-12", "What did the doctor say about my next blood test?", SHARADA,
                  {"must_include": [X("second week of February", "February")]}))
    t.append(task("D2-14", "sharada", "lookup", "2027-01-12", "How much is my new pension?", SHARADA, {"must_include": [A(38600)]}))
    t.append(task("D2-15", "sharada", "update", "2027-01-12", "When is my physio now?", SHARADA,
                  {"must_include": [D("2027-01-15"), T("16:00")], "must_not": [D("2027-01-13")]},
                  extra=[said("The physiotherapy session has been moved to 15 January, same time.", "update", "physio moved", date="2027-01-15")]))
    t.append(task("D2-16", "sharada", "date_window", "2027-01-11", "Anything happening this weekend?", SHARADA,
                  {"must_include": [X("Ravi"), X("temple")]},
                  note="Saturday 16 and Sunday 17 Jan: Ravi arrives on Sunday by the 6 am train; temple committee on Sunday at 10 am."))
    # Arjun
    t.append(task("D2-17", "arjun", "lookup", "2027-01-18", "When is my flight home and what's the flight number?", ARJUN,
                  {"must_include": [D("2027-01-29"), T("06:15"), X("6E-532", "6E 532", "6E532")]}))
    t.append(task("D2-18", "arjun", "semantic", "2027-01-18", "What's in the mid-term syllabus?", ARJUN,
                  {"must_include": [X("units 1 to 3", "unit 1 to 3", "units 1-3", "1 to 3")]}))
    t.append(task("D2-19", "arjun", "unknown", "2027-01-18", "When is the Infosys interview?", ARJUN, {"abstain": True},
                  note="Only the Infosys offer letter (joining 1 March) is saved, not an interview."))
    t.append(task("D2-20", "arjun", "plan", "2027-01-18", "Plan my week.", ARJUN,
                  {"must_include": [X("lab record", "Compiler Design"), X("hackathon", "Hackathon")],
                   "order": [X("lab record", "Compiler Design"), X("hackathon", "Hackathon")]},
                  note="Monday 18 to Sunday 24 Jan: lab record (19 Jan), then the hackathon (22–23 Jan). The rent was due on 10 Jan."))
    return t


def build_review(tasks: List[Dict[str, Any]]) -> str:
    bases = {"priya": PRIYA, "lakshmi": LAKSHMI, "ramesh": RAMESH, "sharada": SHARADA, "arjun": ARJUN}
    counts = Counter(t["type"] for t in tasks)
    out = ["# Second development set (20 new tasks)", "",
           "Generated by `chirii/eval/build_dev2_tasks.py`, committed before any system was run on these tasks. All people and",
           "numbers are fictional. Same marker and rules as the S1 set (`chirii/eval/scoring.py`).", "",
           "| Task type | Count |", "| :--- | ---: |", *[f"| {TYPES[k]} | {counts[k]} |" for k in TYPES], f"| **Total** | **{len(tasks)}** |", ""]
    for persona, label in PERSONAS.items():
        out += [f"## {label}", "", "**Memory (said earlier):**", ""]
        out += [f"{i}. {m['said']}" for i, m in enumerate(bases[persona], 1)] + [""]
        for t in (t for t in tasks if t["persona"] == persona):
            today = date.fromisoformat(t["today"])
            out += [f"### {t['id']}: {TYPES[t['type']]}", "", f"- **Today:** {today.strftime('%A')}, {render_fact({'date': t['today']})}"]
            out += [f"- **Also said:** {m['said']}" for m in t["memory"] if m not in bases[persona]]
            out.append(f"- **Asks:** {t['question']}")
            out += [f"- {line}" for line in render_rules(t["rules"])]
            if t["note"]:
                out.append(f"- *Note:* {t['note']}")
            out.append("")
    return "\n".join(out)


def main():
    tasks = build_tasks()
    with open(TASKS_PATH, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(REVIEW_PATH, "w", encoding="utf-8") as f:
        f.write(build_review(tasks))
    print(f"Wrote {len(tasks)} tasks")


if __name__ == "__main__":
    main()

"""
Development set 5, the last development set (owner, 22 Nov): written and committed with its marking rules BEFORE S4f
(meaning-based matching and the maths validation change) was built. Run once as run (e5); after that the
development-set loop stops whatever the score (CRITERIA.md).

    dev5_memory.jsonl   20 memory tasks with deliberately varied wording: synonyms (specs/glasses, expires/runs out),
                        Hinglish words (shaadi, kiraya, dawai), indirect phrasing ("bills that hit my account")
    dev5_maths.jsonl    10 maths tasks, including routing traps ("Write ₹1,25,00,000 in crore", "weeks between")
    dev5_people.jsonl   10 people tasks

All people, places and numbers are fictional. April 2027.
    python -m chirii.eval.build_dev5_tasks
"""

import json
import os
from datetime import date

from chirii.eval.build_s1_tasks import ANY, PERSONAS, TYPES, A, D, T, X, render_fact, render_rules, said, task

HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_LABEL, MATHS_LABEL = "From your memories", "Calculated"

PRIYA = [
    said("My name is Priya; office people call me PR.", "profile", "profile"),
    said("Mausi's name is Shobha; she lives in Pune.", "person", "aunt: Shobha"),
    said("Shaadi of my colleague Neha is on 18 April in Jaipur.", "important_date", "Neha's wedding", date="2027-04-18"),
    said("Tata Play renewal of ₹399 falls on 9 April.", "bill", "Tata Play renewal", date="2027-04-09", amount=399),
    said("Doctor said to take the Vitamin D sachet every Sunday.", "medicine", "Vitamin D sachet", weekdays=["Sunday"]),
    said("Kiraya of ₹25,000 goes to the landlord on the 5th of every month.", "bill", "rent", day_of_month=5, amount=25000),
    said("Pick up the new specs from Lenskart by 14 April.", "task", "pick up specs", date="2027-04-14"),
]
LAKSHMI = [
    said("My husband Murali works at TVS.", "person", "husband: Murali"),
    said("Divya's tuition fees of ₹6,000 for the quarter are due by 10 April.", "bill", "Divya's tuition fees", date="2027-04-10", amount=6000),
    said("Appa's dawai runs out on 12 April; reorder before that.", "task", "reorder Appa's medicine", date="2027-04-12"),
    said("Karthik's cricket coaching is every Saturday at 7 am.", "routine", "Karthik's cricket coaching", weekdays=["Saturday"], time="07:00"),
    said("Navaratri golu at our home from 9 to 12 April.", "important_date", "Navaratri golu", date="2027-04-09", end_date="2027-04-12"),
    said("Our maid Selvi will be on leave from 13 to 15 April.", "note", "Selvi's leave", date="2027-04-13", end_date="2027-04-15"),
]
RAMESH = [
    said("I am Ramesh.", "profile", "profile"),
    said("My son Kunal studies in Pune.", "person", "son: Kunal"),
    said("Bank loan EMI of ₹18,200 is debited on the 7th of every month.", "bill", "bank loan EMI", day_of_month=7, amount=18200),
    said("The Jio fiber recharge of ₹999 expires on 11 April.", "bill", "Jio fiber recharge", date="2027-04-11", amount=999),
    said("Ganesh Traders owe me ₹12,000 and promised to settle by 20 April.", "receivable", "Ganesh Traders' dues", date="2027-04-20", amount=12000),
    said("My accountant Mr. Desai visits every Friday at 4 pm.", "routine", "Mr. Desai's visit", weekdays=["Friday"], time="16:00"),
]
SHARADA = [
    said("My name is Sharada.", "profile", "profile"),
    said("My BP medicine is Telma 40, one tablet after breakfast.", "medicine", "Telma 40"),
    said("Satsang at the community hall every Thursday at 6 pm.", "routine", "satsang", weekdays=["Thursday"], time="18:00"),
    said("My grandson Rohan's thread ceremony is on 22 April in Chennai.", "important_date", "Rohan's thread ceremony", date="2027-04-22"),
    said("Telephone bill of ₹450 is due on 8 April.", "bill", "telephone bill", date="2027-04-08", amount=450),
    said("Eye drops refill due on 15 April.", "task", "eye drops refill", date="2027-04-15"),
]
ARJUN = [
    said("Placement drive by TCS on 19 April at 9 am in Anna Auditorium.", "exam", "TCS placement drive", date="2027-04-19", time="09:00"),
    said("Gym membership renewal of ₹2,500 on 12 April.", "bill", "gym renewal", date="2027-04-12", amount=2500),
    said("Project submission deadline is 23 April.", "task", "project submission", date="2027-04-23"),
    said("Weekly lab every Tuesday at 2 pm.", "routine", "lab", weekdays=["Tuesday"], time="14:00"),
    said("My friend Kabir owes me ₹700 for the movie.", "receivable", "Kabir's dues", amount=700),
]
BASES = {"priya": PRIYA, "lakshmi": LAKSHMI, "ramesh": RAMESH, "sharada": SHARADA, "arjun": ARJUN}
MON = "2027-04-05"   # Monday 5 April 2027


def mtask(tid, persona, kind, question, rules, route, period=None, today=MON, note=""):
    t = task(tid, persona, kind, today, question, BASES[persona], {**rules, "no_the_user": True, "consistent": True}, note=note)
    t.update({"expect_route": route, "expect_label": MEMORY_LABEL})
    if period:
        t["period"] = list(period)
    return t


def memory_tasks():
    return [
        mtask("D5-M01", "priya", "lookup", "When is Neha's wedding?", {"must_include": [D("2027-04-18")]}, "lookup", note="Saved as 'Shaadi of my colleague Neha'."),
        mtask("D5-M02", "priya", "lookup", "Kiraya kab dena hai aur kitna?", {"must_include": [ANY(D("2027-04-05"), X("5th"), X("today")), A(25000)]}, "lookup"),
        mtask("D5-M03", "priya", "date_window", "Which subscriptions do I need to pay this week?",
              {"must_include": [A(399)], "must_not": [X("To receive", "to receive")]}, "window", period=("2027-04-05", "2027-04-11")),
        mtask("D5-M04", "priya", "lookup", "When should I collect my glasses?", {"must_include": [D("2027-04-14")]}, "lookup", note="Saved as 'specs'."),
        mtask("D5-M05", "lakshmi", "lookup", "When do I need to reorder Appa's medicine?", {"must_include": [D("2027-04-12")]}, "lookup", note="Saved as 'dawai'."),
        mtask("D5-M06", "lakshmi", "date_window", "Anything to pay before Navaratri starts?",
              {"must_include": [X("nothing", "no payments", "none", "don't have any", "do not have any", "no bills")], "must_not": [A(6000)]},
              "window", period=("2027-04-05", "2027-04-08"), note="Golu starts 9 April; the tuition fees are due 10 April."),
        mtask("D5-M07", "lakshmi", "plan", "How does Saturday look for me?", {"must_include": [T("07:00"), X("golu", "Navaratri")]}, "plan",
              period=("2027-04-10", "2027-04-10")),
        mtask("D5-M08", "lakshmi", "lookup", "By when should Divya's tuition fees be paid?", {"must_include": [D("2027-04-10"), A(6000)]}, "lookup"),
        mtask("D5-M09", "ramesh", "date_window", "Which bills hit my account this week?",
              {"must_include": [A(18200), A(999)], "must_not": [A(12000)]}, "window", period=("2027-04-05", "2027-04-11")),
        mtask("D5-M10", "ramesh", "lookup", "How much do Ganesh Traders still have to give me?", {"must_include": [A(12000)]}, "lookup"),
        mtask("D5-M11", "ramesh", "lookup", "When does the Jio plan run out?", {"must_include": [D("2027-04-11")]}, "lookup", note="Saved as 'expires'."),
        mtask("D5-M12", "ramesh", "unknown", "When does Mr. Desai's leave start?", {"abstain": True}, "lookup"),
        mtask("D5-M13", "sharada", "lookup", "Which tablet do I take in the morning?", {"must_include": [X("Telma")]}, "lookup", note="Saved as 'after breakfast'."),
        mtask("D5-M14", "sharada", "lookup", "When is Rohan's upanayanam?", {"must_include": [D("2027-04-22")]}, "lookup", note="Saved as 'thread ceremony'."),
        mtask("D5-M15", "sharada", "date_window", "Is there any bill this week?", {"must_include": [A(450)]}, "window", period=("2027-04-05", "2027-04-11")),
        mtask("D5-M16", "sharada", "lookup", "When does my eye drop bottle need refilling?", {"must_include": [D("2027-04-15")]}, "lookup"),
        mtask("D5-M17", "arjun", "lookup", "When's the TCS drive?", {"must_include": [D("2027-04-19"), T("09:00")]}, "lookup"),
        mtask("D5-M18", "arjun", "lookup", "By when do I have to submit the project?", {"must_include": [D("2027-04-23")]}, "lookup"),
        mtask("D5-M19", "arjun", "date_window", "Anything to renew this month?",
              {"must_include": [A(2500)], "must_not": [X("To receive", "to receive")]}, "window", period=("2027-04-05", "2027-04-30")),
        mtask("D5-M20", "arjun", "lookup", "Kabir se kitne paise lene hain?", {"must_include": [A(700)]}, "lookup"),
    ]


def people_tasks():
    p = lambda tid, persona, question, rules: mtask(tid, persona, "people", question, rules, "people")
    return [
        p("D5-P01", "priya", "Who is Shobha?", {"must_include": [X("aunt", "Mausi", "mother's sister")]}),
        p("D5-P02", "priya", "What do people at office call me?", {"must_include": [X("PR")]}),
        p("D5-P03", "lakshmi", "Who is Murali?", {"must_include": [X("husband")]}),
        p("D5-P04", "lakshmi", "Selvi kaun hai?", {"must_include": [X("maid")]}),
        p("D5-P05", "ramesh", "Who is Mr. Desai?", {"must_include": [X("accountant")]}),
        p("D5-P06", "ramesh", "Who is Kunal's father?", {"must_include": [X("you"), X("Ramesh")]}),
        p("D5-P07", "sharada", "Who is Rohan?", {"must_include": [X("grandson")]}),
        p("D5-P08", "sharada", "Who is Rohan's grandmother?", {"must_include": [X("you"), X("Sharada")]}),
        p("D5-P09", "arjun", "Who is Kabir?", {"must_include": [X("friend")]}),
        p("D5-P10", "arjun", "Is Kabir my cousin?", {"must_include": [X("friend")], "must_not": [X("Kabir is your cousin", "Yes, Kabir")]}),
    ]


def maths(tid, question, number, tol=0.01, note=""):
    return {"id": tid, "kind": "maths", "question": question, "rules": {"number": number, "tolerance": tol, "working": True},
            "expect_route": "maths", "expect_label": MATHS_LABEL, "persona": None, "today": MON, "memory": [], "note": note}


def maths_tasks():
    return [
        maths("D5-F01", "Write ₹1,25,00,000 in crore.", 1.25),
        maths("D5-F02", "How many weeks are there between 1 March and 26 April 2027?", 8),
        maths("D5-F03", "What is 7.5% of ₹8,40,000?", 63000),
        maths("D5-F04", "What is the compound interest on ₹50,000 at 10% a year for 3 years, compounded yearly?", 16550),
        maths("D5-F05", "What is the EMI on a ₹2 lakh loan at 12% a year for 2 years?", 9414.69, tol=1),
        maths("D5-F06", "Convert 30 km to miles.", 18.64, tol=0.01),
        maths("D5-F07", "A ₹1,200 bill plus 18% GST: what is the total?", 1416),
        maths("D5-F08", "Split ₹2,700 equally among me and 4 friends.", 540, note="5 people"),
        maths("D5-F09", "What is 38.5°C in Fahrenheit?", 101.3),
        maths("D5-F10", "What's the final price of ₹3,600 after a 35% discount?", 2340),
    ]


def review(groups):
    out = ["# Development set 5 (the last; written before S4f)", "", "Generated by `chirii/eval/build_dev5_tasks.py`; committed before S4f.", ""]
    for persona, label in PERSONAS.items():
        out += [f"## {label}", "", "**Memory (said earlier):**", ""] + [f"{i}. {m['said']}" for i, m in enumerate(BASES[persona], 1)] + [""]
        for t in (t for t in groups["memory"] + groups["people"] if t["persona"] == persona):
            out += [f"### {t['id']}: {TYPES.get(t['type'], t['type'])} (route: {t['expect_route']})", "",
                    f"- **Today:** {date.fromisoformat(t['today']).strftime('%A')}, {render_fact({'date': t['today']})}", f"- **Asks:** {t['question']}"]
            out += [f"- {line}" for line in render_rules({k: v for k, v in t["rules"].items() if k not in ("consistent", "no_the_user")})]
            out += ([f"- *Note:* {t['note']}"] if t["note"] else []) + [""]
    out += ["## Maths", "", "| Task | Question | Rules |", "| :--- | :--- | :--- |"]
    out += [f"| {t['id']} | {t['question']} | {json.dumps(t['rules'], ensure_ascii=False)} |" for t in groups["maths"]]
    return "\n".join(out) + "\n"


def main():
    groups = {"memory": memory_tasks(), "people": people_tasks(), "maths": maths_tasks()}
    for name, rows in groups.items():
        with open(os.path.join(HERE, f"dev5_{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "DEV5_TASKS.md"), "w", encoding="utf-8") as f:
        f.write(review(groups))
    print({k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()

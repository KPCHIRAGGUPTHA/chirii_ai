"""
Development set 4 (owner, 22 Nov), written and committed with its marking rules BEFORE S4d (fixes, maths formula
library, glossary) and S4e (people and profile memory) were built. Run once afterwards as run (e4).

    dev4_memory.jsonl   15 memory tasks (3 per person; new memories, March 2027, new phrasings)
    dev4_people.jsonl   15 people/profile tasks (names, nicknames, relationships incl. reverse ones, Indian terms such
                        as Amma, Appa, Didi, Bhaiya, Thatha, Dadi; the person's own name in the third person; a
                        relationship whose gender is unknown)
    dev4_maths.jsonl    15 maths tasks: formula-library cases (percentage, GST, discount, simple and compound
                        interest, EMI, split bills, days and weeks between dates, unit conversion, lakh and crore)
    dev4_gk.jsonl       10 general-knowledge tasks, 5 of them Indian everyday terms

Memory and people tasks are marked by scoring.py, maths and general knowledge by chat_scoring.py. Each task records
the route and the answer label it should get. All people, places and numbers are fictional.
    python -m chirii.eval.build_dev4_tasks
"""

import json
import os
from collections import Counter
from datetime import date

from chirii.eval.build_s1_tasks import PERSONAS, TYPES, A, D, T, X, render_fact, render_rules, said, task

HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_LABEL, MATHS_LABEL, GENERAL_LABEL = "From your memories", "Calculated", "General answer"

PRIYA = [
    said("My name is Priya Raman, but friends call me Pri.", "profile", "profile"),
    said("Amma's name is Lalitha.", "person", "mother: Lalitha"),
    said("Karan is my younger brother.", "person", "brother: Karan"),
    said("My son Aarav turns 6 on 14 April.", "person", "son: Aarav", date="2027-04-14"),
    said("Amma's eye check-up at Sankara is on 9 March at 10 am.", "appointment", "Amma's eye check-up", date="2027-03-09", time="10:00"),
    said("Car loan EMI of ₹14,800 goes out on the 10th of every month.", "bill", "car loan EMI", day_of_month=10, amount=14800),
    said("Return the Amazon parcel by 12 March.", "task", "return Amazon parcel", date="2027-03-12"),
    said("Netflix renews on 20 March for ₹649.", "bill", "Netflix", date="2027-03-20", amount=649),
    said("Hotstar renews on 20 March for ₹299.", "bill", "Hotstar", date="2027-03-20", amount=299),
]
RAMESH = [
    said("I am Ramesh Patil.", "profile", "profile"),
    said("My wife's name is Sunita.", "person", "wife: Sunita"),
    said("Bhaiya, my elder brother Mahesh, lives in Nashik.", "person", "elder brother: Mahesh"),
    said("Dadi's 80th birthday is on 18 March.", "person", "Dadi's birthday", date="2027-03-18"),
    said("Sunita's cataract surgery is on 16 March at 8 am.", "appointment", "Sunita's surgery", date="2027-03-16", time="08:00"),
    said("GSTR-1 for February must be filed by 11 March.", "task", "GSTR-1 filing", date="2027-03-11"),
    said("Patel Traders owes me ₹6,500 and promised to pay by 19 March.", "receivable", "Patel Traders' dues", date="2027-03-19", amount=6500),
    said("Shop rent of ₹23,500 on the 5th of every month.", "bill", "shop rent", day_of_month=5, amount=23500),
]
LAKSHMI = [
    said("My name is Lakshmi; at home they call me Lachu.", "profile", "profile"),
    said("Appa stays with us in Chennai.", "person", "father (Appa)"),
    said("My daughter Divya is in Class 10.", "person", "daughter: Divya"),
    said("Didi's name is Revathi; she lives in Bengaluru.", "person", "elder sister: Revathi"),
    said("Thatha's death anniversary puja is on 22 March.", "important_date", "Thatha's anniversary puja", date="2027-03-22"),
    said("Divya's board exams run from 15 to 30 March.", "exam", "Divya's board exams", date="2027-03-15", end_date="2027-03-30"),
    said("Revathi's son Kiran is getting married on 2 April.", "important_date", "Kiran's wedding", date="2027-04-02"),
    said("Electricity bill of ₹2,340 is due on 18 March.", "bill", "electricity bill", date="2027-03-18", amount=2340),
]
SHARADA = [
    said("My name is Sharada Iyer.", "profile", "profile"),
    said("My son Venkat lives in Pune.", "person", "son: Venkat"),
    said("Venkat's wife is Kavya.", "person", "Venkat's wife: Kavya"),
    said("My granddaughter Ananya turns 10 on 25 March.", "person", "granddaughter: Ananya", date="2027-03-25"),
    said("My brother Gopal is visiting on 12 March.", "important_date", "Gopal's visit", date="2027-03-12"),
    said("Dentist Dr. Shetty on 11 March at 4 pm.", "appointment", "dentist", date="2027-03-11", time="16:00"),
    said("Water bill of ₹220 is due on 15 March.", "bill", "water bill", date="2027-03-15", amount=220),
]
ARJUN = [
    said("I'm Arjun Reddy; everyone calls me Ajju.", "profile", "profile"),
    said("My sister Meghna is getting engaged on 27 March.", "important_date", "Meghna's engagement", date="2027-03-27"),
    said("Papa's name is Srinivas.", "person", "father: Srinivas"),
    said("My roommate Faiz owes me ₹1,200 for the trip.", "receivable", "Faiz's dues", amount=1200),
    said("Mid-sem exams are from 22 to 26 March.", "exam", "mid-sem exams", date="2027-03-22", end_date="2027-03-26"),
    said("Hostel fee of ₹38,500 is due on 31 March.", "bill", "hostel fee", date="2027-03-31", amount=38500),
    said("My cousin Tara is my age; we were born a week apart.", "person", "cousin: Tara"),
]
BASES = {"priya": PRIYA, "lakshmi": LAKSHMI, "ramesh": RAMESH, "sharada": SHARADA, "arjun": ARJUN}
CONSISTENT = {"consistent": True, "no_the_user": True}


def mtask(tid, persona, kind, today, question, rules, route, period=None, extra=(), allow=(), note="", label=MEMORY_LABEL):
    t = task(tid, persona, kind, today, question, BASES[persona], {**rules, "no_the_user": True}, extra=extra, allow=allow, note=note)
    t.update({"expect_route": route, "expect_label": label})
    if period:
        t["period"] = list(period)
    return t


def memory_tasks():
    return [
        mtask("D4-M01", "priya", "lookup", "2027-03-08", "When is Amma's eye check-up?", {"must_include": [D("2027-03-09"), T("10:00")]}, "lookup"),
        mtask("D4-M02", "priya", "date_window", "2027-03-08", "What renews on 20 March?", {"must_include": [A(649), A(299)], "must_not": [A(14800)]},
              "window", period=("2027-03-20", "2027-03-20")),
        mtask("D4-M03", "priya", "update", "2027-03-12", "Did I send the Amazon parcel back?",
              {"must_include": [X("yes", "sent", "returned")], "must_not": [X("no record", "don't have a record", "not returned", "haven't")]}, "status",
              extra=[said("Sent back the Amazon parcel today.", "update", "parcel returned", date="2027-03-11", status="done")], note="Said on 11 March."),
        mtask("D4-M04", "ramesh", "lookup", "2027-03-08", "By when does the GSTR-1 have to be filed?", {"must_include": [D("2027-03-11")]}, "lookup"),
        mtask("D4-M05", "ramesh", "date_window", "2027-03-15", "Anything due between 16 and 19 March?",
              {"must_include": [X("Sunita", "surgery"), X("Patel")]}, "window", period=("2027-03-16", "2027-03-19")),
        mtask("D4-M06", "ramesh", "update", "2027-03-20", "Does Patel still owe me anything?",
              {"must_include": [X("no", "cleared", "paid")], "must_not": [X("still owes", "owes you ₹6,500")]}, "status",
              extra=[said("Patel Traders cleared their dues today.", "update", "Patel cleared", date="2027-03-19", status="paid")], note="Said on 19 March."),
        mtask("D4-M07", "lakshmi", "date_window", "2027-03-17", "What do I have this week?",
              {"must_include": [A(2340), X("exam", "exams", "board")], "must_not": [X("puja", "Thatha")]}, "window", period=("2027-03-17", "2027-03-21"),
              note="Wednesday 17 to Sunday 21 March: electricity bill (18th) and Divya's exams (running). The puja is Monday 22nd."),
        mtask("D4-M08", "lakshmi", "lookup", "2027-03-10", "When do Divya's exams end?", {"must_include": [D("2027-03-30")]}, "lookup"),
        mtask("D4-M09", "lakshmi", "update", "2027-03-20", "When is Kiran's wedding now?", {"must_include": [D("2027-04-04")], "must_not": [D("2027-04-02")]}, "lookup",
              extra=[said("Kiran's wedding is moved to 4 to 5 April.", "update", "wedding moved", date="2027-04-04", end_date="2027-04-05")]),
        mtask("D4-M10", "sharada", "lookup", "2027-03-08", "When is Gopal coming?", {"must_include": [D("2027-03-12")]}, "lookup"),
        mtask("D4-M11", "sharada", "date_window", "2027-03-08", "Do I need to pay anything before Gopal comes?",
              {"must_include": [X("nothing", "no payments", "none", "don't have any", "do not have any", "no bills")], "must_not": [A(220)]},
              "window", period=("2027-03-08", "2027-03-11")),
        mtask("D4-M12", "sharada", "unknown", "2027-03-08", "What time is Gopal's train?", {"abstain": True}, "lookup"),
        mtask("D4-M13", "arjun", "date_window", "2027-03-22", "Do I have to pay anything before the engagement?",
              {"must_include": [X("nothing", "no payments", "none", "don't have any", "do not have any", "no bills")], "must_not": [A(38500)]},
              "window", period=("2027-03-22", "2027-03-26"), note="Engagement 27 March; the hostel fee is 31 March."),
        mtask("D4-M14", "arjun", "lookup", "2027-03-15", "How much does Faiz owe me?", {"must_include": [A(1200)]}, "lookup"),
        mtask("D4-M15", "arjun", "lookup", "2027-03-15", "When do my mid-sems end?", {"must_include": [D("2027-03-26")]}, "lookup"),
    ]


def people_tasks():
    p = lambda tid, persona, question, rules, route="people", note="": mtask(tid, persona, "people", "2027-03-08", question, {**rules, **CONSISTENT}, route, note=note)
    return [
        p("D4-P01", "priya", "What is my mother's name?", {"must_include": [X("Lalitha")]}),
        p("D4-P02", "priya", "Who are Lalitha's children?", {"must_include": [X("Karan"), X("you", "Priya")]},
          note="Lalitha is the user's mother and Karan is the user's brother, so both are her children."),
        p("D4-P03", "priya", "When is Aarav's birthday?", {"must_include": [D("2027-04-14")]}),
        p("D4-P04", "priya", "What do my friends call me?", {"must_include": [X("Pri")]}),
        p("D4-P05", "ramesh", "What is Sunita's husband's name?", {"must_include": [X("Ramesh"), X("you")]},
          note="Reverse relationship: Sunita is the user's wife, so her husband is the user."),
        p("D4-P06", "ramesh", "Where does Bhaiya live?", {"must_include": [X("Nashik")]}),
        p("D4-P07", "ramesh", "When is Dadi's birthday?", {"must_include": [D("2027-03-18")]}),
        p("D4-P08", "lakshmi", "What is Didi's name?", {"must_include": [X("Revathi")]}),
        p("D4-P09", "lakshmi", "Who is Kiran?", {"must_include": [X("Revathi's son", "nephew", "Didi's son", "sister's son")]}),
        p("D4-P10", "lakshmi", "Who is Divya's grandfather?", {"must_include": [X("Appa", "your father")]},
          note="Divya is the user's daughter and Appa the user's father."),
        p("D4-P11", "sharada", "Who is Kavya?", {"must_include": [X("Venkat's wife", "daughter-in-law", "son's wife")]}),
        p("D4-P12", "sharada", "Who is Ananya's grandmother?", {"must_include": [X("you"), X("Sharada")]}),
        p("D4-P13", "sharada", "When is Sharada's dentist appointment?", {"must_include": [D("2027-03-11"), T("16:00")]}, route="lookup",
          note="The user's own name in the third person."),
        p("D4-P14", "arjun", "Who are Srinivas's children?", {"must_include": [X("you", "Arjun"), X("Meghna")]}),
        p("D4-P15", "arjun", "Is Tara my brother or my sister?", {"must_include": [X("cousin")],
          "must_not": [X("Tara is your brother", "Tara is your sister", "she is your", "he is your")]},
          note="Gender unknown: Tara is a cousin; Chirii must not assume."),
    ]


def maths(tid, question, number, tol=0.01, note=""):
    return {"id": tid, "kind": "maths", "question": question, "rules": {"number": number, "tolerance": tol, "working": True},
            "expect_route": "maths", "expect_label": MATHS_LABEL, "persona": None, "today": "2027-03-08", "memory": [], "note": note}


def maths_tasks():
    return [
        maths("D4-F01", "What is 18% GST on ₹4,250?", 765),
        maths("D4-F02", "A shirt costs ₹2,499 with 20% off. What do I pay?", 1999.2),
        maths("D4-F03", "What is the simple interest on ₹2,00,000 at 6.5% a year for 4 years?", 52000),
        maths("D4-F04", "What is the compound interest on ₹1,00,000 at 8% a year for 2 years, compounded yearly?", 16640),
        maths("D4-F05", "What is the EMI for a ₹5 lakh loan at 9% a year for 5 years?", 10379.18, tol=1, note="Standard reducing-balance EMI."),
        maths("D4-F06", "Split ₹1,240, ₹860 and ₹499 equally among 3 people.", 866.33, note="₹2,599 / 3"),
        maths("D4-F07", "How many days are there from 10 February to 10 March 2027?", 28),
        maths("D4-F08", "How many weeks are there from 1 January to 26 February 2027?", 8),
        maths("D4-F09", "Convert 5 miles to kilometres.", 8.05, tol=0.01),
        maths("D4-F10", "What is 1.2 crore in lakh?", 120),
        maths("D4-F11", "Write ₹45,00,000 in lakh.", 45),
        maths("D4-F12", "Milk went from ₹54 to ₹56 a litre. By what percent did the price rise?", 3.7, tol=0.01),
        maths("D4-F13", "What is the price of ₹1,000 plus 18% GST?", 1180),
        maths("D4-F14", "1,240 + 860 + 499", 2599),
        maths("D4-F15", "Convert 100°F to Celsius.", 37.78, tol=0.01),
    ]


def gk(tid, question, rules):
    return {"id": tid, "kind": "general", "question": question, "rules": rules, "expect_route": "general", "expect_label": GENERAL_LABEL,
            "persona": None, "today": "2027-03-08", "memory": [], "note": ""}


def gk_tasks():
    return [
        gk("D4-K01", "What is IFSC?", {"must_include": [X("Indian Financial System Code")]}),
        gk("D4-K02", "What does TDS mean?", {"must_include": [X("Tax Deducted at Source")]}),
        gk("D4-K03", "What is a CIBIL score?", {"must_include": [X("credit score", "credit")]}),
        gk("D4-K04", "What is EPF?", {"must_include": [X("Employees' Provident Fund", "Employees Provident Fund", "Employee Provident Fund", "provident fund")]}),
        gk("D4-K05", "What is the difference between NEFT and RTGS?", {"must_include": [X("Real Time Gross Settlement", "real-time", "real time")]}),
        gk("D4-K06", "Which is the longest river in India?", {"must_include": [X("Ganga", "Ganges")]}),
        gk("D4-K07", "Who is known as the Missile Man of India?", {"must_include": [X("Kalam")]}),
        gk("D4-K08", "What is the national animal of India?", {"must_include": [X("tiger", "Tiger")]}),
        gk("D4-K09", "How many players does a cricket team have on the field?", {"must_include": [X("11", "eleven")]}),
        gk("D4-K10", "What is the boiling point of water in Fahrenheit?", {"must_include": [X("212")]}),
    ]


def review(groups):
    out = ["# Development set 4 (written before S4d and S4e)", "", "Generated by `chirii/eval/build_dev4_tasks.py`; committed before S4d and S4e were built.",
           "Answer labels expected: \"From your memories\", \"Calculated\", \"General answer\".", ""]
    for persona, label in PERSONAS.items():
        out += [f"## {label}", "", "**Memory (said earlier):**", ""] + [f"{i}. {m['said']}" for i, m in enumerate(BASES[persona], 1)] + [""]
        for t in (t for t in groups["memory"] + groups["people"] if t["persona"] == persona):
            today = date.fromisoformat(t["today"])
            out += [f"### {t['id']}: {TYPES.get(t['type'], t['type'])} (route: {t['expect_route']})", "",
                    f"- **Today:** {today.strftime('%A')}, {render_fact({'date': t['today']})}"]
            out += [f"- **Also said:** {m['said']}" for m in t["memory"] if m not in BASES[persona]]
            out.append(f"- **Asks:** {t['question']}")
            out += [f"- {line}" for line in render_rules({k: v for k, v in t["rules"].items() if k not in ("consistent", "no_the_user")})]
            if t["rules"].get("consistent"):
                out.append("- Must not both say it isn't saved and give the answer; never says \"the user\"")
            if t["note"]:
                out.append(f"- *Note:* {t['note']}")
            out.append("")
    for name in ("maths", "gk"):
        out += [f"## {'Maths' if name == 'maths' else 'General knowledge'}", "", "| Task | Question | Rules |", "| :--- | :--- | :--- |"]
        out += [f"| {t['id']} | {t['question']} | {json.dumps(t['rules'], ensure_ascii=False)} |" for t in groups[name]] + [""]
    return "\n".join(out) + "\n"


def main():
    groups = {"memory": memory_tasks(), "people": people_tasks(), "maths": maths_tasks(), "gk": gk_tasks()}
    for name, rows in groups.items():
        with open(os.path.join(HERE, f"dev4_{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "DEV4_TASKS.md"), "w", encoding="utf-8") as f:
        f.write(review(groups))
    print({k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()

"""
Development set 3 (owner, 22 Nov), written and committed with its marking rules BEFORE S4c (the question-reading
redesign and general chat) was built. Run once on the S4c system as run (e3): a fresh first-try number.

    dev3_memory.jsonl  30 memory tasks: same 5 people, new memories (February 2027), new phrasings: weekends, weekday
                       names, "end of the month", "next month", periods relative to a saved event, partial matches
                       ("Infosys" saved but no "interview"), Hinglish-lite. Marked by chirii/eval/scoring.py.
    dev3_chat.jsonl    40 general-chat tasks: 15 maths (10 pure, 5 mixed with memory), 15 general knowledge (10 stable
                       facts, 3 needing live information, 2 nobody could know), 10 writing/advice. Marked by
                       chirii/eval/chat_scoring.py.
    dev3_router.jsonl  15 router tests: the route (and period) each question should get.

Every task also records the route it should take ("expect_route"), so router accuracy is measured on all 85 tasks as
well as on the 15 router tests. A task whose question names a period records that period ("period"), which the
marker treats as worked-out dates (recorded in CRITERIA.md before the run).

All people, places and numbers are fictional. Usage:
    python -m chirii.eval.build_dev3_tasks
"""

import json
import os
from collections import Counter
from datetime import date
from typing import Any, Dict, List

from chirii.eval.build_s1_tasks import ANY, PERSONAS, TYPES, A, D, T, X, render_fact, render_rules, said, task

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = ["lookup", "status", "window", "plan", "meaning", "people", "maths", "general", "writing", "mixed"]

PRIYA = [
    said("Water purifier AMC of ₹2,100 is due on 9 February.", "bill", "water purifier AMC", date="2027-02-09", amount=2100),
    said("Mom's cataract follow-up with Dr. Menon on 13 February at 11 am.", "appointment", "mom's cataract follow-up", date="2027-02-13", time="11:00"),
    said("Office annual day on 19 February at 6 pm.", "appointment", "office annual day", date="2027-02-19", time="18:00"),
    said("Trip to Goa with college friends from 26 to 28 February.", "travel", "Goa trip", date="2027-02-26", end_date="2027-02-28"),
    said("Pay the house help ₹7,500 on the 1st of every month.", "bill", "house help", day_of_month=1, amount=7500),
    said("Passport appointment at the PSK on 4 March at 9:30 am.", "appointment", "passport appointment", date="2027-03-04", time="09:30"),
    said("Note: the Goa hotel booking ID is GOA-4471, and I paid ₹9,800 in advance.", "note", "Goa hotel booking", amount=9800),
    said("Netflix ₹649 renews on the 15th of every month.", "bill", "Netflix", day_of_month=15, amount=649),
]
LAKSHMI = [
    said("Karthik's school trip to Mysuru is on 12 February.", "important_date", "Karthik's school trip", date="2027-02-12"),
    said("The school trip fee of ₹1,200 must be paid by 8 February.", "bill", "school trip fee", date="2027-02-08", amount=1200),
    said("Divya's tuition is now every Saturday at 8 am.", "routine", "Divya's tuition", weekdays=["Saturday"], time="08:00"),
    said("Murali's birthday is on 20 February.", "important_date", "Murali's birthday", date="2027-02-20"),
    said("The plumber is coming on 19 February at 3 pm to fix the bathroom tap.", "appointment", "plumber", date="2027-02-19", time="15:00"),
    said("Annual property tax of ₹4,600 is due on 28 February.", "bill", "property tax", date="2027-02-28", amount=4600),
    said("Amma's knee scan at Apollo on 16 February at 7:30 am; no food after midnight.", "appointment", "Amma's knee scan", date="2027-02-16", time="07:30"),
]
RAMESH = [
    said("Wholesale market trip every Thursday at 5 am.", "routine", "wholesale market trip", weekdays=["Thursday"], time="05:00"),
    said("Shop rent of ₹23,000 to Mr. Joshi on the 5th of every month.", "bill", "shop rent", day_of_month=5, amount=23000),
    said("Sharma ji owes me ₹4,800 for the wedding order and promised to pay by 15 February.", "receivable", "Sharma ji's dues", date="2027-02-15", amount=4800),
    said("My son's college fee of ₹65,000 is due on 25 February.", "bill", "son's college fee", date="2027-02-25", amount=65000),
    said("The Holi stock order from Gupta Traders arrives on 24 February; I pay ₹30,000 on delivery.", "bill", "Holi stock order", date="2027-02-24", amount=30000),
    said("Note: the GST portal password was reset, and the new login is with the CA.", "note", "GST portal login"),
    said("The shop is closed on 21 February for my cousin's wedding in Nashik.", "important_date", "shop closed", date="2027-02-21"),
]
SHARADA = [
    said("Book club meets every Saturday at 4 pm at the library.", "routine", "book club", weekdays=["Saturday"], time="16:00"),
    said("Ananya's school play is on 20 February at 5 pm.", "appointment", "Ananya's school play", date="2027-02-20", time="17:00"),
    said("Dentist Dr. Shetty on 9 February at 10 am for a filling.", "appointment", "dentist, Dr. Shetty", date="2027-02-09", time="10:00"),
    said("Electricity bill of ₹640 is due on 12 February.", "bill", "electricity bill", date="2027-02-12", amount=640),
    said("Vitamin B12 tablet once a week, every Sunday.", "medicine", "Vitamin B12", weekdays=["Sunday"]),
    said("My sister Kamala arrives from Pune on 22 February.", "important_date", "Kamala arrives", date="2027-02-22"),
    said("Note: the pension office said the life certificate is due by 30 November every year.", "note", "life certificate rule", date="2026-11-30"),
]
ARJUN = [
    said("Placement aptitude test on 10 February at 2 pm in SJT 301.", "exam", "aptitude test", date="2027-02-10", time="14:00"),
    said("Basketball practice every Monday and Wednesday at 6 pm.", "routine", "basketball practice", weekdays=["Monday", "Wednesday"], time="18:00"),
    said("My sister's engagement in Hyderabad is on 27 February.", "important_date", "sister's engagement", date="2027-02-27"),
    said("Mess bill for February of ₹4,200 is due on 20 February.", "bill", "mess bill", date="2027-02-20", amount=4200),
    said("Return the library book 'Operating System Concepts' by 12 February.", "task", "return library book", date="2027-02-12"),
    said("Bus to Hyderabad on 25 February at 9 pm from Katpadi.", "travel", "bus to Hyderabad", date="2027-02-25", time="21:00"),
    said("Note from the placement cell: carry two printed resumes and the college ID.", "note", "placement cell note"),
]
BASES = {"priya": PRIYA, "lakshmi": LAKSHMI, "ramesh": RAMESH, "sharada": SHARADA, "arjun": ARJUN}


def mtask(tid, persona, kind, today, question, rules, route, period=None, extra=(), allow=(), note=""):
    t = task(tid, persona, kind, today, question, BASES[persona], rules, extra=extra, allow=allow, note=note)
    t["expect_route"] = route
    if period:
        t["period"] = list(period)
    return t


def memory_tasks() -> List[Dict[str, Any]]:
    t = []
    # Priya
    t.append(mtask("D3-01", "priya", "lookup", "2027-02-08", "When is mom's follow-up?", {"must_include": [D("2027-02-13"), T("11:00")]}, "lookup"))
    t.append(mtask("D3-02", "priya", "date_window", "2027-02-11", "What's on this weekend?",
                   {"must_include": [X("follow-up", "Menon", "cataract"), T("11:00")], "must_not": [A(649), X("annual day")]},
                   "window", period=("2027-02-13", "2027-02-14"), note="Saturday 13 and Sunday 14 Feb: mom's follow-up on Saturday at 11 am."))
    t.append(mtask("D3-03", "priya", "date_window", "2027-02-15", "What do I need to pay before the Goa trip?",
                   {"must_include": [A(649)], "must_not": [A(7500), A(2100)]}, "window", period=("2027-02-15", "2027-02-25"),
                   note="Trip starts 26 Feb. From 15 to 25 Feb only Netflix (15th). The house help is on 1 March; the AMC was 9 Feb."))
    t.append(mtask("D3-04", "priya", "plan", "2027-02-17", "What does my Friday look like?",
                   {"must_include": [T("18:00"), X("annual day")]}, "plan", period=("2027-02-19", "2027-02-19")))
    t.append(mtask("D3-05", "priya", "update", "2027-02-10", "When are we going to Goa?",
                   {"must_include": [D("2027-03-05")], "must_not": [D("2027-02-26")]}, "lookup",
                   extra=[said("The Goa trip is postponed to 5 to 7 March.", "update", "Goa trip moved", date="2027-03-05", end_date="2027-03-07")]))
    t.append(mtask("D3-06", "priya", "unknown", "2027-02-08", "What's the Goa flight time?", {"abstain": True}, "lookup",
                   note="Partial match: the Goa trip and hotel are saved, but no flight."))
    # Lakshmi
    t.append(mtask("D3-07", "lakshmi", "lookup", "2027-02-10", "Amma ka scan kab hai?", {"must_include": [D("2027-02-16"), T("07:30")]}, "lookup"))
    t.append(mtask("D3-08", "lakshmi", "semantic", "2027-02-10", "Anything special to remember for Amma's scan?",
                   {"must_include": [X("no food", "midnight", "empty stomach", "fasting", "not eat")]}, "meaning"))
    t.append(mtask("D3-09", "lakshmi", "date_window", "2027-02-22", "What's due before the end of the month?",
                   {"must_include": [A(4600)], "must_not": [A(1200)]}, "window", period=("2027-02-22", "2027-02-28")))
    t.append(mtask("D3-10", "lakshmi", "plan", "2027-02-18", "What's my Friday like?",
                   {"must_include": [T("15:00"), X("plumber")], "must_not": [X("birthday")]}, "plan", period=("2027-02-19", "2027-02-19"),
                   note="Friday 19 Feb: the plumber at 3 pm. Murali's birthday is Saturday."))
    t.append(mtask("D3-11", "lakshmi", "update", "2027-02-06", "Do I still owe anything for Karthik's trip?",
                   {"must_include": [X("paid", "nothing", "no,", "don't owe", "do not owe")],
                    "must_not": [X("still owe", "need to pay", "not paid", "is due", "unpaid")]}, "status",
                   extra=[said("I paid the school trip fee today.", "update", "trip fee paid", date="2027-02-05", status="paid")], note="Said on 5 Feb."))
    t.append(mtask("D3-12", "lakshmi", "unknown", "2027-02-10", "When is Divya's annual day?", {"abstain": True}, "lookup",
                   note="Partial match: Divya's tuition is saved, but no annual day."))
    # Ramesh
    t.append(mtask("D3-13", "ramesh", "date_window", "2027-02-10", "What payments do I have next month?",
                   {"must_include": [A(23000)], "must_not": [A(65000), A(30000)]}, "window", period=("2027-03-01", "2027-03-31")))
    t.append(mtask("D3-14", "ramesh", "date_window", "2027-02-16", "Do I need to pay anything before the Holi stock arrives?",
                   {"must_include": [X("nothing", "no payments", "none", "don't have any", "do not have any", "no bills")],
                    "must_not": [A(65000), A(23000)]}, "window", period=("2027-02-16", "2027-02-23"),
                   note="The stock arrives on 24 Feb. From 16 to 23 Feb nothing is due (the college fee is 25 Feb)."))
    t.append(mtask("D3-15", "ramesh", "update", "2027-02-16", "Has Sharma ji paid me?",
                   {"must_include": [X("yes", "paid")], "must_not": [X("not paid", "still owes", "hasn't paid")]}, "status",
                   extra=[said("Sharma ji paid the full ₹4,800 today.", "update", "Sharma ji paid", date="2027-02-16", amount=4800)]))
    t.append(mtask("D3-16", "ramesh", "lookup", "2027-02-10", "Sharma ji se kitna lena hai?", {"must_include": [A(4800)]}, "lookup"))
    t.append(mtask("D3-17", "ramesh", "plan", "2027-02-17", "What's my Thursday like?",
                   {"must_include": [T("05:00"), X("market", "wholesale")]}, "plan", period=("2027-02-18", "2027-02-18")))
    t.append(mtask("D3-18", "ramesh", "unknown", "2027-02-10", "When is the GST filing due?", {"abstain": True}, "lookup",
                   note="Partial match: a GST portal note is saved, but no filing date."))
    # Sharada
    t.append(mtask("D3-19", "sharada", "lookup", "2027-02-08", "When is the dentist?", {"must_include": [D("2027-02-09"), T("10:00")]}, "lookup"))
    t.append(mtask("D3-20", "sharada", "date_window", "2027-02-18", "Anything this weekend?",
                   {"must_include": [X("play", "Ananya"), X("book club")]}, "window", period=("2027-02-20", "2027-02-21")))
    t.append(mtask("D3-21", "sharada", "date_window", "2027-02-15", "What do I have before Kamala comes?",
                   {"must_include": [X("play", "Ananya")], "must_not": [X("dentist", "Shetty")]}, "window", period=("2027-02-15", "2027-02-21"),
                   note="Kamala arrives 22 Feb. The dentist was on 9 Feb."))
    t.append(mtask("D3-22", "sharada", "semantic", "2027-02-08", "What did the pension office say?",
                   {"must_include": [X("life certificate"), X("30 November", "30th November", "November 30")]}, "meaning"))
    t.append(mtask("D3-23", "sharada", "update", "2027-02-08", "When's my filling now?",
                   {"must_include": [D("2027-02-11"), T("10:00")], "must_not": [D("2027-02-09")]}, "lookup",
                   extra=[said("The dentist appointment is moved to 11 February, same time.", "update", "dentist moved", date="2027-02-11")]))
    t.append(mtask("D3-24", "sharada", "unknown", "2027-02-08", "What is Kamala's train number?", {"abstain": True}, "lookup",
                   note="Partial match: Kamala's arrival is saved, but no train."))
    # Arjun
    t.append(mtask("D3-25", "arjun", "lookup", "2027-02-08", "Aptitude test kab hai?", {"must_include": [D("2027-02-10"), T("14:00")]}, "lookup"))
    t.append(mtask("D3-26", "arjun", "semantic", "2027-02-08", "What should I take for the placement test?",
                   {"must_include": [X("resume", "resumes"), X("ID", "college ID", "ID card")]}, "meaning"))
    t.append(mtask("D3-27", "arjun", "plan", "2027-02-13", "What does my Monday look like?",
                   {"must_include": [T("18:00"), X("basketball")]}, "plan", period=("2027-02-15", "2027-02-15")))
    t.append(mtask("D3-28", "arjun", "date_window", "2027-02-16", "Do I have to pay anything before I go home for the engagement?",
                   {"must_include": [A(4200)]}, "window", period=("2027-02-16", "2027-02-24"),
                   note="Going home by bus on 25 Feb (engagement 27 Feb); the mess bill (20 Feb) falls before."))
    t.append(mtask("D3-29", "arjun", "update", "2027-02-12", "Did I return the library book?",
                   {"must_include": [X("yes", "returned")], "must_not": [X("not returned", "haven't returned", "still")]}, "status",
                   extra=[said("Returned the library book today.", "update", "library book returned", date="2027-02-11", status="done")],
                   note="Said on 11 Feb. 'Returned' is a new kind of update (not 'paid' or 'moved')."))
    t.append(mtask("D3-30", "arjun", "unknown", "2027-02-08", "When is the placement interview?", {"abstain": True}, "lookup",
                   note="Partial match: the aptitude test is saved, but no interview."))
    return t


def chat(tid, kind, question, rules, route, persona=None, today="2027-02-15", note=""):
    return {"id": tid, "kind": kind, "question": question, "rules": rules, "expect_route": route, "persona": persona,
            "today": today, "memory": BASES[persona] if persona else [], "note": note}


def chat_tasks() -> List[Dict[str, Any]]:
    t = [
        # maths: pure
        chat("M-01", "maths", "What is 18% GST on ₹2,450?", {"number": 441}, "maths"),
        chat("M-02", "maths", "If I invest ₹50,000 at 7% simple interest for 3 years, how much interest do I earn?", {"number": 10500}, "maths"),
        chat("M-03", "maths", "Split ₹3,780 equally among 4 people.", {"number": 945}, "maths"),
        chat("M-04", "maths", "What's 15% of 640?", {"number": 96}, "maths"),
        chat("M-05", "maths", "Convert 72 km/h to metres per second.", {"number": 20}, "maths"),
        chat("M-06", "maths", "₹12,600 paid in 6 equal instalments: how much is each?", {"number": 2100}, "maths"),
        chat("M-07", "maths", "How many days are there from 3 March to 18 April?", {"number": 46}, "maths"),
        chat("M-08", "maths", "Write 2.5 lakh as a plain number.", {"number": 250000}, "maths"),
        chat("M-09", "maths", "What is the area of a 12 ft by 14 ft room in square feet?", {"number": 168}, "maths"),
        chat("M-10", "maths", "Petrol is ₹104.50 a litre. What do 35 litres cost?", {"number": 3657.5}, "maths"),
        # maths with memory (mixed)
        chat("M-11", "maths", "How much will I spend on Netflix and the house help next month?", {"number": 8149}, "mixed", "priya", "2027-02-08",
             note="March: house help ₹7,500 (1st) + Netflix ₹649 (15th)."),
        chat("M-12", "maths", "If Sharma ji pays half now, how much will be left?", {"number": 2400}, "mixed", "ramesh", "2027-02-10"),
        chat("M-13", "maths", "What does the February mess bill come to per day?", {"number": 150}, "mixed", "arjun", "2027-02-08",
             note="₹4,200 / 28 days."),
        chat("M-14", "maths", "If I pay the property tax in two equal parts, how much is each part?", {"number": 2300}, "mixed", "lakshmi", "2027-02-10"),
        chat("M-15", "maths", "How many days until Kamala arrives?", {"number": 7}, "mixed", "sharada", "2027-02-15"),
        # general knowledge: stable facts
        chat("G-01", "general", "What is the capital of Odisha?", {"must_include": [X("Bhubaneswar")]}, "general"),
        chat("G-02", "general", "Who wrote Gitanjali?", {"must_include": [X("Tagore")]}, "general"),
        chat("G-03", "general", "At what temperature does water boil at sea level, in Celsius?", {"must_include": [X("100")]}, "general"),
        chat("G-04", "general", "Which planet is called the Red Planet?", {"must_include": [X("Mars")]}, "general"),
        chat("G-05", "general", "How many states does India have?", {"must_include": [X("28")]}, "general"),
        chat("G-06", "general", "What does UPI stand for?", {"must_include": [X("Unified Payments Interface")]}, "general"),
        chat("G-07", "general", "Who was the first Prime Minister of India?", {"must_include": [X("Nehru")]}, "general"),
        chat("G-08", "general", "What is the chemical symbol for iron?", {"must_include": [X("Fe")]}, "general"),
        chat("G-09", "general", "In which year did India become a republic?", {"must_include": [X("1950")]}, "general"),
        chat("G-10", "general", "What is the largest ocean on Earth?", {"must_include": [X("Pacific")]}, "general"),
        # general knowledge: needs live information
        chat("G-11", "general", "What is the gold price today?", {"no_live": True}, "general"),
        chat("G-12", "general", "What's the weather in Vellore right now?", {"no_live": True}, "general"),
        chat("G-13", "general", "Who won yesterday's cricket match?", {"no_live": True}, "general"),
        # general knowledge: nobody could know
        chat("G-14", "general", "What was the population of Kodur village in 1951?", {"unsure": True, "no_numbers": True}, "general"),
        chat("G-15", "general", "Who is the current headmaster of the government high school in Kodur?", {"unsure": True}, "general"),
        # writing and advice
        chat("W-01", "writing", "Write a short leave application to my manager for 18 and 19 February, for a family function.",
             {"must_include": [X("leave"), X("19 February", "19th February", "18 and 19", "18th and 19th")], "max_words": 150}, "writing", "priya", "2027-02-10"),
        chat("W-02", "writing", "Write a polite WhatsApp message to my landlord asking him not to raise the rent.",
             {"must_include": [X("rent")], "max_words": 120}, "writing", "priya", "2027-02-10"),
        chat("W-03", "writing", "Give me 3 tips to save electricity at home.",
             {"min_items": 3, "must_include": [X("light", "lights", "LED", "fan", "AC", "appliance", "appliances", "switch")]}, "writing"),
        chat("W-04", "writing", "How do I politely say no to lending money to a friend?", {"must_include": [X("no", "sorry", "can't", "cannot")], "max_words": 180}, "writing"),
        chat("W-05", "writing", "Draft an email to my bank asking to update my mobile number.",
             {"must_include": [X("mobile number"), X("request", "kindly", "please")], "max_words": 160}, "writing", "sharada", "2027-02-10"),
        chat("W-06", "writing", "Write a short birthday wish for my husband Murali.", {"must_include": [X("Murali"), X("birthday")], "max_words": 90},
             "writing", "lakshmi", "2027-02-18"),
        chat("W-07", "writing", "Suggest a simple weekly timetable to prepare for my placement test.",
             {"min_items": 3, "must_include": [X("aptitude", "test", "practice")]}, "writing", "arjun", "2027-02-01"),
        chat("W-08", "writing", "What should my mother keep in mind after cataract surgery?", {"must_include": [X("doctor", "eye drops", "drops")]}, "writing", "priya", "2027-02-08"),
        chat("W-09", "writing", "Write a thank-you note to the temple committee.", {"must_include": [X("thank", "Thank")], "max_words": 110}, "writing", "sharada", "2027-02-10"),
        chat("W-10", "writing", "Explain UPI to my grandmother in simple words.", {"must_include": [X("phone", "mobile"), X("bank")], "max_words": 140}, "writing"),
    ]
    return t


def router_tests() -> List[Dict[str, Any]]:
    r = lambda tid, q, route, persona, today, period=None: {"id": tid, "question": q, "expect_route": route, "persona": persona, "today": today,
                                                            "memory": BASES[persona], "period": list(period) if period else None}
    return [
        r("R-01", "What does my Saturday look like?", "plan", "lakshmi", "2027-02-17", ("2027-02-20", "2027-02-20")),
        r("R-02", "Is my mess bill still pending?", "status", "arjun", "2027-02-16"),
        r("R-03", "How much do I need for payments this week?", "window", "ramesh", "2027-02-22", ("2027-02-22", "2027-02-28")),
        r("R-04", "What is 18% of 2,450?", "maths", "priya", "2027-02-10"),
        r("R-05", "If my mess bill is split into 4, how much is each part?", "mixed", "arjun", "2027-02-10"),
        r("R-06", "Who is Kamala?", "people", "sharada", "2027-02-10"),
        r("R-07", "What is the capital of Kerala?", "general", "sharada", "2027-02-10"),
        r("R-08", "Write a leave letter for tomorrow.", "writing", "priya", "2027-02-10"),
        r("R-09", "When is my sister's engagement?", "lookup", "arjun", "2027-02-10"),
        r("R-10", "Anything before the Goa trip?", "window", "priya", "2027-02-15", ("2027-02-15", "2027-02-25")),
        r("R-11", "Tell me about my week.", "plan", "sharada", "2027-02-15", ("2027-02-15", "2027-02-21")),
        r("R-12", "What's the weather today?", "general", "ramesh", "2027-02-10"),
        r("R-13", "What did I note about the hotel booking?", "meaning", "priya", "2027-02-10"),
        r("R-14", "Mera Friday kaisa hai?", "plan", "lakshmi", "2027-02-17", ("2027-02-19", "2027-02-19")),
        r("R-15", "How can I make mom's follow-up visit less stressful for her?", "mixed", "priya", "2027-02-10"),
    ]


def review(mem, chats, routers) -> str:
    out = ["# Development set 3 (written before S4c)", "",
           "Generated by `chirii/eval/build_dev3_tasks.py` and committed before S4c was built. All fictional.", "",
           f"- {len(mem)} memory tasks: " + ", ".join(f"{TYPES[k]} {v}" for k, v in Counter(t['type'] for t in mem).items()),
           f"- {len(chats)} general-chat tasks: " + ", ".join(f"{k} {v}" for k, v in Counter(t['kind'] for t in chats).items()),
           f"- {len(routers)} router tests", "", "Routes: " + ", ".join(ROUTES) + ".", ""]
    for persona, label in PERSONAS.items():
        out += [f"## {label}", "", "**Memory (said earlier):**", ""] + [f"{i}. {m['said']}" for i, m in enumerate(BASES[persona], 1)] + [""]
        for t in (t for t in mem if t["persona"] == persona):
            today = date.fromisoformat(t["today"])
            out += [f"### {t['id']}: {TYPES[t['type']]} (route: {t['expect_route']})", "",
                    f"- **Today:** {today.strftime('%A')}, {render_fact({'date': t['today']})}"]
            out += [f"- **Also said:** {m['said']}" for m in t["memory"] if m not in BASES[persona]]
            out.append(f"- **Asks:** {t['question']}")
            if t.get("period"):
                out.append(f"- **Period:** {t['period'][0]} to {t['period'][1]}")
            out += [f"- {line}" for line in render_rules(t["rules"])] + ([f"- *Note:* {t['note']}"] if t["note"] else []) + [""]
    out += ["## General chat", "", "| Task | Kind | Person | Question | Expected route | Rules |", "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    for c in chats:
        out.append(f"| {c['id']} | {c['kind']} | {c['persona'] or '-'} | {c['question']} | {c['expect_route']} | {json.dumps(c['rules'], ensure_ascii=False)} |")
    out += ["", "## Router tests", "", "| Test | Person | Today | Question | Expected route | Expected period |", "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    for r in routers:
        out.append(f"| {r['id']} | {r['persona']} | {r['today']} | {r['question']} | {r['expect_route']} | {r['period'] or '-'} |")
    return "\n".join(out) + "\n"


def main():
    mem, chats, routers = memory_tasks(), chat_tasks(), router_tests()
    for name, rows in (("dev3_memory.jsonl", mem), ("dev3_chat.jsonl", chats), ("dev3_router.jsonl", routers)):
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "DEV3_TASKS.md"), "w", encoding="utf-8") as f:
        f.write(review(mem, chats, routers))
    print(f"Wrote {len(mem)} memory, {len(chats)} chat, {len(routers)} router tasks")


if __name__ == "__main__":
    main()

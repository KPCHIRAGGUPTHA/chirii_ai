"""
The 500-item scale check (S5), written and committed with its pass bars before it is run (CRITERIA.md).

One fictional user with 500 memories: 25 "needle" memories that the questions are about, and 475 fillers made by code
from everyday templates (a fixed random seed, so the set is the same every time). Three needles share their kind with
many fillers (gas, electricity and water bills in other months), so the question must pick the right month.

The same 30 questions are asked twice: with all 500 memories, and with a small store (the 25 needles and 15 fillers).
    25 lookups about needles (20 plain, 3 "crowded" bills, 2 notes)
     5 questions about things that are not saved (one of them shares a name with a needle)
plus 3 list questions (a week's bills, a day's plan, "before an event"), reported but not scored.

All names, places and numbers are fictional. Today is Monday 7 June 2027; everything was said on 1 June 2027.
    python -m chirii.eval.build_scale_set
"""

import json
import os
import random
from datetime import date, timedelta
from typing import Any, Dict, List

from chirii.eval.build_s1_tasks import ANY, A, D, T, X, said

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY, SAID_ON = "2027-06-07", "2027-06-01"
SEED = 20271122
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

NEEDLES = [
    said("Ishaan's passport appointment at the Seva Kendra is on 18 August at 10:30 am.", "appointment", "Ishaan's passport appointment", date="2027-08-18", time="10:30"),
    said("Plumber Murugan will come on 23 June at 4 pm to fix the bathroom tap.", "appointment", "Plumber Murugan", date="2027-06-23", time="16:00"),
    said("Yoga class every Wednesday at 6:15 am at the community hall.", "routine", "Yoga class", weekdays=["Wednesday"], time="06:15"),
    said("Zoya's school PTM is on 26 June at 9 am.", "appointment", "Zoya's PTM", date="2027-06-26", time="09:00"),
    said("Car service at the Hyundai workshop on 2 July.", "appointment", "Car service", date="2027-07-02"),
    said("Order Diwali sweets from Anand Sweets by 25 October.", "task", "Diwali sweets", date="2027-10-25"),
    said("Thatha's cataract surgery is on 14 September at 7 am.", "appointment", "Thatha's cataract surgery", date="2027-09-14", time="07:00"),
    said("Aarav's swimming fee of ₹1,800 is due on 30 June.", "bill", "Aarav's swimming fee", date="2027-06-30", amount=1800),
    said("Dr. Bhattacharya, the ENT doctor, on 11 July at 5:30 pm.", "appointment", "Dr. Bhattacharya (ENT)", date="2027-07-11", time="17:30"),
    said("LIC premium of ₹14,250 is due on 28 September.", "bill", "LIC premium", date="2027-09-28", amount=14250),
    said("Sneha's wedding sangeet is on 12 December in Udaipur.", "important_date", "Sneha's sangeet", date="2027-12-12"),
    said("The fridge warranty ends on 30 November.", "important_date", "Fridge warranty", date="2027-11-30"),
    said("Meenakshi owes me ₹2,200 for the Ooty trip.", "receivable", "Meenakshi's dues", amount=2200),
    said("Kindle Unlimited renewal of ₹169 on 19 June.", "bill", "Kindle Unlimited renewal", date="2027-06-19", amount=169),
    said("Paati's mango pickle needs 2 kg raw mangoes and 250 g mustard.", "note", "Paati's mango pickle"),
    said("Tatkal booking for the Chennai train opens on 9 July at 10 am.", "task", "Tatkal booking", date="2027-07-09", time="10:00"),
    said("Neighbour Kumaran is collecting our parcel on 15 June.", "task", "Kumaran's parcel", date="2027-06-15"),
    said("Guitar lesson with Fernandes every Friday at 7 pm.", "routine", "Guitar lesson", weekdays=["Friday"], time="19:00"),
    said("Rent agreement renewal with Mr. Pillai is due on 31 August.", "task", "Rent agreement renewal", date="2027-08-31"),
    said("Vaccination for our dog Bruno on 5 July at 11 am.", "appointment", "Bruno's vaccination", date="2027-07-05", time="11:00"),
    # crowded: many fillers share the kind, in other months
    said("Gas bill of ₹1,145 due on 16 October.", "bill", "Gas bill", date="2027-10-16", amount=1145),
    said("Electricity bill of ₹2,980 due on 9 December.", "bill", "Electricity bill", date="2027-12-09", amount=2980),
    said("Water bill of ₹640 due on 21 August.", "bill", "Water bill", date="2027-08-21", amount=640),
    # notes
    said("Note: the spare scooter key is in the blue tin on the kitchen shelf.", "note", "Spare scooter key"),
    said("Note: the Wi-Fi router is behind the TV in the hall.", "note", "Wi-Fi router"),
]
CROWDED = {"gas": 10, "electricity": 12, "water": 8}      # no filler bill of this kind in this month

NAMES = ["Anita", "Bharat", "Chitra", "Deepak", "Esha", "Farhan", "Gauri", "Harish", "Indu", "Jatin", "Kavya", "Lokesh",
         "Mansi", "Naveen", "Omkar", "Pallavi", "Qadir", "Rekha", "Sameer", "Tara", "Uday", "Vandana", "Waseem", "Yamini"]
SURNAMES = ["Rao", "Iyer", "Gupta", "Menon", "Kulkarni", "Sinha", "Joshi", "Nair"]
UTILITIES = ["water", "electricity", "broadband", "gas", "DTH", "mobile"]


def iso_text(rng: random.Random):
    d = date(2027, 6, 8) + timedelta(days=rng.randrange(0, 350))
    return d, f"{d.day} {MONTHS[d.month - 1]}"


def hhmm(rng: random.Random):
    h, m = rng.choice([8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]), rng.choice([0, 0, 30])
    shown = f"{h % 12 or 12}{':%02d' % m if m else ''} {'am' if h < 12 else 'pm'}"
    return f"{h:02d}:{m:02d}", shown


def filler(rng: random.Random) -> Dict[str, Any]:
    n = rng.choice(NAMES)
    kind = rng.randrange(9)
    if kind == 0:
        d, ds = iso_text(rng)
        return said(f"{n}'s birthday is on {ds}.", "important_date", f"{n}'s birthday", date=d.isoformat())
    if kind == 1:
        while True:
            d, ds = iso_text(rng)
            u = rng.choice(UTILITIES)
            if CROWDED.get(u) != d.month:
                break
        amt = rng.randrange(150, 4000)
        return said(f"Pay the {u} bill of ₹{amt:,} by {ds}.", "bill", f"{u} bill", date=d.isoformat(), amount=amt)
    if kind == 2:
        d, ds = iso_text(rng)
        t, ts = hhmm(rng)
        topic = rng.choice(["budget", "project review", "quarterly report", "vendor contract", "team outing", "appraisal"])
        return said(f"Meeting with {n} about the {topic} on {ds} at {ts}.", "appointment", f"Meeting with {n}", date=d.isoformat(), time=t)
    if kind == 3:
        amt = rng.randrange(100, 3000)
        thing = rng.choice(["movie tickets", "dinner", "cab ride", "books", "concert passes", "groceries"])
        return said(f"{n} owes me ₹{amt:,} for the {thing}.", "receivable", f"{n}'s dues", amount=amt)
    if kind == 4:
        d, ds = iso_text(rng)
        what = rng.choice(["rice", "atta", "detergent", "notebooks", "light bulbs", "batteries"])
        shop = rng.choice(["D-Mart", "More", "Reliance Fresh", "the corner shop"])
        return said(f"Buy {what} from {shop} on {ds}.", "task", f"Buy {what}", date=d.isoformat())
    if kind == 5:
        d, ds = iso_text(rng)
        thing = rng.choice(["umbrella", "drill machine", "cookbook", "suitcase", "extension cord"])
        return said(f"Return {n}'s {thing} by {ds}.", "task", f"Return {n}'s {thing}", date=d.isoformat())
    if kind == 6:
        pref = rng.choice(["green tea", "morning calls", "WhatsApp over email", "window seats", "south Indian food"])
        return said(f"Note: {n} prefers {pref}.", "note", f"{n} prefers {pref}")
    if kind == 7:
        day = rng.choice(["Monday", "Tuesday", "Thursday", "Saturday", "Sunday"])
        t, ts = hhmm(rng)
        return said(f"Badminton with {n} every {day} at {ts}.", "routine", f"Badminton with {n}", weekdays=[day], time=t)
    d, ds = iso_text(rng)
    t, ts = hhmm(rng)
    s = rng.choice(SURNAMES)
    return said(f"Dermatologist Dr. {s} on {ds} at {ts}.", "appointment", f"Dr. {s}", date=d.isoformat(), time=t)


def fillers(count: int) -> List[Dict[str, Any]]:
    rng, out, seen = random.Random(SEED), [], set()
    while len(out) < count:
        f = filler(rng)
        if f["said"] not in seen:
            seen.add(f["said"])
            out.append(f)
    return out


def q(tid, kind, question, rules, note=""):
    return {"id": tid, "type": kind, "today": TODAY, "question": question, "rules": {**rules, "consistent": True, "no_the_user": True},
            "note": note}


def questions() -> List[Dict[str, Any]]:
    return [
        q("SC-01", "lookup", "When is Ishaan's passport appointment?", {"must_include": [D("2027-08-18"), T("10:30")]}),
        q("SC-02", "lookup", "When is the plumber coming?", {"must_include": [D("2027-06-23"), T("16:00")]}),
        q("SC-03", "lookup", "What time is my yoga class?", {"must_include": [T("06:15")]}),
        q("SC-04", "lookup", "When is Zoya's PTM?", {"must_include": [D("2027-06-26"), T("09:00")]}),
        q("SC-05", "lookup", "When is the car service?", {"must_include": [D("2027-07-02")]}),
        q("SC-06", "lookup", "By when should I order the Diwali sweets?", {"must_include": [D("2027-10-25")]}),
        q("SC-07", "lookup", "When is Thatha's cataract surgery?", {"must_include": [D("2027-09-14"), T("07:00")]}),
        q("SC-08", "lookup", "How much is Aarav's swimming fee and when is it due?", {"must_include": [A(1800), D("2027-06-30")]}),
        q("SC-09", "lookup", "When am I seeing Dr. Bhattacharya?", {"must_include": [D("2027-07-11"), T("17:30")]}),
        q("SC-10", "lookup", "How much is the LIC premium?", {"must_include": [A(14250)]}),
        q("SC-11", "lookup", "Where is Sneha's sangeet?", {"must_include": [X("Udaipur")]}),
        q("SC-12", "lookup", "When does the fridge warranty end?", {"must_include": [D("2027-11-30")]}),
        q("SC-13", "lookup", "How much does Meenakshi owe me?", {"must_include": [A(2200)]}),
        q("SC-14", "lookup", "When is the Kindle renewal?", {"must_include": [D("2027-06-19")]}),
        q("SC-15", "lookup", "How many raw mangoes does Paati's pickle need?", {"must_include": [X("2 kg", "2kg", "two kg", "2 kilo")]}),
        q("SC-16", "lookup", "When does Tatkal booking open for the Chennai train?", {"must_include": [D("2027-07-09"), T("10:00")]}),
        q("SC-17", "lookup", "When is Kumaran collecting the parcel?", {"must_include": [D("2027-06-15")]}),
        q("SC-18", "lookup", "What time is my guitar lesson?", {"must_include": [T("19:00")]}),
        q("SC-19", "lookup", "When is the rent agreement renewal due?", {"must_include": [D("2027-08-31")]}),
        q("SC-20", "lookup", "When is Bruno's vaccination?", {"must_include": [D("2027-07-05"), T("11:00")]}),
        q("SC-21", "lookup", "How much is the gas bill due in October?", {"must_include": [A(1145)]}, note="crowded"),
        q("SC-22", "lookup", "When is the electricity bill due in December, and how much?", {"must_include": [D("2027-12-09"), A(2980)]}, note="crowded"),
        q("SC-23", "lookup", "How much is the water bill due in August?", {"must_include": [A(640)]}, note="crowded"),
        q("SC-24", "lookup", "Where is the spare scooter key?", {"must_include": [X("blue tin")]}),
        q("SC-25", "lookup", "Where is the Wi-Fi router?", {"must_include": [X("behind the TV", "behind the tv")]}),
        q("SC-26", "unknown", "When is Ishaan's visa interview?", {"abstain": True}, note="Ishaan's passport appointment is saved"),
        q("SC-27", "unknown", "What is Zoya's blood group?", {"abstain": True}),
        q("SC-28", "unknown", "When is the car insurance due?", {"abstain": True}, note="the car service is saved"),
        q("SC-29", "unknown", "How much does Rohit owe me?", {"abstain": True}),
        q("SC-30", "unknown", "When is Thatha's dentist appointment?", {"abstain": True}, note="Thatha's cataract surgery is saved"),
    ]


LIST_QUESTIONS = ["Which bills do I have to pay this week?", "What's my plan for Saturday?", "Anything to pay before Sneha's sangeet?"]


def main():
    all_fill = fillers(475)
    stores = {"large": NEEDLES + all_fill, "small": NEEDLES + all_fill[:15]}
    out = {"today": TODAY, "said_on": SAID_ON, "stores": stores, "questions": questions(), "list_questions": LIST_QUESTIONS}
    with open(os.path.join(HERE, "scale_set.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print({k: len(v) for k, v in stores.items()}, len(out["questions"]))


if __name__ == "__main__":
    main()

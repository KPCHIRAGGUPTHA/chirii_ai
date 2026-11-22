"""
How Chirii writes items for people: dates with weekdays, 12-hour times, rupees in Indian grouping, a short title, and
the "Got it" summary shown after each saved memory with anything it is unsure about.
"""

import re
from datetime import date
from typing import Any, Dict, List

KIND_LABELS = {"bill": "Bill", "appointment": "Appointment", "task": "To do", "medicine": "Medicine", "routine": "Routine",
               "important_date": "Important date", "exam": "Exam", "travel": "Travel", "shift": "Work shift", "income": "Income",
               "receivable": "Money owed to you", "note": "Note", "update": "Change to an earlier item",
               "profile": "About you", "person": "Person"}
DATED_KINDS = {"appointment", "exam", "travel", "important_date", "shift"}


def say_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def say_time(hhmm: str) -> str:
    hour, minute = (int(x) for x in hhmm.split(":"))
    return f"{hour % 12 or 12}:{minute:02d} {'am' if hour < 12 else 'pm'}"


def say_amount(rupees: int) -> str:
    digits = str(rupees)
    if len(digits) > 3:  # Indian grouping: 1,95,000
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join(([head] if head else []) + groups) + "," + tail
    return f"₹{digits}"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


ABBREVIATIONS = ("dr", "mr", "mrs", "ms", "smt", "shri", "sri", "prof", "st", "sr", "jr")


def clauses(text: str) -> list:
    """The text split at full stops, semicolons and commas, but never after an honorific ('Dr.', 'Mrs.', 'Smt.')."""
    parts, start = [], 0
    for match in re.finditer(r"[.;](?:\s+|$)|,\s+", text):
        if text[match.start()] == "." and text[start:match.start()].split() and \
                text[start:match.start()].split()[-1].lower().rstrip(".") in ABBREVIATIONS:
            continue
        parts.append(text[start:match.start()])
        start = match.end()
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]
# Where a name stops and the details start: "BESCOM electricity bill | is ₹1,860", "Hostel mess fee | of ₹38,500"
TITLE_CUT = re.compile(r"\s+(?:(?:is|are|was|were|must|goes|costs?|due|on|at|by|before|between|every|each|of(?=\s+(?:₹|rs\b|inr\b|\d))|"
                       r"for(?=\s+(?:₹|\d)))\b|(?=₹)|(?=rs\.?\s*\d))", re.I)


def first_clause(text: str) -> str:
    """Text up to the first full stop, semicolon, colon or comma, not counting the full stop of 'Dr.' or 'Prof.'."""
    for match in re.finditer(r"[.;:](?:\s|$)|,\s", text):
        before = text[:match.start()].split()        # S5: "Note." left nothing before the stop and crashed
        if text[match.start()] == "." and before and before[-1].lower().rstrip(".") in ABBREVIATIONS:
            continue
        return text[:match.start()]
    return text


def make_title(text: str) -> str:
    """A short name for the item: the start of what the user said, up to where the details begin, without leading
    filler, at most 60 characters. Changed 22 Nov (S4): titles used to stop at 'Dr.' and run on into dates."""
    title = re.sub(r"^\s*(note|correction|reminder|remind me to)(\s+from)?\s*[:,-]?\s*", "", text.strip(), flags=re.I)
    # a leading date ("From 14 to 18 December I'm on the night shift", "On 10 December I have a team call")
    title = re.sub(r"^(from|on)\s+[\d\w ]*?\b(january|february|march|april|may|june|july|august|september|october|november|december)\b,?\s+",
                   "", title, flags=re.I)
    title = first_clause(title).strip()
    for cut in TITLE_CUT.finditer(title):
        words = title[:cut.start()].split()
        if len(words) >= 2 and words[-1].lower() not in ("i", "i'm", "we", "you", "he", "she", "they", "it"):
            title = title[:cut.start()]
            break
    title = re.sub(r"^(my|our|the)\s+", "", title, flags=re.I)
    # S4f: "Kiraya of ₹18,000 goes…" was titled "Kiraya of"; a connecting word left at the cut is dropped
    title = re.sub(r"(\s+(of|for|at|is|are|on|to|from|by|with|and))+\s*$", "", title, flags=re.I)
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "…"
    return title[:1].upper() + title[1:] if title else text[:60]


def when_text(item: Dict[str, Any]) -> List[str]:
    parts = []
    if item.get("date"):
        prefix = "due " if item["kind"] in ("bill", "task") else "on "
        parts.append(prefix + say_date(item["date"]) + (f" to {say_date(item['end_date'])}" if item.get("end_date") else ""))
        if item.get("other_dates"):  # S3 bug: "exams on 7, 9 and 11 December" showed only 7 December
            parts.append("also on " + ", ".join(say_date(d) for d in item["other_dates"]))
    if item.get("day_of_month"):
        parts.append(f"on the {ordinal(item['day_of_month'])} of every month")
    if item.get("weekdays"):
        parts.append("every " + " and ".join(item["weekdays"]))
    if item.get("time"):
        parts.append("at " + say_time(item["time"]) + (f" until {say_time(item['end_time'])}" if item.get("end_time") else ""))
    if item.get("dose_times"):
        parts.append(", ".join(item["dose_times"]))
    return parts


def uncertainties(item: Dict[str, Any]) -> List[str]:
    """What Chirii isn't sure about in how it read this item. Empty means nothing to flag."""
    reasons = []
    source = item.get("source", "rules")
    if source == "model" and item.get("needs_review"):
        reasons.append("I guessed what kind of item this is")
    elif source.startswith("rules (") and item.get("needs_review"):
        reasons.append("I couldn't read this sentence fully")
    if item["kind"] == "update" and item.get("status") == "unlinked":
        reasons.append("I couldn't tell which earlier item this changes")
    if item["kind"] == "bill":
        if item.get("amount") is None:
            reasons.append("no amount found")
        if not (item.get("date") or item.get("day_of_month")):
            reasons.append("no due date found")
    if item["kind"] in DATED_KINDS and not item.get("date"):
        reasons.append("no date found")
    if item["kind"] == "appointment" and item.get("date") and not item.get("time"):
        reasons.append("no time found")
    return reasons


def understood(item: Dict[str, Any], changes: List[Dict[str, Any]] = (), target: Dict[str, Any] = None) -> Dict[str, Any]:
    """The "Got it" card for a saved memory."""
    parts = [KIND_LABELS.get(item["kind"], item["kind"])]
    if item.get("amount") is not None and item["kind"] != "update":
        parts.append(say_amount(item["amount"]))
    parts += when_text(item) if item["kind"] != "update" else []
    if item["kind"] in ("profile", "person"):                 # S4e: show the people facts read from the sentence
        parts += people_facts(item)
    if item["kind"] == "update" and target is not None:
        described = []
        for change in changes:
            if change["field"] == "status":
                described.append(f"marked {change['new']}")
            elif change["field"] == "date":
                described.append(f"date {say_date(change['old'])} → {say_date(change['new'])}" if change["old"] else f"date now {say_date(change['new'])}")
            elif change["field"] == "time":
                described.append(f"time now {say_time(change['new'])}")
            elif change["field"] == "paid_amount":
                described.append(f"{say_amount(target['remaining'])} still owed")
        parts.append(f"updated \"{target.get('title') or make_title(target['text'])}\"" + (": " + ", ".join(described) if described else ""))
    return {
        "id": item["id"], "kind": item["kind"], "kind_label": KIND_LABELS.get(item["kind"], item["kind"]),
        "title": item.get("title") or make_title(item["text"]), "date": item.get("date"), "end_date": item.get("end_date"),
        "time": item.get("time"), "amount": item.get("amount"), "day_of_month": item.get("day_of_month"),
        "weekdays": item.get("weekdays", []), "summary": ", ".join(parts), "uncertain": uncertainties(item),
        "confirmed": bool(item.get("confirmed")),
    }


def people_facts(item: Dict[str, Any]) -> List[str]:
    """'your name: Priya Raman', 'Lalitha: your mother', for the card of a profile or person sentence."""
    from chirii.app import people  # people imports this module, so import it here
    ppl = people.build([item])
    facts = [f"your name: {n}" for n in ppl.names] + [f"{who} call you {nick}" for nick, who in ppl.nicknames]
    for p in ppl.people.values():
        facts.append(f"{p.label()}: {people.describe(p, ppl)}" + (f" (also called {', '.join(sorted(p.aliases))})" if p.name and p.aliases else ""))
    return facts

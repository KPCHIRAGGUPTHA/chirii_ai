"""
How Chirii writes items for people: dates with weekdays, 12-hour times, rupees in Indian grouping, a short title, and
the "Got it" summary shown after each saved memory with anything it is unsure about.
"""

import re
from datetime import date
from typing import Any, Dict, List

KIND_LABELS = {"bill": "Bill", "appointment": "Appointment", "task": "To do", "medicine": "Medicine", "routine": "Routine",
               "important_date": "Important date", "exam": "Exam", "travel": "Travel", "shift": "Work shift", "income": "Income",
               "receivable": "Money owed to you", "note": "Note", "update": "Change to an earlier item"}
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


def make_title(text: str) -> str:
    """A short name for the item: the first clause of what the user said, without leading filler, at most 60 chars."""
    title = re.sub(r"^\s*(note|correction|reminder|remind me to)\s*[:,-]?\s*", "", text.strip(), flags=re.I)
    title = re.split(r"(?<=[a-z0-9)])[.;:](?:\s|$)|,\s", title, maxsplit=1)[0].strip()
    title = re.sub(r"^(my|our|the)\s+", "", title, flags=re.I)
    if len(title) > 60:
        title = title[:57].rsplit(" ", 1)[0] + "…"
    return title[:1].upper() + title[1:] if title else text[:60]


def when_text(item: Dict[str, Any]) -> List[str]:
    parts = []
    if item.get("date"):
        prefix = "due " if item["kind"] in ("bill", "task") else "on "
        parts.append(prefix + say_date(item["date"]) + (f" to {say_date(item['end_date'])}" if item.get("end_date") else ""))
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

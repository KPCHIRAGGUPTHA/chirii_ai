"""
Hybrid sentence reader: the rules first, Qwen only when the rules give up, and code checks every value Qwen returns.

1. The rules parser (extract.py) runs first: fast, and the same answer every time.
2. If the rules fall back to a plain note, or miss a key field (a bill with no amount or due date, an appointment with
   no date), Qwen is asked for the same item as JSON. llama-server constrains its reply to a JSON schema in which
   every date, time, amount, day and weekday can only be one the code found in the sentence (or empty).
3. Code checks every date, time and amount Qwen gives against the sentence itself. Each one must be something the
   sentence actually says: a date written in it, a word like "tomorrow", "kal" or "Sunday", a day like "the 15th" or
   "18 tareekh"; a time or hour written in it; a number written in it. Qwen chooses among what is there, it cannot add.
4. If any value fails the check, Qwen's reply is thrown away. The item stays what the rules made (often a plain note)
   and is marked for the user to check.

Rules values always win: Qwen only fills fields the rules left empty, and decides the kind only when the rules couldn't.
Every use of the fallback is logged with counts only (no text).
"""

import json
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from chirii.app.llm.client import ChatModel, LLMError
from chirii.app.logging_setup import log_event
from chirii.app.memory.extract import NOTE_START, WEEKDAYS, Item, extract, find_amounts, find_dates, find_times

KINDS = ["bill", "appointment", "task", "medicine", "routine", "important_date", "exam", "travel", "shift", "income",
         "receivable", "note"]
KIND_HELP = {
    "bill": "a payment the user must make (bill, fee, EMI, premium, rent)",
    "appointment": "something at a set time (doctor, meeting, visit, class, function)",
    "task": "something the user must do by a date (renew, submit, book, collect)",
    "medicine": "a medicine and when to take it",
    "routine": "something that repeats on set weekdays",
    "important_date": "a birthday, anniversary, festival or family occasion",
    "exam": "an exam", "travel": "a journey", "shift": "a work shift",
    "income": "money the user receives", "receivable": "money someone owes the user",
    "note": "anything else to remember",
}
DATED_KINDS = {"appointment", "exam", "travel", "important_date", "shift"}

RELATIVE_WORDS = {r"\b(today|tonight|aaj)\b": [0], r"\b(tomorrow|tmrw|tmr|tomorow)\b": [1], r"\byesterday\b": [-1],
                  r"\bday after tomorrow\b": [2], r"\bkal\b": [1, -1], r"\bparso(n)?\b": [2, -2]}
BARE_DAY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b|\b(\d{1,2})\s*(?:tareekh|tarikh|tarik)\b", re.I)
IN_DAYS = re.compile(r"\bin\s+(\d{1,2})\s+days?\b", re.I)
BARE_HOUR = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(?:baje|bje|o'?clock)\b|\bat\s+(\d{1,2})(?:[:.](\d{2}))?\b(?!\s*(?:st|nd|rd|th|%|kg|days?))", re.I)
BARE_NUMBER = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{2,3})+|\d{2,7})(?![\d.]|\s*(?:st|nd|rd|th|kg|g|litres?|l|am|pm|%))", re.I)
THOUSANDS = re.compile(r"\b(\d+(?:\.\d+)?)\s*k\b", re.I)


def needs_model(item: Item, text: str) -> bool:
    """True when the rules gave up: a fallback plain note, or a kind whose key field is missing."""
    if item.kind == "update":
        return False
    if item.kind == "note":
        return not NOTE_START.search(text)
    if item.kind in DATED_KINDS and not item.date:
        return True
    if item.kind == "bill" and (item.amount is None or not (item.date or item.day_of_month)):
        return True
    if item.kind == "receivable" and item.amount is None:
        return True
    return False


# ---------------------------------------------------------------- what the sentence actually says

def grounded_dates(text: str, said_on: date) -> Set[date]:
    found = {d for _, d in find_dates(text, said_on)[0]}
    end = find_dates(text, said_on)[1]
    if end:
        found.add(end)
    lowered = text.lower()
    for pattern, offsets in RELATIVE_WORDS.items():
        if re.search(pattern, lowered):
            found |= {said_on + timedelta(days=o) for o in offsets}
    for i, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}\b", text, re.I):
            found |= {said_on + timedelta(days=k) for k in range(0, 8) if (said_on + timedelta(days=k)).weekday() == i}
    for m in BARE_DAY.finditer(text):
        day = int(m.group(1) or m.group(2))
        for months_ahead in (0, 1):
            index = said_on.month - 1 + months_ahead
            try:
                found.add(date(said_on.year + index // 12, index % 12 + 1, day))
            except ValueError:
                pass
    for m in IN_DAYS.finditer(text):
        found.add(said_on + timedelta(days=int(m.group(1))))
    return found


def grounded_times(text: str) -> Set[str]:
    starts, end = find_times(text)
    found = {t for _, t in starts} | ({end} if end else set())
    for m in BARE_HOUR.finditer(text):
        hour = int(m.group(1) or m.group(3))
        minute = int(m.group(2) or m.group(4) or 0)
        if 1 <= hour <= 12 and minute < 60:
            found |= {f"{hour % 12:02d}:{minute:02d}", f"{hour % 12 + 12:02d}:{minute:02d}"}
        elif hour <= 23 and minute < 60:
            found.add(f"{hour:02d}:{minute:02d}")
    return found


def grounded_amounts(text: str) -> Set[int]:
    found = {a for _, a in find_amounts(text)}
    for m in BARE_NUMBER.finditer(text):
        found.add(int(m.group(1).replace(",", "")))
    for m in THOUSANDS.finditer(text):
        found.add(round(float(m.group(1)) * 1000))
    return found


def grounded_days_of_month(text: str) -> Set[int]:
    return {int(n) for n in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text) if 1 <= int(n) <= 31}


def validate(reply: Dict[str, Any], text: str, said_on: date) -> List[str]:
    """Problems with Qwen's item. Empty means every value is grounded in the sentence."""
    problems = []
    if reply.get("kind") not in KINDS:
        problems.append("kind")
    dates = grounded_dates(text, said_on)
    for name in ("date", "end_date"):
        value = reply.get(name)
        if value:
            try:
                if date.fromisoformat(value) not in dates:
                    problems.append(name)
            except ValueError:
                problems.append(name)
    if reply.get("time") and reply["time"] not in grounded_times(text):
        problems.append("time")
    if reply.get("amount") is not None and reply["amount"] not in grounded_amounts(text):
        problems.append("amount")
    if reply.get("day_of_month") is not None and reply["day_of_month"] not in grounded_days_of_month(text):
        problems.append("day_of_month")
    for weekday in reply.get("weekdays") or []:
        if not re.search(rf"\b{weekday}\b", text, re.I):
            problems.append("weekdays")
    return problems


def _choice(values: List[Any], kind: str) -> Dict[str, Any]:
    """A field that may only be null or one of the given values."""
    return {"anyOf": [{"type": kind, "enum": values}, {"type": "null"}]} if values else {"type": "null"}


def build_schema(text: str, said_on: date) -> Dict[str, Any]:
    """The JSON schema for this sentence: each date, time, amount, day or weekday Qwen may give is limited to the
    values the code found in the sentence (first run, 21 Nov: unrestricted, Qwen added weekdays and today's date that
    the sentence never said). The check in validate() still runs afterwards."""
    weekdays = [w for w in WEEKDAYS if re.search(rf"\b{w}\b", text, re.I)]
    dates = sorted(d.isoformat() for d in grounded_dates(text, said_on))
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS},
            "date": _choice(dates, "string"), "end_date": _choice(dates, "string"),
            "time": _choice(sorted(grounded_times(text)), "string"),
            "amount": _choice(sorted(grounded_amounts(text)), "integer"),
            "day_of_month": _choice(sorted(grounded_days_of_month(text)), "integer"),
            "weekdays": {"type": "array", "items": {"type": "string", "enum": weekdays}} if weekdays else {"type": "array", "maxItems": 0},
        },
        "required": ["kind", "date", "end_date", "time", "amount", "day_of_month", "weekdays"],
    }


def build_messages(text: str, said_on: date) -> List[Dict[str, str]]:
    kinds = "\n".join(f"- {k}: {v}" for k, v in KIND_HELP.items())
    system = (
        "You turn one sentence that a user told their personal assistant into a memory item in JSON. The sentence may "
        "be casual, incomplete or mix English with Hindi words.\n"
        f"Today is {WEEKDAYS[said_on.weekday()]}, {said_on.day} {said_on.strftime('%B %Y')} ({said_on.isoformat()}).\n\n"
        f"kind is one of:\n{kinds}\n\n"
        "Fields:\n"
        "- date: the day it happens or is due, as YYYY-MM-DD. Work out words like tomorrow, kal, Sunday or 'the 15th' "
        "from today. null if no day is given.\n"
        "- end_date: the last day of a range, else null.\n"
        "- time: 24-hour HH:MM, else null. If am or pm is not said and not clear, use null.\n"
        "- amount: rupees as a whole number (8k means 8000), else null.\n"
        "- day_of_month: for something every month, the day number, else null.\n"
        "- weekdays: for something every week, the weekday names, else [].\n"
        "Never guess. Use null for anything the sentence does not say."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


class HybridExtractor:

    def __init__(self, model: ChatModel):
        self.model = model

    def __call__(self, text: str, said_on: date) -> Item:
        item = extract(text, said_on)
        if not needs_model(item, text):
            return item
        try:
            reply = json.loads(self.model.chat(build_messages(text, said_on), max_tokens=160, json_schema=build_schema(text, said_on),
                                               purpose="extract").text)
        except (LLMError, ValueError):
            item.needs_review = True
            item.source = "rules (model unavailable)"
            log_event("extract_fallback", status="error", item_type=item.kind, stage="extract")
            return item
        problems = validate(reply, text, said_on) if isinstance(reply, dict) else ["not an object"]
        if problems:
            item.needs_review = True
            item.source = "rules (model reply rejected)"
            log_event("extract_fallback", status="rejected", item_type=item.kind, result_count=len(problems), stage="extract")
            return item
        return self._merge(item, reply)

    @staticmethod
    def _merge(item: Item, reply: Dict[str, Any]) -> Item:
        if item.kind == "note":  # the rules couldn't tell what it is; otherwise the rules' kind stands
            item.kind = reply["kind"]
            # Code checks Qwen's dates and amounts against the sentence, but nothing can check its choice of kind, so
            # the user is asked to confirm it (added 21 Nov after set B: Qwen turned a plain note into a date).
            item.needs_review = reply["kind"] != "note"
        for name in ("date", "end_date", "time", "amount", "day_of_month"):
            if getattr(item, name) in (None, "") and reply.get(name) not in (None, ""):
                setattr(item, name, reply[name])
        if not item.weekdays and reply.get("weekdays"):
            item.weekdays = list(reply["weekdays"])
        item.source = "model"
        log_event("extract_fallback", status="accepted", item_type=item.kind, stage="extract")
        return item

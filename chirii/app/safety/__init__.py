"""
S5 safety layer.

1. Self-harm (reuses phase14/safety_layer.py, adapted for Chirii). Every question, and every memory being saved, is
   checked BEFORE Qwen sees it. If it shows first-person self-harm or suicide intent, Chirii gives a fixed supportive
   reply with Tele-MANAS 14416 and 112, and Qwen is not called. It never gives method information. phase14's rule
   (a first-person cue AND a harm cue, or one unambiguous direct phrase; third-party and helper wording suppressed)
   keeps idioms ("this exam is killing me") and tech wording ("kill a process") from firing. Added for Chirii:
   - Hinglish and Indian wording ("marna chahta hoon", "jeena nahi chahti", "khudkushi", "aatmahatya");
   - a saved memory that fires is still saved (it is the person's own note), and the support message is shown with it.

2. Attribute check (owner, after run (e5), D5-M12: "When does Mr. Desai's leave start?" was answered from his weekly
   visit). A question about an attribute or event of something ("leave", "interview", "joining", "fee", "surgery",
   "flight"…) needs a memory that mentions that attribute or event: exactly, by stem, by the synonym map, or by
   meaning above the calibrated threshold. Two places:
   - before answering, these words are key terms for the specific-item check, so a memory that doesn't mention one
     gives "not saved" or a partial match ("I have Mr. Desai's visit (every Friday at 4 pm), but nothing about his
     leave");
   - after answering, a code check on the final answer: an answer drawn from a memory may not give that memory's date,
     time or weekday for a different event named in the question. If it does, the answer is replaced by the partial
     match. The fact-check guard can't catch this: the date is real, it just belongs to something else.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from phase14.safety_layer import SUPPORT_MESSAGE, check as phase14_check

# ---------------------------------------------------------------- 1. self-harm

HINGLISH = re.compile(
    r"\b(main|mai|mein|mujhe|mujhko)\b.{0,25}\b(marna|mar jaana|mar jana|mar jaun|mar jaaun|khatam karna)\b.{0,15}\b(chahta|chahti|chahata|hai|hoon|hu|hun)\b"
    r"|\bjeena nahi\b.{0,10}\b(chahta|chahti|hai)\b|\bjeene ka (man|mann) nahi\b|\bzindagi (khatam|khtm) karna\b"
    r"|\bkhud ?kushi\b|\bkhudkhushi\b|\baatmahatya\b|\batmahatya\b|\bsuicide (karna|kar lunga|kar lungi|kar lu)\b", re.I)
HINGLISH_THIRD_PARTY = re.compile(r"\b(dost|friend|bhai|behen|didi|bhaiya|papa|mummy|amma|appa|beta|beti)\b.{0,20}\b(ko|ka|ki|ke)\b", re.I)


def self_harm(text: str) -> Optional[str]:
    """The support message if the text shows self-harm or suicide intent, else None."""
    if phase14_check(text).fired:
        return SUPPORT_MESSAGE
    if HINGLISH.search(text) and not HINGLISH_THIRD_PARTY.search(text):
        return SUPPORT_MESSAGE
    return None


# ---------------------------------------------------------------- 2. attribute check

# Attributes and events a question can ask about. A memory must mention the one asked about.
EVENT_NOUNS = {
    "interview", "joining", "surgery", "operation", "flight", "fee", "fees", "salary", "increment", "bonus", "pension",
    "retirement", "transfer", "promotion", "admission", "result", "results", "exam", "exams", "wedding", "engagement",
    "birthday", "anniversary", "funeral", "vacation", "holiday", "holidays", "trip", "deadline", "renewal", "premium",
    "refund", "deposit", "rent", "emi", "loan", "checkup", "check-up", "scan", "vaccination", "vaccine", "delivery",
    "appointment", "meeting", "ceremony", "party", "function", "concert", "booking", "ticket", "visa", "passport",
    "license", "licence", "insurance", "warranty", "registration", "convocation", "graduation", "internship", "shift",
}
# Words that are events only when used as nouns: "Mr. Desai's leave", "my leave", "sick leave", "the visit".
NOUN_ONLY = re.compile(
    r"(?:'s|\b(?:my|his|her|their|our|your|the|a|an|annual|sick|casual|maternity|paternity|earned|next|last|this))\s+"
    r"(leave|visit|test|match|class|practice|duty|break|launch|move|stay|treatment|therapy|course)\b"
    r"|\b(leave|visit|test|match|class|duty|treatment|therapy|course)\s+(start|starts|begin|begins|end|ends|date|dates|time|timing|fee|fees)\b",
    re.I)


# What an item's kind already says about it: a bill is a fee or payment even if the sentence says "membership"
KIND_WORDS = {"bill": "fee fees bill payment dues charges", "appointment": "appointment visit",
              "exam": "exam exams test", "receivable": "dues payment", "income": "salary payment"}


def item_words(item: Dict[str, Any]) -> str:
    """The text an item is matched on: what was said, its title, and the words its kind implies."""
    return f"{item['text']} {item.get('title') or ''} {KIND_WORDS.get(item.get('kind'), '')}"


def asked_attributes(question: str) -> List[str]:
    """The attribute or event words a question asks about, in order ("leave" in "When does Mr. Desai's leave start?")."""
    q = question.replace("’", "'")
    found: List[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z\-]*", q):
        if word.lower() in EVENT_NOUNS and word.lower() not in found:
            found.append(word.lower())
    for m in NOUN_ONLY.finditer(q):
        word = (m.group(1) or m.group(2)).lower()
        if word not in found:
            found.append(word)
    return found


def _answer_facts(answer: str, today) -> Tuple[set, set, set]:
    from chirii.app.memory.extract import find_times
    from chirii.eval.scoring import find_dates               # the same date reader the fact-check guard uses
    dates = set(find_dates(answer, today))
    starts, end = find_times(answer)
    times = {t for _, t in starts} | ({end} if end else set())
    days = {d for d in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday") if re.search(rf"\b{d}s?\b", answer, re.I)}
    return dates, times, days


def _supplies(item: Dict[str, Any], dates: set, times: set, days: set) -> bool:
    """Whether an answer gives this item's own date, time or weekday."""
    from datetime import date as _date
    item_dates = {(_date.fromisoformat(d).day, _date.fromisoformat(d).month)
                  for d in [item.get("date"), item.get("end_date"), *(item.get("other_dates") or [])] if d}
    item_times = {t for t in [item.get("time"), *(item.get("other_times") or [])] if t}
    item_days = {w.lower() for w in item.get("weekdays") or []}
    return bool(item_dates & dates or item_times & times or item_days & days)


def attribute_check(question: str, answer: str, items: List[Dict[str, Any]], today) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """(item, missing attributes) when the answer's date, time or weekday belongs only to saved items that don't
    mention the attribute or event asked about; None when the answer is fine.

    `items` is everything saved, not only what Qwen was given: a borrowed date may come from any memory. A date that
    belongs to no memory at all is left to the fact-check guard."""
    from chirii.app.router import term_matches
    asked = asked_attributes(question)
    if not asked or not items:
        return None
    dates, times, days = _answer_facts(answer, today)
    if not (dates or times or days):
        return None
    text = item_words
    suppliers = [i for i in items if _supplies(i, dates, times, days)]
    if not suppliers or any(all(term_matches(a, text(i)) for a in asked) for i in suppliers):
        return None
    item = suppliers[0]
    return item, [a for a in asked if not term_matches(a, text(item))]

"""
S1 eval scorer. The rules are fixed here before any system is run (see chirii/CRITERIA.md).

A task passes only if every rule on it passes:
    must_include   every listed fact appears in the reply
    must_not       none of these facts appear (stale, wrong or out-of-range items)
    order          these facts appear in this order (plans)
    flag_clash     the reply points out a clash
    abstain        the reply says the answer isn't in memory
    no invention   every date and rupee amount in the reply exists in the user's memory or the task's "allow" list, or is
                   a correct value worked out from them (see below). This applies to every task: Chirii must never make up
                   a date, an amount, a day count or a weekday.

Worked-out values allowed when correct (owner decision, 21 Nov 2026):
    today          today's date, and today's weekday when the reply names it ("Today is Monday")
    sums           the sum of two or more amounts in memory ("₹26,150 in total this week")
    day counts     "in 5 days", "2 days late", "3 days before the exam": the count must match the dates in memory
    weekdays       a weekday written next to a date ("Saturday, 12 December") must be that date's real weekday
A wrong sum, day count or weekday counts as invented and fails the task.

Facts are typed so that different wordings count: {"date": "2026-12-12"} matches "12 December", "Dec 12", "12/12" and
so on; {"amount": 1860} matches "₹1,860", "Rs 1860" or "1860 rupees"; {"time": "18:00"} matches "6 pm" or "18:00";
{"text": [...]} matches any listed phrase as whole words; {"any_of": [...], "min": n} needs n of the facts.
"""

import re
from datetime import date, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
# Marker fix, 22 Nov (CRITERIA.md: S3 re-marked, originals kept): a date's day and month must be on the same line, and
# a month ending a sentence is not joined to a numbered list item after it. In S3, "15 December.\n2. Pay" was read as
# 2 December.
DATE_PATTERNS = [
    re.compile(rf"\b{DAY}(?:[ \t]+of)?[ \t]+({MONTH_PATTERN})\b\.?", re.I),
    re.compile(rf"\b({MONTH_PATTERN})(?:\.(?=[ \t]+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\.(?:\s|$))))?[ \t]+{DAY}\b", re.I),
    re.compile(r"\b(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?\b"),
]
# Marker fix, 22 Nov: a day of the asked period that the reply says is empty ("Tuesday 15 December: no appointments")
# is not an invented date.
EMPTY_DAY = re.compile(r"^[\s:,\-–]*\(?\s*(nothing|no\b|none|free|empty|clear)", re.I)
AMOUNT_PATTERNS = [
    re.compile(r"(?:₹|\brs\.?|\binr)\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)(\s*(?:lakhs?|lacs?))?", re.I),
    # A number written before the word: "1860 rupees", "1.95 lakh rupees". "Rs" after a number is not read, so the
    # "12" in "Dec 12, Rs 1860" stays a date.
    re.compile(r"\b(\d(?:[\d,]*\d)?(?:\.\d+)?)(\s*(?:lakhs?|lacs?))?\s*rupees\b", re.I),
]
CLASH_WORDS = re.compile(r"\b(clash(es|ing)?|overlap(s|ping)?|conflict(s|ing)?|same time|double[- ]booked|at the same time)\b", re.I)
ABSTAIN_PHRASES = re.compile(
    r"(i (do not|don't) (have|know|see|find)|you (haven't|have not|didn't|did not) (told|tell|mentioned|mention|saved|save|given|added)"
    r"|not in (your|my) (memory|notes|records)|no (record|information|details?|note|entry) (of|about|for|on)"
    r"|(isn't|is not|wasn't|was not|aren't) (in|saved|stored|recorded)|nothing (saved|stored|noted|recorded) (about|for)"
    r"|i (can't|cannot) (find|see|tell)|i'?m not sure|no way (for me )?to know"
    # Marker change after run (e3), owner's decision: the specified partial-match answer ("I have your Goa trip …,
    # but nothing about flight") counts as saying it isn't in memory. Run (e3) is re-marked next to its original.
    r"|but nothing about|nothing (saved )?about (a|an|the|any|your)?\s*\w+|but no \w+ (is |was )?(saved|date|time|details?))",
    re.I,
)
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_PATTERN = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b\.?", re.I)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
                "thirty": 30}
DAY_COUNT = re.compile(r"\b(\d{1,3}|" + "|".join(NUMBER_WORDS) + r")\s+(days?|weeks?)\b", re.I)
FORWARD_BEFORE = re.compile(r"\b(in|after|within)\s+(just\s+|only\s+|another\s+)?$", re.I)
FORWARD_AFTER = re.compile(r"^\s*(from (now|today)|away|left|to go|remaining|later)\b", re.I)
BACKWARD_AFTER = re.compile(r"^\s*(ago|late|overdue|back|since|past)\b", re.I)
GAP_ONLY = re.compile(r"[\s,()\-–]*")


def _year_for(month: int, today: date) -> int:
    """Dates without a year are read as the nearest one to today."""
    candidates = [date(today.year + offset, month, 1) for offset in (-1, 0, 1)]
    return min(candidates, key=lambda d: abs((d - today).days)).year


def find_dates(text: str, today: date) -> Set[Tuple[int, int]]:
    """Every (day, month) mentioned in the text."""
    found: Set[Tuple[int, int]] = set()
    for match in DATE_PATTERNS[0].finditer(text):
        found.add((int(match.group(1)), MONTHS[match.group(2).lower()]))
    for match in DATE_PATTERNS[1].finditer(text):
        found.add((int(match.group(2)), MONTHS[match.group(1).lower()]))
    for match in DATE_PATTERNS[2].finditer(text):
        day, month = int(match.group(1)), int(match.group(2))
        if 1 <= day <= 31 and 1 <= month <= 12 and not _looks_like_time(text, match):
            found.add((day, month))
    return {(d, m) for d, m in found if 1 <= d <= 31}


def _looks_like_time(text: str, match: re.Match) -> bool:
    if match.group(0).count(".") == 1 and match.group(3) is None:
        after = text[match.end():match.end() + 4].lower()
        return after.strip().startswith(("am", "pm", "a.m", "p.m"))
    return False


def find_amounts(text: str) -> Set[int]:
    """Every rupee amount mentioned with ₹, Rs, INR or 'rupees'. '1.95 lakh' is 195000."""
    found: Set[int] = set()
    for pattern in AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1).replace(",", "").rstrip(".")
            if not raw or not re.fullmatch(r"\d+(\.\d+)?", raw):
                continue
            value = float(raw)
            if match.group(2):
                value *= 100_000
            found.add(round(value))
    return found


def time_variants(hhmm: str) -> List[str]:
    hour, minute = (int(x) for x in hhmm.split(":"))
    twelve = hour % 12 or 12
    suffix = "am" if hour < 12 else "pm"
    variants = {f"{hour:02d}:{minute:02d}", f"{hour}:{minute:02d}", f"{twelve}:{minute:02d} {suffix}", f"{twelve}:{minute:02d}{suffix}",
                f"{twelve}.{minute:02d} {suffix}", f"{twelve}.{minute:02d}{suffix}"}
    if minute == 0:
        variants |= {f"{twelve} {suffix}", f"{twelve}{suffix}", f"{twelve} o'clock"}
    return sorted(variants)


def _position(text: str, fact: Dict[str, Any], today: date) -> Optional[int]:
    """Index of the first place the fact appears, or None."""
    lowered = text.lower()
    if "text" in fact:
        positions = [m.start() for phrase in fact["text"] for m in [re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", lowered)] if m]
        return min(positions) if positions else None
    if "date" in fact:
        target = date.fromisoformat(fact["date"])
        positions = []
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(text):
                if (target.day, target.month) in find_dates(match.group(0), today):
                    positions.append(match.start())
        return min(positions) if positions else None
    if "amount" in fact:
        positions = []
        for pattern in AMOUNT_PATTERNS:
            for match in pattern.finditer(text):
                if fact["amount"] in find_amounts(match.group(0)):
                    positions.append(match.start())
        return min(positions) if positions else None
    if "time" in fact:
        positions = []
        for variant in time_variants(fact["time"]):
            match = re.search(rf"(?<![\d:.]){re.escape(variant.lower())}(?![\d])", lowered.replace("p.m.", "pm").replace("a.m.", "am"))
            if match:
                positions.append(match.start())
        return min(positions) if positions else None
    if "any_of" in fact:
        hits = [p for p in (_position(text, sub, today) for sub in fact["any_of"]) if p is not None]
        return min(hits) if len(hits) >= fact.get("min", 1) else None
    raise ValueError(f"Unknown fact: {fact}")


def contains(text: str, fact: Dict[str, Any], today: date) -> bool:
    return _position(text, fact, today) is not None


def allowed_facts(task: Dict[str, Any]) -> Tuple[Set[Tuple[int, int]], Set[int]]:
    """Dates and amounts the reply may mention: the user's memory, the task's allow list, and today."""
    today = date.fromisoformat(task["today"])
    dates = {(today.day, today.month)}
    amounts: Set[int] = set()
    for entry in task["memory"]:
        item = entry.get("item", {})
        for key in ("date", "end_date"):
            if item.get(key):
                d = date.fromisoformat(item[key])
                dates.add((d.day, d.month))
        if item.get("amount") is not None:
            amounts.add(item["amount"])
        if item.get("day_of_month"):
            dates |= {(item["day_of_month"], month) for month in range(1, 13)}
        amounts |= set(item.get("other_amounts", []))
    for fact in task.get("allow", []):
        if "date" in fact:
            d = date.fromisoformat(fact["date"])
            dates.add((d.day, d.month))
        if "amount" in fact:
            amounts.add(fact["amount"])
    return dates, amounts


def _full_date(day: int, month: int, today: date) -> Optional[date]:
    try:
        return date(_year_for(month, today), month, day)
    except ValueError:
        return None


def memory_dates(task: Dict[str, Any]) -> Set[date]:
    """Calendar dates the memory points to, used to check day counts. Monthly items count in last, this and next
    month; weekly routines count in the four weeks either side of today."""
    today = date.fromisoformat(task["today"])
    dates, _ = allowed_facts(task)
    full = {d for d in (_full_date(day, month, today) for day, month in dates) if d}
    for entry in task["memory"]:
        item = entry.get("item", {})
        if item.get("day_of_month"):
            for offset in (-1, 0, 1):
                index = today.month - 1 + offset
                year, month = today.year + index // 12, index % 12 + 1
                try:
                    full.add(date(year, month, item["day_of_month"]))
                except ValueError:
                    pass
        for name in item.get("weekdays", []):
            target = WEEKDAYS.index(name.lower())
            full |= {today + timedelta(days=k) for k in range(-28, 29) if (today + timedelta(days=k)).weekday() == target}
    return full


def allowed_sums(amounts: Set[int]) -> Set[int]:
    """Every sum of two or more different amounts in memory."""
    values = sorted(amounts)
    return {sum(combo) for size in range(2, len(values) + 1) for combo in combinations(values, size)}


def _day_count_value(match: re.Match) -> int:
    raw = match.group(1).lower()
    number = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
    return number * (7 if match.group(2).lower().startswith("week") else 1)


def invented_day_counts(reply: str, task: Dict[str, Any]) -> List[str]:
    """Day counts that don't match the memory. 'In N days' must land on a date in memory, 'N days late' must point back
    to one, and any other 'N days' must be the gap between two dates in memory (today included). A count the user said
    themselves, in memory ("for 5 days") or in the question ("in the next 7 days"), is allowed."""
    today = date.fromisoformat(task["today"])
    dates = memory_dates(task) | {today}
    gaps = {abs((a - b).days) for a in dates for b in dates}
    user_text = [e.get("said", "") for e in task["memory"]] + [task.get("question", "")]
    said_counts = {_day_count_value(m) for text in user_text for m in DAY_COUNT.finditer(text)}
    wrong = []
    for match in DAY_COUNT.finditer(reply):
        n = _day_count_value(match)
        before, after = reply[max(0, match.start() - 20):match.start()], reply[match.end():match.end() + 15]
        if n in said_counts:
            continue
        if FORWARD_BEFORE.search(before) or FORWARD_AFTER.search(after):
            ok = today + timedelta(days=n) in dates
        elif BACKWARD_AFTER.search(after):
            ok = today - timedelta(days=n) in dates
        else:
            ok = n in gaps
        if not ok:
            wrong.append(match.group(0))
    return wrong


def _weekday_index(word: str) -> int:
    word = word.lower()
    return next(i for i, name in enumerate(WEEKDAYS) if name.startswith(word[:3]))


def invented_weekdays(reply: str, task: Dict[str, Any]) -> List[str]:
    """A weekday written right next to a date ("Saturday, 12 December", "12 Dec (Sat)"), or next to the word 'today',
    must be the right one. Weekdays on their own ("tuition every Thursday") are not checked."""
    today = date.fromisoformat(task["today"])
    spans = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(reply):
            found = find_dates(match.group(0), today)
            if len(found) == 1:
                full = _full_date(*next(iter(found)), today)
                if full:
                    spans.append((match.start(), match.end(), full))
    wrong = []
    for match in WEEKDAY_PATTERN.finditer(reply):
        named = _weekday_index(match.group(1))
        neighbours = []
        for start, end, full in spans:
            gap = reply[match.end():start] if start >= match.end() else reply[end:match.start()]
            if len(gap) <= 4 and GAP_ONLY.fullmatch(gap):
                neighbours.append(full)
        near_today = (re.search(r"\btoday(\s+is|,|\s*\(|\s*-)?\s*$", reply[max(0, match.start() - 12):match.start()], re.I)
                      or re.match(r"\s*\(?\s*today\b", reply[match.end():match.end() + 10], re.I))
        if neighbours:
            if all(full.weekday() != named for full in neighbours):
                wrong.append(f"{match.group(1)} {neighbours[0].day}/{neighbours[0].month}")
        elif near_today and today.weekday() != named:
            wrong.append(f"{match.group(1)} (today)")
    return wrong


def asked_period(question: str, today: date) -> Optional[Tuple[date, date]]:
    """The days a question asks about, for the empty-day rule: 'this week', 'next week', 'next N days', 'my week'."""
    q = question.lower()
    m = re.search(r"\bnext\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+days?\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        return today, today + timedelta(days=n)
    if re.search(r"\bnext week\b", q):
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if re.search(r"\b(this|my) week\b|\bplan for (the|this) week\b", q):
        return today, today + timedelta(days=6 - today.weekday())
    # Added 22 Nov (S4b), for the period-end rule; the empty-day rule uses the same periods
    m = re.search(r"\b(?:coming|following)\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+days?\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        return today, today + timedelta(days=n)
    if re.search(r"\b(this|end of (the|this))\s+month\b|\bbefore the end of\b", q):
        nxt = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
        return today, nxt - timedelta(days=1)
    if re.search(r"\bday after tomorrow\b|\bparso\b", q):
        return today + timedelta(days=2), today + timedelta(days=2)
    if re.search(r"\b(tomorrow|tmrw|kal)\b", q):
        return today + timedelta(days=1), today + timedelta(days=1)
    if re.search(r"\b(today|tonight|aaj)\b|\b(plan|my) day\b", q):
        return today, today
    return None


def period_ends(task: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Marker change, 22 Nov (S4b, CRITERIA.md): the first and last day of the period the question names are worked-out
    values, like day counts ("next 7 days" from 7 December: 7 and 14 December)."""
    today = date.fromisoformat(task["today"])
    if task.get("period"):  # dev set 3 records the period its question names (CRITERIA.md, before run (e3))
        return {(d.day, d.month) for d in (date.fromisoformat(x) for x in task["period"])}
    period = asked_period(task.get("question", ""), today)
    return {(d.day, d.month) for d in period} if period else set()


def weekly_dates(task: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Marker change, 22 Nov (S4b): the days a weekly routine in memory falls on (four weeks either side of today)."""
    today = date.fromisoformat(task["today"])
    found = set()
    for entry in task["memory"]:
        for name in entry.get("item", {}).get("weekdays", []):
            target = WEEKDAYS.index(name.lower())
            for k in range(-28, 29):
                d = today + timedelta(days=k)
                if d.weekday() == target:
                    found.add((d.day, d.month))
    return found


def empty_days_said(reply: str, task: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Days of the asked period that the reply only says are empty."""
    today = date.fromisoformat(task["today"])
    period = asked_period(task.get("question", ""), today)
    if not period:
        return set()
    found = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(reply):
            days = find_dates(match.group(0), today)
            if len(days) != 1:
                continue
            day, month = next(iter(days))
            full = _full_date(day, month, today)
            if full and period[0] <= full <= period[1] and EMPTY_DAY.match(reply[match.end():].split("\n", 1)[0]):
                found.add((day, month))
    return found


def invented_facts(reply: str, task: Dict[str, Any]) -> Dict[str, List[Any]]:
    today = date.fromisoformat(task["today"])
    dates, amounts = allowed_facts(task)
    dates = dates | empty_days_said(reply, task) | period_ends(task) | weekly_dates(task)
    extra_amounts = sorted(find_amounts(reply) - amounts)
    if extra_amounts:
        sums = allowed_sums(amounts)
        extra_amounts = [a for a in extra_amounts if a not in sums]
    return {
        "dates": sorted(f"{d}/{m}" for d, m in find_dates(reply, today) - dates),
        "amounts": extra_amounts,
        "day_counts": invented_day_counts(reply, task),
        "weekdays": invented_weekdays(reply, task),
    }


def score_task(task: Dict[str, Any], reply: str) -> Dict[str, Any]:
    """Returns the verdict and which rules failed, so every result can be checked by hand."""
    today = date.fromisoformat(task["today"])
    rules = task["rules"]
    failures: List[str] = []

    for fact in rules.get("must_include", []):
        if not contains(reply, fact, today):
            failures.append(f"missing {fact}")
    for fact in rules.get("must_not", []):
        if contains(reply, fact, today):
            failures.append(f"contains forbidden {fact}")
    order = rules.get("order", [])
    if order:
        positions = [_position(reply, fact, today) for fact in order]
        if None not in positions and positions != sorted(positions):
            failures.append("plan items out of order")
    if rules.get("flag_clash") and not CLASH_WORDS.search(reply):
        failures.append("clash not pointed out")
    if rules.get("abstain") and not ABSTAIN_PHRASES.search(reply.replace("’", "'")):
        failures.append("did not say it isn't in memory")
    # Rules added for development set 4 (written before S4d/S4e): an answer may not both say it has nothing and give
    # the answer, and never calls the person "the user"
    if rules.get("consistent") and ABSTAIN_PHRASES.search(reply.replace("’", "'")) and rules.get("must_include") \
            and all(contains(reply, f, today) for f in rules["must_include"]):
        failures.append("says it isn't saved and also gives the answer")
    if rules.get("no_the_user") and re.search(r"\bthe user('s)?\b", reply, re.I):
        failures.append("calls the person 'the user'")
    invented = {kind: values for kind, values in invented_facts(reply, task).items() if values}
    if invented:
        failures.append(f"invented {invented}")

    return {"id": task["id"], "passed": not failures, "failures": failures, "invented": bool(invented)}

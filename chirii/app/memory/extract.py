"""
Turns one sentence the user tells Chirii into a structured memory item: what kind of thing it is, and its dates, times,
amounts and repeats.

This is plain code, not the language model. Dates and amounts are what a memory assistant must never get wrong, and
code reads "₹1,95,000", "the 5th of every month" or "between 9 and 11 am" the same way every time. The model is used
later only to word answers. Sentences the rules don't understand are still kept, as notes, so nothing the user says
is lost.

Year rule: a date without a year is read as the one nearest to the day it was said (said in December, "8 January" is
next January and "25 November" is last month).
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_M = "|".join(sorted(MONTHS, key=len, reverse=True))
_D = r"(\d{1,2})(?:st|nd|rd|th)?"
_WD = "|".join(WEEKDAYS)

NEGATED = re.compile(rf"\bnot\s+(?:on\s+)?(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)?(?:\s+(?:of\s+)?(?:{_M})\b)?", re.I)
DATE_RANGE = re.compile(rf"\bfrom\s+{_D}(?:\s+(?:of\s+)?({_M}))?\s+(?:to|till|until|-|–)\s+{_D}\s+(?:of\s+)?({_M})\b", re.I)
DAY_MONTH = re.compile(rf"\b{_D}(?:\s+of)?\s+({_M})\b\.?", re.I)
MONTH_DAY = re.compile(rf"\b({_M})\.?\s+{_D}\b", re.I)
MONTHLY = re.compile(rf"(?:\bon\s+)?\bthe\s+{_D}\s+of\s+(?:every|each)\s+month\b|\bevery\s+month\s+on\s+the\s+{_D}\b|\bmonthly\s+on\s+the\s+{_D}\b", re.I)
WEEKLY = re.compile(rf"\bevery\s+((?:{_WD})(?:\s*,\s*(?:{_WD}))*(?:\s*(?:,\s*)?and\s+(?:{_WD}))?)\b", re.I)
NEXT_WEEKDAY = re.compile(rf"\b(?:next|this|on)\s+({_WD})\b", re.I)
RELATIVE = {"today": 0, "tonight": 0, "tomorrow": 1, "day after tomorrow": 2, "yesterday": -1}

_T = r"(\d{1,2})(?:[:.](\d{2}))?"
_AP = r"(a\.?m\.?|p\.?m\.?)"
TIME_BETWEEN = re.compile(rf"\bbetween\s+{_T}\s*{_AP}?\s+and\s+{_T}\s*{_AP}", re.I)
TIME_RANGE = re.compile(rf"\b(?:from\s+)?{_T}\s*{_AP}?\s*(?:to|till|until|-|–)\s*{_T}\s*{_AP}", re.I)
TIME_ONE = re.compile(rf"\b{_T}\s*{_AP}(?![a-z])", re.I)
ENDS_AT = re.compile(rf"\bends?\s+(?:at\s+)?{_T}\s*{_AP}", re.I)
NOON = re.compile(r"\b(noon|midday)\b", re.I)

AMOUNT_BEFORE = re.compile(r"(?:₹|\brs\.?|\binr)\s*(\d(?:[\d,]*\d)?(?:\.\d+)?)(\s*(?:lakhs?|lacs?))?", re.I)
AMOUNT_AFTER = re.compile(r"\b(\d(?:[\d,]*\d)?(?:\.\d+)?)(\s*(?:lakhs?|lacs?))?\s*rupees\b", re.I)

DOSE_TIMES = re.compile(r"\b(before breakfast|after breakfast|before lunch|after lunch|before dinner|after dinner|at night|"
                        r"in the morning|in the evening|at bedtime|night|morning|evening)\b", re.I)

PAID = re.compile(r"(?<!\bbe )(?<!\bto )\bpaid\b", re.I)
MOVED = re.compile(r"^\s*correction\b|\b(moved|postponed|rescheduled|extended|preponed|shifted|changed)\b|\bnow\s+(?:on|at)\b", re.I)
CANCELLED = re.compile(r"\b(cancelled|canceled|called off)\b", re.I)
SAME_TIME = re.compile(r"\bsame time\b", re.I)

KIND_RULES: List[Tuple[str, re.Pattern]] = [
    ("medicine", re.compile(r"\b\d+\s*(mg|ml|mcg|iu)\b|\b(tablets?|capsules?|syrup|insulin|vitamin|eye ?drops)\b", re.I)),
    ("receivable", re.compile(r"\bowes?\b|\bon credit\b", re.I)),
    ("income", re.compile(r"\b(credited|salary|pension)\b", re.I)),
    ("exam", re.compile(r"\bexams?\b", re.I)),
    ("travel", re.compile(r"\b(train|flight|bus)\b|\bpnr\b", re.I)),
    ("shift", re.compile(r"\bshift\b", re.I)),
    ("important_date", re.compile(r"\b(birthday|anniversary|engagement)\b", re.I)),
]
BILL_WORDS = re.compile(r"\b(bill|fees?|emi|premium|rent|pay|paid|due|insurance|tax|subscription|recharge)\b", re.I)
TASK_WORDS = re.compile(r"\b(must|need to|have to|has to|last date|deposit|collect|submit|update|form|pickup|will be ready|return)\b"
                        r"|\bby\s+\d|\bbefore\b", re.I)
APPOINTMENT_WORDS = re.compile(r"\b(appointment|check-?up|doctor|dentist|visit|meeting)\b", re.I)
PAST_WORDS = re.compile(r"\b(was|were|submitted|delivered|did|went|got|received)\b", re.I)
NOTE_START = re.compile(r"^\s*note\b|\bsaid\s*:|\blist\b", re.I)


@dataclass
class Item:
    kind: str
    text: str
    said_on: str
    date: Optional[str] = None
    end_date: Optional[str] = None
    time: Optional[str] = None
    end_time: Optional[str] = None
    amount: Optional[int] = None
    day_of_month: Optional[int] = None
    weekdays: List[str] = field(default_factory=list)
    dose_times: List[str] = field(default_factory=list)
    other_dates: List[str] = field(default_factory=list)
    other_times: List[str] = field(default_factory=list)
    other_amounts: List[int] = field(default_factory=list)
    action: Optional[str] = None      # updates only: paid, moved or cancelled
    keep_time: bool = False           # updates only: "same time"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _year_for(month: int, day: int, said_on: date) -> Optional[date]:
    candidates = []
    for year in (said_on.year - 1, said_on.year, said_on.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            pass
    return min(candidates, key=lambda d: abs((d - said_on).days)) if candidates else None


def _hhmm(hour: str, minute: Optional[str], ampm: Optional[str]) -> Optional[str]:
    h, m = int(hour), int(minute or 0)
    if ampm:
        pm = ampm.lower().startswith("p")
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if pm else 0)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return f"{h:02d}:{m:02d}"


def find_amounts(text: str) -> List[Tuple[int, int]]:
    """(position, amount in rupees) for every amount written with ₹, Rs, INR or 'rupees'."""
    found = {}
    for pattern in (AMOUNT_BEFORE, AMOUNT_AFTER):
        for m in pattern.finditer(text):
            raw = m.group(1).replace(",", "")
            if not re.fullmatch(r"\d+(\.\d+)?", raw):
                continue
            value = float(raw) * (100_000 if m.group(2) else 1)
            found.setdefault(m.start(), round(value))
    return sorted(found.items())


def find_times(text: str) -> Tuple[List[Tuple[int, str]], Optional[str]]:
    """(position, HH:MM) for each start time, and an end time if the sentence gives one."""
    starts: List[Tuple[int, str]] = []
    end: Optional[str] = None
    used: List[Tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        return all(m.end() <= a or m.start() >= b for a, b in used)

    for pattern in (TIME_BETWEEN, TIME_RANGE):
        for m in pattern.finditer(text):
            if not free(m):
                continue
            h1, m1, ap1, h2, m2, ap2 = m.groups()
            first = _hhmm(h1, m1, ap1 or ap2)
            second = _hhmm(h2, m2, ap2)
            if first and second:
                starts.append((m.start(), first))
                end = end or second
                used.append(m.span())
    for m in ENDS_AT.finditer(text):
        t = _hhmm(*m.groups())
        if t and free(m):
            end = t
            used.append(m.span())
    for m in TIME_ONE.finditer(text):
        if free(m):
            t = _hhmm(*m.groups())
            if t:
                starts.append((m.start(), t))
                used.append(m.span())
    for m in NOON.finditer(text):
        starts.append((m.start(), "12:00"))
    return sorted(starts), end


def find_dates(text: str, said_on: date) -> Tuple[List[Tuple[int, date]], Optional[date]]:
    """(position, date) for each date, and an end date for 'from 14 to 18 December'."""
    found: Dict[int, date] = {}
    end: Optional[date] = None
    used: List[Tuple[int, int]] = []
    for m in DATE_RANGE.finditer(text):
        d1, mon1, d2, mon2 = m.groups()
        month2 = MONTHS[mon2.lower()]
        first = _year_for(MONTHS[mon1.lower()] if mon1 else month2, int(d1), said_on)
        last = _year_for(month2, int(d2), said_on)
        if first and last:
            found[m.start()] = first
            end = last
            used.append(m.span())
    for pattern, order in ((DAY_MONTH, "dm"), (MONTH_DAY, "md")):
        for m in pattern.finditer(text):
            if any(a <= m.start() < b for a, b in used):
                continue
            day, mon = (m.group(1), m.group(2)) if order == "dm" else (m.group(2), m.group(1))
            d = _year_for(MONTHS[mon.lower()], int(day), said_on)
            if d:
                found.setdefault(m.start(), d)
    if not found:
        lowered = text.lower()
        for phrase in sorted(RELATIVE, key=len, reverse=True):
            m = re.search(rf"\b{phrase}\b(?!'s)", lowered)
            if m:
                found[m.start()] = said_on + timedelta(days=RELATIVE[phrase])
                break
    if not found and not WEEKLY.search(text):
        m = NEXT_WEEKDAY.search(text)
        if m:
            target = WEEKDAYS.index(m.group(1).capitalize())
            ahead = (target - said_on.weekday()) % 7 or 7
            found[m.start()] = said_on + timedelta(days=ahead)
    return sorted(found.items()), end


def classify(text: str, has_amount: bool, has_date: bool, has_time: bool, weekly: bool) -> str:
    if NOTE_START.search(text):
        return "note"
    if PAID.search(text) or MOVED.search(text) or CANCELLED.search(text):
        return "update"
    for kind, pattern in KIND_RULES:
        if pattern.search(text):
            return kind
    if has_amount and BILL_WORDS.search(text):
        return "bill"
    if weekly:
        return "routine"
    if TASK_WORDS.search(text):
        return "task"
    if has_time or APPOINTMENT_WORDS.search(text):
        return "appointment"
    if PAST_WORDS.search(text):
        return "note"
    if has_date:
        return "important_date"
    return "note"


def extract(text: str, said_on: date) -> Item:
    """One sentence → one structured item."""
    clean = NEGATED.sub(" ", text)
    amounts = find_amounts(clean)
    times, end_time = find_times(clean)
    dates, end_date = find_dates(clean, said_on)
    monthly = MONTHLY.search(clean)
    weekly = WEEKLY.search(clean)
    weekdays = [w.capitalize() for w in re.findall(_WD, weekly.group(1), re.I)] if weekly else []

    kind = classify(text, bool(amounts), bool(dates or monthly), bool(times), bool(weekdays))
    item = Item(kind=kind, text=text.strip(), said_on=said_on.isoformat())
    if dates:
        item.date = dates[0][1].isoformat()
        item.other_dates = [d.isoformat() for _, d in dates[1:]]
    if end_date:
        item.end_date = end_date.isoformat()
    if times:
        item.time = times[0][1]
        item.other_times = [t for _, t in times[1:]]
    item.end_time = end_time
    if amounts:
        item.amount = amounts[0][1]
        item.other_amounts = [a for _, a in amounts[1:]]
    if monthly:
        item.day_of_month = int(next(g for g in monthly.groups() if g))
    item.weekdays = weekdays
    if kind == "medicine":
        item.dose_times = [m.group(1).lower() for m in DOSE_TIMES.finditer(text)]
    if kind == "update":
        item.action = "cancelled" if CANCELLED.search(text) else "paid" if PAID.search(text) else "moved"
        item.keep_time = bool(SAME_TIME.search(text))
    return item

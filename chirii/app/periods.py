"""
Period reader (S4c): the days a question is about, worked out by code.

    explicit dates          "on 3 March", "from 3 to 5 March", "between 3 March and 9 March"
    relative days           today, tonight, tomorrow, day after tomorrow, aaj, kal, parso
    counts                  "next 5 days", "coming 2 weeks", "this fortnight"
    weeks                   "this week", "my week", "rest of the week", "next week"
    weekends                "this weekend", "the weekend", "next weekend", "before the weekend" (today to Friday)
    months                  "this month", "end of the month", "before the end of the month", "next month"
    weekday names           "my Saturday", "on Friday", "this Friday", "next Monday", "Friday ko"
    saved events            "before the Goa trip", "until Kamala comes", "before we leave for Madurai": the event's
                            date is looked up in memory; "before" ends the day before it

A weekday or month name is never an item name. Returns a Period with its dates and the words to show in a header,
or None when the question names no period.
"""

import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from chirii.app.formatting import make_title, say_date
from chirii.eval.scoring import find_dates

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
SHORT = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "fourteen": 14, "fifteen": 15, "thirty": 30, "a": 1}
EVENT_WORDS = {"the", "a", "an", "my", "our", "we", "i", "go", "going", "leave", "leaving", "for", "to", "home", "comes", "come",
               "coming", "arrives", "arrive", "arriving", "starts", "start", "begins", "is", "are", "it", "they", "he", "she",
               "get", "back", "there", "of", "trip", "on", "at", "in"}


@dataclass
class Period:
    start: date
    end: date
    words: str                                  # for a header: "this weekend", "before the Goa trip"
    kind: str = "range"                         # day, range, weekday, event
    event: Optional[Dict[str, Any]] = field(default=None)

    def header(self) -> str:
        if self.kind == "event" and self.event:
            return f"before {self.event.get('title') or make_title(self.event['text'])} on {short(date.fromisoformat(self.event['date']))}"
        if self.start == self.end:
            return f"on {short(self.start)}"
        return f"{self.words} ({short(self.start)} to {short(self.end)})" if self.words else f"from {short(self.start)} to {short(self.end)}"


def short(d: date) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def month_end(d: date) -> date:
    return d.replace(day=monthrange(d.year, d.month)[1])


def next_weekday(today: date, target: int, skip_today: bool = False) -> date:
    ahead = (target - today.weekday()) % 7
    if ahead == 0 and skip_today:
        ahead = 7
    return today + timedelta(days=ahead)


def weekday_in(question: str) -> Optional[int]:
    q = question.lower()
    for i, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}s?\b", q):
            return i
    for name, i in SHORT.items():
        if re.search(rf"\b(my|on|this|next|every)\s+{name}\b", q):
            return i
    return None


def read_period(question: str, today: date, items: Optional[List[Dict[str, Any]]] = None) -> Optional[Period]:
    q = question.lower().replace("’", "'")
    # counts of days and weeks
    m = re.search(r"\b(?:next|coming|following)\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+(days?|weeks?)\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        days = n * (7 if m.group(2).startswith("week") else 1)
        return Period(today, today + timedelta(days=days), f"the next {m.group(1)} {m.group(2)}")
    if re.search(r"\b(this|next|coming)?\s*fortnight\b", q):
        return Period(today, today + timedelta(days=14), "the next two weeks")
    # weekends
    saturday = next_weekday(today, 5)
    if today.weekday() == 6:                       # on a Sunday, "this weekend" is today
        saturday = today - timedelta(days=1)
    if re.search(r"\bbefore the weekend\b", q):
        friday = next_weekday(today, 4)
        return Period(today, max(today, friday), "before the weekend")
    if re.search(r"\bnext weekend\b", q):
        sat = saturday + timedelta(days=7)
        return Period(sat, sat + timedelta(days=1), "next weekend")
    if re.search(r"\b(this|the|coming)?\s*weekend\b", q):
        return Period(max(today, saturday), saturday + timedelta(days=1), "this weekend")
    # weeks
    if re.search(r"\bnext week\b", q):
        start = today + timedelta(days=7 - today.weekday())
        return Period(start, start + timedelta(days=6), "next week")
    if re.search(r"\b(this|my|the|rest of the)\s+week\b|\bweekly\b", q):
        return Period(today, today + timedelta(days=6 - today.weekday()), "this week")
    # months
    if re.search(r"\bnext month\b", q):
        first = month_end(today) + timedelta(days=1)
        return Period(first, month_end(first), "next month")
    if re.search(r"\b(this|the|rest of the)\s+month\b|\bend of (the|this) month\b|\bmonth[- ]end\b", q):
        return Period(today, month_end(today), "this month")
    # relative days
    if re.search(r"\bday after tomorrow\b|\bparso\b", q):
        return Period(today + timedelta(days=2), today + timedelta(days=2), "the day after tomorrow", "day")
    if re.search(r"\b(tomorrow|tmrw|tmr|kal)\b", q):
        return Period(today + timedelta(days=1), today + timedelta(days=1), "tomorrow", "day")
    if re.search(r"\b(today|tonight|aaj)\b|\bmy day\b|\bplan (for )?the day\b", q):
        return Period(today, today, "today", "day")
    # weekday names
    wd = weekday_in(question)
    if wd is not None:
        day = next_weekday(today, wd, skip_today=bool(re.search(rf"\bnext\s+{WEEKDAYS[wd]}\b", q)))
        return Period(day, day, WEEKDAYS[wd].capitalize(), "weekday")
    # explicit ranges where only the second date names the month: "from 8 to 10 March", "8-10 March"
    m = re.search(r"\b(?:from\s+|between\s+)?(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|till|until|and|-|–)\s*(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\b", q)
    if m:
        last = sorted(find_dates(f"{m.group(2)} {m.group(3)}", today))
        if last:
            day2, month = last[0]
            year = today.year if date(today.year, month, 1) >= today.replace(day=1) - timedelta(days=31) else today.year + 1
            try:
                start, end = date(year, month, int(m.group(1))), date(year, month, day2)
                if start <= end:
                    return Period(start, end, "", "range")
            except ValueError:
                pass
    # explicit dates or ranges
    found = sorted(find_dates(question, today))
    if found:
        full = []
        for day, month in found:
            for year in (today.year, today.year + 1):
                try:
                    d = date(year, month, day)
                except ValueError:
                    continue
                if d >= today - timedelta(days=31):
                    full.append(d)
                    break
        if full:
            return Period(min(full), max(full), "", "day" if len(full) == 1 else "range")
    # a saved event
    m = re.search(r"\b(before|until|till|by the time)\s+(.+?)[?.!]*$", q)
    if m and items:
        event = find_event(m.group(2), today, items)
        if event is not None:
            when = date.fromisoformat(event["date"])
            end = when - timedelta(days=1) if m.group(1) != "by the time" else when
            if end >= today:
                return Period(today, end, f"before {event.get('title') or make_title(event['text'])}", "event", event)
    return None


def find_event(words: str, today: date, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The saved item an event phrase refers to: the dated, upcoming item sharing the most words with it."""
    from chirii.app.memory.store import keywords  # local import: store imports formatting, which periods also uses
    wanted = keywords(" ".join(w for w in re.findall(r"[a-z']+", words.lower()) if w not in EVENT_WORDS))
    if not wanted:
        return None
    best, best_score = None, 0
    for item in items:
        if not item.get("date") or item.get("kind") == "update":
            continue
        if date.fromisoformat(item.get("end_date") or item["date"]) < today:
            continue
        score = len(wanted & keywords(item["text"] + " " + (item.get("title") or "")))
        if score > best_score or (score == best_score and score and item["date"] < best["date"]):
            best, best_score = item, score
    return best if best_score else None

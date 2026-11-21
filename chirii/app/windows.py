"""
The days a question asks about ("this week", "tomorrow", "next 5 days", "before the end of this month", "on 3 December").
Used by retrieval, the router and the planner.
"""

import re
from calendar import monthrange
from datetime import date, timedelta
from typing import Optional, Tuple

from chirii.eval.scoring import find_dates

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10, "fourteen": 14, "thirty": 30}


def question_window(question: str, today: date) -> Optional[Tuple[date, date]]:
    """The days a question asks about, if it names any."""
    q = question.lower()
    m = re.search(r"\b(?:next|coming|following)\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+days?\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)]
        return today, today + timedelta(days=n)
    if re.search(r"\b(next|coming)\s+week\b", q):
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if re.search(r"\b(this|my)\s+week\b|\bweek\b.*\bplan\b|\bplan\b.*\bweek\b|\bweekly\b", q):
        return today, today + timedelta(days=6 - today.weekday())
    if re.search(r"\b(this|end of (the|this))\s+month\b|\bbefore the end of\b", q):
        return today, today.replace(day=monthrange(today.year, today.month)[1])
    if re.search(r"\bday after tomorrow\b|\bparso\b", q):
        return today + timedelta(days=2), today + timedelta(days=2)
    if re.search(r"\b(tomorrow|tmrw|kal)\b", q):
        return today + timedelta(days=1), today + timedelta(days=1)
    if re.search(r"\b(today|tonight|aaj)\b|\bplan my day\b|\bmy day\b", q):
        return today, today
    if re.search(r"\b(upcoming|coming up|due soon|soon)\b", q):
        return today, today + timedelta(days=7)
    named = sorted(find_dates(question, today))
    if named:
        day, month = named[0]
        for year in (today.year, today.year + 1):
            try:
                d = date(year, month, day)
            except ValueError:
                continue
            if d >= today - timedelta(days=31):
                return d, d
    return None

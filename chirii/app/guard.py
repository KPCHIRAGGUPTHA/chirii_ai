"""
Answer fact-check guard: before an answer is shown, every date, rupee amount, day count and weekday in it must come
from the memory lines the model was given, or be worked out correctly from them (today, sums, day counts, weekdays).

It uses the S1 marker's own functions (chirii/eval/scoring.py), so the app and the eval agree on what "invented" means.
One rule is stricter than the marker: today's date may appear only when the answer is talking about today, or a
memory falls on today. The marker allows today's date anywhere; the slice demo showed Qwen putting today's date on a
surgery that was on 3 December, which the marker's rule would not catch.
"""

import re
from datetime import date
from typing import Any, Dict, List

from chirii.eval.scoring import DATE_PATTERNS, EMPTY_DAY, asked_period, find_dates, invented_facts

TODAY_WORDS = re.compile(r"\b(today|tonight|aaj)\b", re.I)


def facts_as_task(items: List[Dict[str, Any]], histories: Dict[int, List[Dict[str, Any]]], today: date, question: str) -> Dict[str, Any]:
    """The given memory lines in the shape the marker reads."""
    memory, allow = [], []
    for item in items:
        memory.append({"said": item["text"], "item": {
            "date": item.get("date"), "end_date": item.get("end_date"), "amount": item.get("amount"),
            "day_of_month": item.get("day_of_month"), "other_amounts": item.get("other_amounts", []),
            "weekdays": item.get("weekdays", []),
        }})
        allow += [{"date": d} for d in item.get("other_dates", [])]
        for name in ("remaining", "paid_amount"):
            if item.get(name):
                allow.append({"amount": item[name]})
        for change in histories.get(item["id"], []):  # old and new values shown as "changed from … to …"
            for value in (change.get("old_value"), change.get("new_value")):
                if value and change["field"] in ("date", "end_date"):
                    allow.append({"date": value})
                if value and change["field"] in ("amount", "paid_amount") and value.isdigit():
                    allow.append({"amount": int(value)})
    return {"today": today.isoformat(), "question": question, "memory": memory, "allow": allow}


def falls_on(item: Dict[str, Any], day: date) -> bool:
    if item.get("date"):
        first = date.fromisoformat(item["date"])
        last = date.fromisoformat(item["end_date"]) if item.get("end_date") else first
        if first <= day <= last:
            return True
    if day.isoformat() in item.get("other_dates", []) or item.get("day_of_month") == day.day:
        return True
    if day.strftime("%A") in item.get("weekdays", []):
        return True
    return item.get("kind") == "medicine" and not item.get("date")


def check_answer(answer: str, items: List[Dict[str, Any]], histories: Dict[int, List[Dict[str, Any]]], today: date,
                 question: str) -> List[str]:
    """Problems found in the answer; empty means it passed."""
    task = facts_as_task(items, histories, today, question)
    problems = [f"{kind}: {values}" for kind, values in invented_facts(answer, task).items() if values]
    period = asked_period(question, today)
    starts_today = bool(period) and period[0] == today  # S4b: "this week" and "next 7 days" start today
    if (today.day, today.month) in find_dates(answer, today) and not starts_today and not any(falls_on(i, today) for i in items):
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(answer):
                if (today.day, today.month) in find_dates(match.group(0), today):
                    nearby = answer[max(0, match.start() - 30):match.end() + 30]
                    says_empty = EMPTY_DAY.match(answer[match.end():].split("\n", 1)[0])
                    if not TODAY_WORDS.search(nearby) and not says_empty:
                        problems.append("today's date given for something not on today")
                        return problems
    return problems

"""
Marker for the general-chat tasks of development set 3 (written with the tasks, before S4c was built).

    number       the reply contains this number (commas and ₹ ignored; within 0.01)
    must_include every listed fact appears (as in scoring.py: {"text": [...]} matches any phrase as whole words)
    must_not     none of these appear
    no_live      the reply says it can't give live or up-to-date information, and gives no figure as if it were current
    unsure       the reply says it doesn't know or isn't sure
    no_numbers   the reply states no number of 3 or more digits that is not in the question (no invented population figures)
    max_words    the reply is at most this many words
    min_items    the reply has at least this many list items or numbered points
    no invention for tasks with a person's memory: any rupee amount in the reply must be in that memory, in the
                 question, or a correct sum, as in scoring.py
"""

import re
from datetime import date
from typing import Any, Dict, List

from chirii.eval.scoring import allowed_sums, contains, find_amounts

NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w])")
NO_LIVE = re.compile(r"(live|real[- ]time|up[- ]to[- ]date|current|latest|today's)\s+(information|data|prices?|updates?|news|weather|results?|scores?|rates?)"
                     r"|(can't|cannot|can not|don't|do not|unable to)\s+(check|access|see|browse|look up|get|provide|give)\b"
                     r"|no (internet|live|web) (access|connection|information)|not connected to the internet", re.I)
UNSURE = re.compile(r"\b(i'?m not sure|i am not sure|i don't know|i do not know|not certain|no reliable information|i don't have (that|this|reliable)|"
                    r"i can't (say|tell|be sure)|i cannot (say|tell)|unable to (say|find|confirm))", re.I)
ITEM_LINE = re.compile(r"^\s*(?:\d+[.)]|[-•*])\s+\S", re.M)


def numbers_in(text: str) -> List[float]:
    return [float(m.group(1).replace(",", "")) for m in NUMBER.finditer(text.replace("₹", " "))]


def score_chat(task: Dict[str, Any], reply: str) -> Dict[str, Any]:
    today = date.fromisoformat(task["today"])
    rules = task["rules"]
    failures: List[str] = []
    if "number" in rules and not any(abs(n - rules["number"]) < 0.01 for n in numbers_in(reply)):
        failures.append(f"missing the number {rules['number']}")
    for fact in rules.get("must_include", []):
        if not contains(reply, fact, today):
            failures.append(f"missing {fact}")
    for fact in rules.get("must_not", []):
        if contains(reply, fact, today):
            failures.append(f"contains forbidden {fact}")
    if rules.get("no_live") and not NO_LIVE.search(reply.replace("’", "'")):
        failures.append("did not say it has no live information")
    if rules.get("no_live") and (find_amounts(reply) or any(n >= 100 for n in numbers_in(reply))):
        failures.append("gave a figure as if it were current")
    if rules.get("unsure") and not UNSURE.search(reply.replace("’", "'")):
        failures.append("did not say it doesn't know")
    if rules.get("no_numbers") and any(n >= 100 and n not in numbers_in(task["question"]) for n in numbers_in(reply)):
        failures.append("stated a number it could not know")
    if "max_words" in rules and len(reply.split()) > rules["max_words"]:
        failures.append(f"longer than {rules['max_words']} words")
    if "min_items" in rules and len(ITEM_LINE.findall(reply)) < rules["min_items"]:
        failures.append(f"fewer than {rules['min_items']} points")
    if task.get("memory"):
        allowed = {e["item"]["amount"] for e in task["memory"] if e["item"].get("amount") is not None}
        allowed |= set(find_amounts(task["question"]))
        if "number" in rules:
            allowed.add(round(rules["number"]))
        extra = [a for a in find_amounts(reply) if a not in allowed and a not in allowed_sums(allowed)]
        if extra:
            failures.append(f"invented amounts {extra}")
    return {"id": task["id"], "passed": not failures, "failures": failures}

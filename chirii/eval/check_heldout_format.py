"""
Format check for the held-out tasks (S6). It reads `chirii/eval/heldout_tasks.jsonl` (or the file given) and reports
whether it matches what the marker reads. It never changes the file: the tasks' content is the task writer's and the
owner's (chirii/CRITERIA.md, "Held-out set: authorship, freeze and scoring").

    python -m chirii.eval.check_heldout_format [path]

Errors mean the marker or the runner cannot use a task as written. Warnings point at rules a right answer could not
pass (a needed date or amount that is in no item and not in `allow`); they are for the owner to judge.
"""

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chirii.eval.scoring import score_task

PATH = Path(__file__).with_name("heldout_tasks.jsonl")
PERSONAS = ["priya", "lakshmi", "ramesh", "sharada", "arjun"]
TYPES = ["lookup", "date_window", "plan", "unknown", "semantic", "update"]
ITEM_TYPES = {"bill", "appointment", "task", "medicine", "routine", "important_date", "exam", "travel", "shift", "income",
              "receivable", "note", "person", "profile", "update"}
ITEM_FIELDS = {"type", "title", "date", "end_date", "time", "amount", "day_of_month", "weekdays", "status"}
WEEKDAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
TASK_FIELDS = {"id", "persona", "type", "today", "question", "memory", "rules", "allow", "note", "scored"}
RULE_KEYS = {"must_include", "must_not", "order", "flag_clash", "abstain", "consistent", "no_the_user"}


def _iso(value: Any) -> bool:
    try:
        date.fromisoformat(value)
        return isinstance(value, str) and len(value) == 10
    except (TypeError, ValueError):
        return False


def fact_errors(fact: Any, where: str) -> List[str]:
    if not isinstance(fact, dict) or len(set(fact) - {"min"}) != 1:
        return [f"{where}: a fact must be one of date / amount / time / text / any_of: {fact!r}"]
    if "date" in fact:
        return [] if _iso(fact["date"]) else [f"{where}: bad date {fact['date']!r}"]
    if "amount" in fact:
        return [] if isinstance(fact["amount"], int) and fact["amount"] >= 0 else [f"{where}: amount must be whole rupees"]
    if "time" in fact:
        return [] if isinstance(fact["time"], str) and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", fact["time"]) else [f"{where}: time must be HH:MM"]
    if "text" in fact:
        ok = isinstance(fact["text"], list) and fact["text"] and all(isinstance(t, str) and t.strip() for t in fact["text"])
        return [] if ok else [f"{where}: text must be a list of phrases"]
    if "any_of" in fact:
        if not isinstance(fact["any_of"], list) or not fact["any_of"]:
            return [f"{where}: any_of must be a list of facts"]
        return [e for i, sub in enumerate(fact["any_of"]) for e in fact_errors(sub, f"{where}.any_of[{i}]")]
    return [f"{where}: unknown fact {fact!r}"]


def task_problems(task: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors, warnings = [], []
    tid = task.get("id", "?")
    missing, extra = TASK_FIELDS - set(task), set(task) - TASK_FIELDS
    if missing:
        errors.append(f"{tid}: missing fields {sorted(missing)}")
    if extra:
        warnings.append(f"{tid}: extra fields {sorted(extra)} (ignored by the marker)")
    if task.get("persona") not in PERSONAS:
        errors.append(f"{tid}: persona {task.get('persona')!r}")
    if task.get("type") not in TYPES:
        errors.append(f"{tid}: type {task.get('type')!r}")
    if not _iso(task.get("today")):
        return errors + [f"{tid}: today must be YYYY-MM-DD"], warnings
    today = date.fromisoformat(task["today"])
    if not isinstance(task.get("question"), str) or not task["question"].strip():
        errors.append(f"{tid}: empty question")
    if task.get("scored") is not True:
        errors.append(f"{tid}: scored must be true")
    memory = task.get("memory")
    if not isinstance(memory, list) or not memory:
        return errors + [f"{tid}: memory must be a non-empty list"], warnings
    said_dates, item_dates, item_amounts = [], set(), set()
    for i, entry in enumerate(memory):
        where = f"{tid}.memory[{i}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("said"), str) or not entry["said"].strip():
            errors.append(f"{where}: needs a 'said' sentence")
            continue
        if not _iso(entry.get("said_on")):
            errors.append(f"{where}: said_on must be YYYY-MM-DD")
        elif date.fromisoformat(entry["said_on"]) > today:
            errors.append(f"{where}: said_on is after today")
        else:
            said_dates.append(entry["said_on"])
        item = entry.get("item")
        if not isinstance(item, dict) or item.get("type") not in ITEM_TYPES:
            errors.append(f"{where}: item.type must be one of {sorted(ITEM_TYPES)}")
            continue
        if set(item) - ITEM_FIELDS:
            errors.append(f"{where}: unknown item fields {sorted(set(item) - ITEM_FIELDS)}")
        for key in ("date", "end_date"):
            if key in item and not _iso(item[key]):
                errors.append(f"{where}: item.{key} must be YYYY-MM-DD")
            elif key in item:
                item_dates.add(item[key])
        if "time" in item and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(item["time"])):
            errors.append(f"{where}: item.time must be HH:MM")
        if "amount" in item:
            if not isinstance(item["amount"], int):
                errors.append(f"{where}: item.amount must be whole rupees")
            else:
                item_amounts.add(item["amount"])
        if "day_of_month" in item and not (isinstance(item["day_of_month"], int) and 1 <= item["day_of_month"] <= 31):
            errors.append(f"{where}: item.day_of_month must be 1-31")
        if "weekdays" in item and not (isinstance(item["weekdays"], list) and set(item["weekdays"]) <= WEEKDAYS):
            errors.append(f"{where}: item.weekdays must be weekday names")
    if said_dates != sorted(said_dates):
        warnings.append(f"{tid}: memory is not in the order it was said")
    rules = task.get("rules")
    if not isinstance(rules, dict) or not rules:
        return errors + [f"{tid}: rules must be a non-empty object"], warnings
    if set(rules) - RULE_KEYS:
        errors.append(f"{tid}: unknown rules {sorted(set(rules) - RULE_KEYS)}")
    for key in ("must_include", "must_not", "order"):
        if key in rules:
            if not isinstance(rules[key], list):
                errors.append(f"{tid}: rules.{key} must be a list")
                continue
            for j, fact in enumerate(rules[key]):
                errors += fact_errors(fact, f"{tid}.rules.{key}[{j}]")
    for key in ("flag_clash", "abstain", "consistent", "no_the_user"):
        if key in rules and not isinstance(rules[key], bool):
            errors.append(f"{tid}: rules.{key} must be true or false")
    allow = task.get("allow")
    if not isinstance(allow, list):
        errors.append(f"{tid}: allow must be a list")
        allow = []
    for j, fact in enumerate(allow):
        if not isinstance(fact, dict) or not ({"date", "amount"} >= set(fact)) or not fact:
            errors.append(f"{tid}.allow[{j}]: only date or amount")
        else:
            errors += fact_errors(fact, f"{tid}.allow[{j}]")
    # a needed date or amount that no item and no allow entry justifies: a right answer would fail "no invented"
    allowed_dates = item_dates | {f["date"] for f in allow if isinstance(f, dict) and "date" in f} | {task["today"]}
    allowed_amounts = item_amounts | {f["amount"] for f in allow if isinstance(f, dict) and "amount" in f}
    for fact in rules.get("must_include", []) if isinstance(rules.get("must_include"), list) else []:
        if isinstance(fact, dict) and "date" in fact and fact["date"] not in allowed_dates:
            warnings.append(f"{tid}: must_include date {fact['date']} is in no item and not in allow")
        if isinstance(fact, dict) and "amount" in fact and fact["amount"] not in allowed_amounts:
            warnings.append(f"{tid}: must_include amount {fact['amount']} is in no item and not in allow "
                            "(fine only if it is a sum of saved amounts)")
    if task.get("type") == "unknown" and rules.get("abstain") is not True:
        errors.append(f"{tid}: an unknown task needs abstain: true")
    if task.get("type") == "plan" and not rules.get("order"):
        warnings.append(f"{tid}: a plan task without an order rule")
    if not errors:
        try:
            score_task(task, "")                      # the marker reads it without error
        except Exception as e:                        # noqa: BLE001 - report any marker failure as a format error
            errors.append(f"{tid}: the marker could not read it ({type(e).__name__}: {e})")
    return errors, warnings


def check(path: Path) -> Tuple[List[str], List[str]]:
    errors, warnings, tasks = [], [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not valid JSON ({e.msg})")
    if len(tasks) != 30:
        errors.append(f"{len(tasks)} tasks, not 30")
    ids = [t.get("id") for t in tasks]
    if ids != [f"H-{i:02d}" for i in range(1, len(tasks) + 1)]:
        errors.append("ids must be H-01 … H-30 in order")
    pairs = Counter((t.get("type"), t.get("persona")) for t in tasks)
    for kind in TYPES:
        for person in PERSONAS:
            if pairs[(kind, person)] != 1:
                errors.append(f"{kind} x {person}: {pairs[(kind, person)]} tasks, not 1")
    for t in tasks:
        e, w = task_problems(t)
        errors += e
        warnings += w
    return errors, warnings


def main(argv: List[str]) -> int:
    path = Path(argv[0]) if argv else PATH
    if not path.exists():
        print(f"{path} does not exist yet: copy the task writer's heldout_tasks.jsonl there first.")
        return 1
    errors, warnings = check(path)
    for e in errors:
        print("ERROR  ", e)
    for w in warnings:
        print("warning", w)
    print(f"{path.name}: {len(errors)} errors, {len(warnings)} warnings. The file was not changed.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

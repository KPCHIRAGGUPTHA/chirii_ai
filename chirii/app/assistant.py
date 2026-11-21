"""
Answering a question from the user's memory (demo slice version; S3 refines retrieval and adds the safety layer).

1. Code picks the memories to show the model: the closest matches by meaning, plus everything due in the next 14 days.
2. Code writes each memory as a fact line with its current date (and weekday), time, amount and status, so the model
   never has to work out a date or a sum itself. Items that were changed say what changed.
3. Qwen words the answer from those lines only, and says so when the answer isn't there.
"""

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List

from chirii.app.llm.client import ChatModel
from chirii.app.logging_setup import log_event
from chirii.app.memory.store import MemoryStore

SEARCH_K = 5
UPCOMING_DAYS = 14
MAX_FACTS = 15
KIND_LABELS = {"bill": "Bill", "appointment": "Appointment", "task": "To do", "medicine": "Medicine", "routine": "Routine",
               "important_date": "Important date", "exam": "Exam", "travel": "Travel", "shift": "Work shift", "income": "Income",
               "receivable": "Money owed to the user", "note": "Note", "update": "Unmatched update"}
NOT_IN_MEMORY = "I don't have that in your memory."


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


def fact_line(item: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    details = []
    if item["date"]:
        details.append(("from " if item["end_date"] else "on ") + say_date(item["date"]) + (f" to {say_date(item['end_date'])}" if item["end_date"] else ""))
    if item["day_of_month"]:
        details.append(f"on day {item['day_of_month']} of every month")
    if item["weekdays"]:
        details.append("every " + " and ".join(item["weekdays"]))
    if item["time"]:
        details.append("at " + say_time(item["time"]) + (f" until {say_time(item['end_time'])}" if item["end_time"] else ""))
    if item["dose_times"]:
        details.append("taken " + ", ".join(item["dose_times"]))
    if item["amount"] is not None:
        details.append(f"amount {say_amount(item['amount'])}")
        if item["paid_amount"]:
            details.append(f"{say_amount(item['paid_amount'])} paid so far, {say_amount(item['remaining'])} still owed")
    status = {"paid": "PAID", "cancelled": "CANCELLED", "unlinked": "not matched to an earlier item"}.get(item["status"])
    if status:
        details.append(f"status: {status}")
    changed = [f"{h['field']} changed from {h['old_value']} to {h['new_value']}" for h in history if h["field"] in ("date", "time")]
    line = f"[{KIND_LABELS.get(item['kind'], item['kind'])}] The user said: \"{item['text']}\""
    if details:
        line += " | Current details: " + "; ".join(details)
    if changed:
        line += " | Updated since: " + "; ".join(changed) + ". Use the current details."
    return line


def system_prompt(today: date) -> str:
    return (
        "You are Chirii, a personal memory assistant. Answer the user's question using only the memory facts given.\n"
        f"Today is {say_date(today.isoformat())}.\n"
        "Rules:\n"
        f"- If the facts do not contain the answer, say \"{NOT_IN_MEMORY}\" Then, if something related is in the facts, "
        "you may mention it.\n"
        "- Never make up a date, time, amount, name or place. Copy dates and amounts exactly as the facts give them.\n"
        "- When a fact has 'Current details', use those, not older values in what the user said.\n"
        "- Items marked PAID or CANCELLED are not due any more.\n"
        "- Answer briefly: one to four sentences, or a short list."
    )


@dataclass
class Answer:
    text: str
    fact_ids: List[int]
    facts: List[str] = field(default_factory=list)
    duration_ms: int = 0


class Assistant:

    def __init__(self, store: MemoryStore, model: ChatModel):
        self.store = store
        self.model = model

    def select_facts(self, question: str, today: date) -> List[Dict[str, Any]]:
        chosen: Dict[int, Dict[str, Any]] = {}
        for hit in self.store.search(question, k=SEARCH_K):
            chosen.setdefault(hit["item"]["id"], hit["item"])
        for entry in self.store.agenda(today, today + timedelta(days=UPCOMING_DAYS)):
            if len(chosen) >= MAX_FACTS:
                break
            chosen.setdefault(entry["item"]["id"], entry["item"])
        return list(chosen.values())[:MAX_FACTS]

    def ask(self, question: str, today: date) -> Answer:
        started = time.perf_counter()
        items = self.select_facts(question, today)
        facts = [fact_line(i, self.store.history(i["id"])) for i in items]
        memory = "\n".join(f"{n}. {f}" for n, f in enumerate(facts, 1)) or "(no memories saved yet)"
        messages = [{"role": "system", "content": system_prompt(today)},
                    {"role": "user", "content": f"Memory facts:\n{memory}\n\nQuestion: {question}"}]
        reply = self.model.chat(messages, max_tokens=256, purpose="answer")
        duration = round((time.perf_counter() - started) * 1000)
        log_event("answer", user=self.store.user_id, result_count=len(items), duration_ms=duration, stage="slice")
        return Answer(text=reply.text, fact_ids=[i["id"] for i in items], facts=facts, duration_ms=duration)

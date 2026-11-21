"""
Answering a question from the user's memory.

1. Code picks the memories to show the model.
   - trimmed (default): only memories relevant to the question, i.e. close matches by meaning, plus whatever falls in
     a time window the question names ("this week", "tomorrow", "next 5 days").
   - full (the slice as first built; kept as S3's baseline): the 5 closest matches plus everything due in 14 days.
2. Code writes each memory as a fact line with its current date and weekday, time, amount and status.
3. Qwen words the answer from those lines only.
4. Guard (default on): every date, amount, day count and weekday in the answer is checked against the lines
   (chirii/app/guard.py). On a failure Qwen gets one retry with only the top 3 relevant lines; if that fails too, code
   builds the answer from the matching memories itself.
5. If the answer relies on a memory the user hasn't confirmed, code says so after the answer.
"""

import re
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from chirii.app.formatting import KIND_LABELS, make_title, say_amount, say_date, say_time, when_text
from chirii.app.guard import check_answer
from chirii.app.llm.client import ChatModel
from chirii.app.logging_setup import log_event
from chirii.app.memory.store import MemoryStore, keywords
from chirii.eval.scoring import find_amounts, find_dates

__all__ = ["Assistant", "Answer", "NOT_IN_MEMORY", "fact_line", "say_amount", "say_date", "say_time", "question_window"]

NOT_IN_MEMORY = "I don't have that in your memory."
# full retrieval (baseline)
SEARCH_K = 5
UPCOMING_DAYS = 14
MAX_FACTS = 15
# trimmed retrieval: chosen from how MiniLM scores related and unrelated sentences in general, not from the eval tasks
MIN_SIMILARITY = 0.20
RELATIVE_TO_TOP = 0.60
MAX_MATCHES = 5
MAX_WINDOW = 10
RETRY_FACTS = 3

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


def fact_line(item: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    details = when_text(item)
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
    if not item.get("confirmed"):
        line += " | Not yet confirmed by the user"
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


def code_answer(items: List[Dict[str, Any]]) -> str:
    """The answer built by code when the model's answers failed the check twice: what is saved, word for word."""
    if not items:
        return NOT_IN_MEMORY
    lines = ["Here is what I have saved about that:"]
    for item in items:
        details = when_text(item)
        if item["amount"] is not None:
            details.append(say_amount(item["amount"]) + (f", {say_amount(item['remaining'])} still owed" if item["paid_amount"] else ""))
        if item["status"] in ("paid", "cancelled"):
            details.append(item["status"])
        lines.append(f"- You said: \"{item['text']}\"" + (f" Now: {'; '.join(details)}." if details else ""))
    return "\n".join(lines)


@dataclass
class Answer:
    text: str
    fact_ids: List[int]
    facts: List[str] = field(default_factory=list)
    duration_ms: int = 0
    guard: str = "off"                      # off, pass, retry (passed on retry), fallback (code-built answer)
    problems: List[str] = field(default_factory=list)
    unconfirmed: List[str] = field(default_factory=list)
    first_answer: Optional[str] = None      # the model's first answer, kept for the S3 report when the guard acted


class Assistant:

    def __init__(self, store: MemoryStore, model: ChatModel, guard: bool = True, trimmed: bool = True):
        self.store = store
        self.model = model
        self.guard = guard
        self.trimmed = trimmed

    # ------------------------------------------------------------------ choosing facts

    def select_full(self, question: str, today: date) -> List[Dict[str, Any]]:
        chosen: Dict[int, Dict[str, Any]] = {}
        for hit in self.store.search(question, k=SEARCH_K):
            chosen.setdefault(hit["item"]["id"], hit["item"])
        for entry in self.store.agenda(today, today + timedelta(days=UPCOMING_DAYS)):
            if len(chosen) >= MAX_FACTS:
                break
            chosen.setdefault(entry["item"]["id"], entry["item"])
        return list(chosen.values())[:MAX_FACTS]

    def matches(self, question: str) -> List[Dict[str, Any]]:
        hits = self.store.search(question, k=MAX_MATCHES * 2)
        if not hits:
            return []
        top = hits[0]["score"]
        return [h["item"] for h in hits if h["similarity"] >= MIN_SIMILARITY and h["score"] >= RELATIVE_TO_TOP * top][:MAX_MATCHES]

    def select_relevant(self, question: str, today: date) -> List[Dict[str, Any]]:
        chosen: Dict[int, Dict[str, Any]] = {}
        window = question_window(question, today)
        if window:
            for entry in self.store.agenda(*window)[:MAX_WINDOW]:
                chosen.setdefault(entry["item"]["id"], entry["item"])
        for item in self.matches(question):
            chosen.setdefault(item["id"], item)
        return list(chosen.values())

    # ------------------------------------------------------------------ answering

    def _ask_model(self, question: str, today: date, items: List[Dict[str, Any]], histories) -> str:
        facts = [fact_line(i, histories[i["id"]]) for i in items]
        memory = "\n".join(f"{n}. {f}" for n, f in enumerate(facts, 1)) or "(no memory facts match this question)"
        messages = [{"role": "system", "content": system_prompt(today)},
                    {"role": "user", "content": f"Memory facts:\n{memory}\n\nQuestion: {question}"}]
        return self.model.chat(messages, max_tokens=256, purpose="answer").text

    def ask(self, question: str, today: date) -> Answer:
        started = time.perf_counter()
        items = self.select_relevant(question, today) if self.trimmed else self.select_full(question, today)
        histories = {i["id"]: self.store.history(i["id"]) for i in items}
        text = self._ask_model(question, today, items, histories)
        answer = Answer(text=text, fact_ids=[i["id"] for i in items], facts=[fact_line(i, histories[i["id"]]) for i in items])

        if self.guard:
            problems = check_answer(text, items, histories, today, question)
            answer.guard = "pass"
            if problems:
                answer.first_answer, answer.problems = text, problems
                top = (self.matches(question) or items)[:RETRY_FACTS]
                top_histories = {i["id"]: self.store.history(i["id"]) for i in top}
                retry = self._ask_model(question, today, top, top_histories)
                if not check_answer(retry, top, top_histories, today, question):
                    answer.text, answer.guard = retry, "retry"
                else:
                    answer.text, answer.guard = code_answer(top), "fallback"
                items = top
            log_event("answer_guard", user=self.store.user_id, status=answer.guard, result_count=len(answer.problems), stage="answer")

        answer.unconfirmed = self.unconfirmed_used(answer.text, items, today)
        if answer.unconfirmed:
            titles = "; ".join(f"\"{t}\"" for t in answer.unconfirmed)
            answer.text += f"\n\n(This uses something you haven't confirmed yet: {titles}.)"
        answer.duration_ms = round((time.perf_counter() - started) * 1000)
        log_event("answer", user=self.store.user_id, result_count=len(items), duration_ms=answer.duration_ms, stage="answer")
        return answer

    @staticmethod
    def unconfirmed_used(text: str, items: List[Dict[str, Any]], today: date) -> List[str]:
        """Titles of unconfirmed memories the answer relies on: it repeats one of their dates or amounts, or names them."""
        used = []
        answer_dates, answer_amounts, answer_words = find_dates(text, today), find_amounts(text), keywords(text)
        for item in items:
            if item.get("confirmed"):
                continue
            title = item.get("title") or make_title(item["text"])
            dates = {(date.fromisoformat(item["date"]).day, date.fromisoformat(item["date"]).month)} if item.get("date") else set()
            title_words = keywords(title)
            named = len(title_words & answer_words) >= min(2, len(title_words)) if title_words else False
            if dates & answer_dates or (item.get("amount") is not None and item["amount"] in answer_amounts) or named:
                used.append(title)
        return used

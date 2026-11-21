"""
Question router (S4): decides how a question is answered, so the model is only asked to word answers.

    plan     "Plan my day", "my week": code builds the timeline, orders it and finds clashes (planner.py)
    window   "What's due this week?", "Which bills do I still have to pay?": code filters, counts and totals
    status   "Is the water bill paid?", "Did I take my tablet?": code reads the item's status
    lookup   "When is my dentist appointment?": retrieval, then Qwen, then the fact-check guard
    meaning  "What did the doctor say about food?": the same path as lookup

The rules run first. Qwen is asked only when no rule matches, and its reply is limited by a JSON schema to these five
names. Every question also gets its specific terms ("electricity", "LIC", "Compiler Design"), which the specific-item
check uses: if the best memory doesn't mention any of them, Chirii says it hasn't got that saved.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

from chirii.app.windows import question_window
from chirii.app.llm.client import ChatModel, LLMError
from chirii.app.logging_setup import log_event

KINDS = ["plan", "window", "status", "lookup", "meaning"]

PLAN = re.compile(r"\b(plan|planner|schedule|agenda)\b|\bmy (day|week)\b|\b(aaj|kal) ka plan\b", re.I)
STATUS = re.compile(r"^\s*(did|have|has|is|are|was|were)\b.*\b(paid|taken|took|done|submitted|collected|finished|booked|renewed|"
                    r"cleared|settled|returned|delivered)\b|^\s*did i\b", re.I)
WINDOW_WORDS = re.compile(r"\b(what|which|any|anything|due|pay|payments?|bills?|fees?|coming up|upcoming|have|dates?|kya)\b", re.I)
STILL_TO_PAY = re.compile(r"\bstill (have to|need to|got to)? ?pay\b|\bstill due\b|\b(unpaid|pending|outstanding)\b", re.I)
MEANING = re.compile(r"^\s*(what did (i|the|my)|what does|what do(es)? \w+ want|what should|which topics|where did|where is|"
                     r"what do i need to (buy|carry|take|bring)|what was|tell me about)\b", re.I)
LOOKUP = re.compile(r"^\s*(when|what time|how much|how many|what is|what's|where|who|which|kab|kitna|kitni)\b|\b(kab|kitna|kitni)\b", re.I)

# What a window question is about
PAY = re.compile(r"\b(pay|payments?|bills?|fees?|emis?|premiums?|rent|dues?)\b", re.I)
OWED_TO_ME = re.compile(r"\bowe(s|d)? me\b|\bowed to me\b|\bmoney (due|owed) to me\b|\bcollect\b", re.I)
TODO = re.compile(r"\b(finish|to do|complete|submit|tasks?|to-?dos?)\b", re.I)
DATES = re.compile(r"\b(birthdays?|anniversar(y|ies)|important dates?|family dates?|festivals?|occasions?)\b", re.I)
DUE = re.compile(r"\bdue\b", re.I)

# Words that name a specific thing, so a memory about a different thing must not answer the question
SPECIFIC_THINGS = {
    "electricity", "bijli", "water", "gas", "cylinder", "broadband", "internet", "wifi", "phone", "mobile", "recharge",
    "dth", "lic", "insurance", "rent", "emi", "loan", "credit", "card", "tax", "gst", "school", "tuition", "college",
    "hostel", "mess", "semester", "exam", "dentist", "eye", "surgery", "scan", "passport", "aadhaar", "pan", "kyc",
    "bank", "account", "salary", "pension", "train", "flight", "bus", "cgpa", "marks", "result",
}
SYNONYMS = {"bijli": "electricity", "bp": "pressure", "dr": "doctor", "doctors": "doctor", "checkup": "check"}
QUESTION_WORDS = {"when", "what", "where", "who", "which", "how", "is", "are", "was", "did", "do", "does", "my", "our", "the",
                  "i", "me", "a", "an", "and", "or", "to", "for", "of", "on", "in", "at", "it", "much", "time", "due", "kab",
                  "hai", "ka", "ki", "ke", "mera", "meri", "mere", "kya", "kitna", "kitni"}


@dataclass
class Route:
    kind: str
    window: Optional[Tuple[date, date]] = None
    focus: List[str] = field(default_factory=list)   # window questions: pay, owed, todo, dates, all
    specific: List[str] = field(default_factory=list)
    source: str = "rules"


def specific_terms(question: str) -> List[str]:
    """Names in the question (capitalised words other than the first, and abbreviations like LIC), then words for
    specific kinds of thing ("electricity", "gas"). Names come first and are marked with a leading '!'."""
    names, things = [], []
    words = re.findall(r"[A-Za-z][A-Za-z\-]*(?:'s)?", question)
    for i, word in enumerate(words):
        clean = re.sub(r"'s$", "", word)
        lower = clean.lower()
        if lower in QUESTION_WORDS:
            continue
        is_name = (i > 0 and clean[:1].isupper()) or (clean.isupper() and len(clean) >= 2)
        if is_name and lower not in {n.lower() for n in names}:
            names.append(clean)
        elif SYNONYMS.get(lower, lower) in SPECIFIC_THINGS and lower not in things:
            things.append(lower)
    return ["!" + n for n in names] + things


def term_matches(term: str, item_text: str) -> bool:
    """Whether a memory mentions a term, allowing synonyms (bijli = electricity, BP = blood pressure) and a shared
    beginning of at least 3 letters (GST and GSTR-3B)."""
    term = SYNONYMS.get(term.lstrip("!").lower(), term.lstrip("!").lower())
    words = {SYNONYMS.get(w, w) for w in re.findall(r"[a-z]+", item_text.lower())}
    return any(w == term or (min(len(w), len(term)) >= 3 and (w.startswith(term) or term.startswith(w))) for w in words)


def mentions_specific(terms: List[str], item_text: str) -> bool:
    """True if the memory matches the question's names (when it has any), otherwise its specific things."""
    names = [t for t in terms if t.startswith("!")]
    things = [t for t in terms if not t.startswith("!")]
    if names:
        return any(term_matches(n, item_text) for n in names)
    if things:
        return any(term_matches(t, item_text) for t in things)
    return True


def window_focus(question: str) -> List[str]:
    focus = []
    if OWED_TO_ME.search(question):
        focus.append("owed")
    elif PAY.search(question):
        focus.append("pay")
    if TODO.search(question) and "pay" in focus:
        focus.append("todo")
    if DATES.search(question):
        focus.append("dates")
    if not focus:
        focus = ["due"] if DUE.search(question) else ["all"]
    return focus


def plan_window(question: str, today: date) -> Tuple[date, date]:
    q = question.lower()
    if re.search(r"\bnext week\b", q):
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if re.search(r"\bweek\b", q):
        return today, today + timedelta(days=6 - today.weekday())
    if re.search(r"\b(tomorrow|kal)\b", q):
        return today + timedelta(days=1), today + timedelta(days=1)
    return today, today


def route_by_rules(question: str, today: date) -> Optional[Route]:
    specific = specific_terms(question)
    if PLAN.search(question):
        return Route("plan", plan_window(question, today), specific=specific)
    if STATUS.search(question):
        return Route("status", specific=specific)
    window = question_window(question, today)
    if STILL_TO_PAY.search(question):
        window = window or (today, today + timedelta(days=90))
        return Route("window", window, window_focus(question) if PAY.search(question) else ["pay"], specific)
    if window and WINDOW_WORDS.search(question) and not re.match(r"^\s*(when|what time|how much)\b", question, re.I):
        return Route("window", window, window_focus(question), specific)
    if MEANING.search(question):
        return Route("meaning", specific=specific)
    if LOOKUP.search(question):
        return Route("lookup", specific=specific)
    return None


def route(question: str, today: date, model: Optional[ChatModel] = None) -> Route:
    decided = route_by_rules(question, today)
    if decided is not None:
        log_event("route", status=decided.kind, stage="router")
        return decided
    kind = "lookup"
    source = "default"
    if model is not None:
        schema = {"type": "object", "properties": {"kind": {"type": "string", "enum": KINDS}}, "required": ["kind"]}
        messages = [{"role": "system", "content": (
            "Classify the user's question to their personal memory assistant. plan: they want their day or week planned. "
            "window: they want a list of what falls in a period (due this week, coming up). status: they ask whether "
            "something was done or paid. lookup: they ask for one fact (when, how much, what time). meaning: they ask "
            "what a note says about something.")},
            {"role": "user", "content": question}]
        try:
            kind = json.loads(model.chat(messages, max_tokens=20, json_schema=schema, purpose="route").text)["kind"]
            source = "model"
        except (LLMError, ValueError, KeyError):
            pass
    window = question_window(question, today) if kind == "window" else plan_window(question, today) if kind == "plan" else None
    if kind == "window" and window is None:
        window = (today, today + timedelta(days=7))
    log_event("route", status=kind, stage="router_" + source)
    return Route(kind, window, window_focus(question) if kind == "window" else [], specific_terms(question), source)

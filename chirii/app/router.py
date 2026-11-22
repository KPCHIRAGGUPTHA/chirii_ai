"""
Question router (S4c): one router for everything Chirii is asked.

    memory      lookup ("when is…"), status ("is it paid?", "did I…?"), window (what falls in a period),
                plan (a day's or week's timeline), meaning (what a note says)
    people      "who is Kamala?" (the route exists; people memory is built in the people step)
    general     maths (worked out by the calculator), general knowledge, writing and advice
    mixed       maths or advice that needs the person's memory ("if my mess bill is split into 4…")

A personal question (my, I, me, we, mera…, or a name saved in memory) never goes to pure general knowledge.

The rules run first. If no rule is sure, Qwen picks the route, the period and the named item inside a JSON schema
that allows only valid values; code then checks its choice (the period must give real dates, a named item must exist,
the route must fit the question's words). If the choice fails the checks, a safe default is used: show the relevant
memories for a personal question, general knowledge otherwise. Nothing is guessed.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from chirii.app.llm.client import ChatModel, LLMError
from chirii.app.logging_setup import log_event
from chirii.app.periods import WEEKDAYS, Period, read_period

ROUTES = ["lookup", "status", "window", "plan", "meaning", "people", "maths", "general", "writing", "mixed"]
MEMORY_ROUTES = {"lookup", "status", "window", "plan", "meaning", "people"}
KINDS = ROUTES  # kept for older callers

PERSONAL = re.compile(r"\b(my|me|mine|i|i'm|i've|i'll|we|our|us|mera|meri|mere|mujhe|hum|hamara|hamari|apna|apni)\b", re.I)
RELATIONS = re.compile(r"\b(mom|mum|mother|dad|father|amma|appa|papa|mummy|wife|husband|son|daughter|sister|brother|"
                       r"granddaughter|grandson|nephew|niece|aunt|aunty|uncle|landlord|maid|cook)\b", re.I)
PLAN = re.compile(r"\b(plan|planner|schedule|agenda|timetable for my)\b|\bmy (day|week)\b|\b(aaj|kal) ka plan\b|"
                  r"\b(look|looks|looking) like\b|\bhow (does|do|will) my \w+ look\b|\bhow'?s my \w+ looking\b|"
                  r"\bwhat'?s my \w+ like\b|\bon my plate\b|\bkaisa (hai|rahega)\b|"
                  r"\btell me about my (day|week|weekend|\w+day)\b", re.I)
STATUS = re.compile(r"^\s*(did|have|has|is|are|was|were|do|does)\b.*\b(paid|pay|taken|took|done|submitted|collected|finished|"
                    r"booked|renewed|cleared|settled|returned|delivered|pending|owe|due)\b|^\s*did i\b", re.I)
LIST_WORDS = re.compile(r"\b(what|which|any|anything|due|pay|payments?|bills?|fees?|coming up|upcoming|happening|have|on|dates?|kya|"
                        r"how much)\b", re.I)
STILL_TO_PAY = re.compile(r"\bstill (have to|need to|got to)? ?pay\b|\b(unpaid|outstanding)\b", re.I)
MEANING = re.compile(r"^\s*(what did (i|the|my)|what does|what do(es)? \w+ want|what should i (take|carry|bring|remember)|which topics|"
                     r"where did|where is|what do i need to (buy|carry|take|bring|get)|what was|anything special|tell me about the)\b", re.I)
LOOKUP = re.compile(r"^\s*(when|what time|how much|how many|what is|what's|where|who|which)\b|\b(kab|kitna|kitni|kya hai)\b", re.I)
# S4d: the name part is case-sensitive, so "tell me about my week" never treats "my" as a name
WHO_IS = re.compile(r"^\s*(?i:who is|who's|tell me about)\s+([A-Z][\w.]*(?:\s+[A-Z][\w.]*)?)")
WRITING = re.compile(r"^\s*(write|draft|compose|rewrite|translate|suggest|explain|give me \d* ?(tips|ideas|ways)|list \d* ?(tips|ideas|ways)|"
                     r"help me (write|say|word))\b", re.I)
ADVICE = re.compile(r"^\s*(how (do|can|should) i|how to|what should|any tips|is it (ok|okay|safe)|should i)\b", re.I)
LIVE = re.compile(r"\b(today'?s|right now|now|currently|current|latest|live|yesterday'?s?|this morning'?s)\b.*\b(price|rate|weather|news|"
                  r"score|match|result|traffic|temperature)\b|\b(price|rate|weather|news|score)\b.*\b(today|now|right now|currently)\b", re.I)
MATHS_WORDS = re.compile(r"\d+(\.\d+)?\s*%|\b(percent|interest|split|divide|divided|share equally|equally|per (day|month|person|kg|litre)|"
                         r"each part|each person|instalments?|installments?|convert|km/h|m/s|square (feet|metres|meters)|area|"
                         r"how many days|days (from|between|until|till)|in total|altogether|sum of|total of|add|half|double|twice|"
                         r"plus|minus|times|multiplied)\b|\d\s*[-+*/x×÷]\s*\d|\bcost of\b|\bwhat do \d+ \w+ cost\b", re.I)
PAY = re.compile(r"\b(pay|payments?|bills?|fees?|emis?|premiums?|rent)\b", re.I)
OWED_TO_ME = re.compile(r"\bowe(s|d)? me\b|\bowed to me\b|\bmoney (due|owed) to me\b|\bto receive\b|\blena hai\b", re.I)
TODO = re.compile(r"\b(finish|to do|complete|submit|tasks?|to-?dos?)\b", re.I)
DATES = re.compile(r"\b(birthdays?|anniversar(y|ies)|important dates?|family dates?|festivals?|occasions?)\b", re.I)
DUE = re.compile(r"\bdue\b", re.I)

SPECIFIC_THINGS = {
    "electricity", "bijli", "water", "gas", "cylinder", "broadband", "internet", "wifi", "phone", "mobile", "recharge",
    "dth", "lic", "insurance", "rent", "emi", "loan", "credit", "card", "tax", "gst", "school", "tuition", "college",
    "hostel", "mess", "semester", "exam", "dentist", "eye", "surgery", "scan", "passport", "aadhaar", "pan", "kyc",
    "bank", "account", "salary", "pension", "train", "flight", "bus", "cgpa", "marks", "result",
    # S4c: kinds of event or document, so "Infosys interview" needs both words to match
    "interview", "appointment", "meeting", "party", "wedding", "engagement", "birthday", "anniversary", "function",
    "play", "practice", "class", "trip", "filing", "renewal", "deadline", "annual", "match", "concert", "ceremony",
    "visa", "ticket", "booking",
}
SYNONYMS = {"bijli": "electricity", "bp": "pressure", "dr": "doctor", "doctors": "doctor", "checkup": "check",
            "physio": "physiotherapy", "mom": "mom", "amma": "amma"}
QUESTION_WORDS = {"when", "what", "where", "who", "which", "how", "is", "are", "was", "did", "do", "does", "my", "our", "the",
                  "i", "me", "a", "an", "and", "or", "to", "for", "of", "on", "in", "at", "it", "much", "time", "due", "kab",
                  "hai", "ka", "ki", "ke", "ko", "se", "mera", "meri", "mere", "kya", "kitna", "kitni", "lena", "dena", "now"}
NOT_NAMES = set(WEEKDAYS) | {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                             "november", "december", "today", "tomorrow", "ok", "okay", "please", "hi", "hello"}


@dataclass
class Route:
    kind: str
    window: Optional[Tuple[date, date]] = None
    focus: List[str] = field(default_factory=list)   # window questions: pay, owed, todo, dates, due, all
    specific: List[str] = field(default_factory=list)
    source: str = "rules"
    named: bool = True        # the question named the period (otherwise it is a default, and is not stated)
    period: Optional[Period] = None
    personal: bool = False


def specific_terms(question: str) -> List[str]:
    """Names in the question (capitalised words other than the first, and abbreviations like LIC; never weekday or
    month names), then words for specific kinds of thing. Names are marked with a leading '!'."""
    names, things = [], []
    words = re.findall(r"[A-Za-z][A-Za-z\-]*(?:'s)?", question)
    for i, word in enumerate(words):
        clean = re.sub(r"'s$", "", word)
        lower = clean.lower()
        if lower in QUESTION_WORDS or lower in NOT_NAMES:
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
    """S4 rule, kept for meaning questions: the memory matches the question's names (if any), otherwise one thing."""
    names = [t for t in terms if t.startswith("!")]
    things = [t for t in terms if not t.startswith("!")]
    if names:
        return any(term_matches(n, item_text) for n in names)
    if things:
        return any(term_matches(t, item_text) for t in things)
    return True


def matched_terms(terms: List[str], item_text: str) -> List[str]:
    return [t for t in terms if term_matches(t, item_text)]


def mentions_all(terms: List[str], item_text: str) -> bool:
    """S4c rule for lookups and status questions: every key term must match ("Infosys" and "interview")."""
    return all(term_matches(t, item_text) for t in terms)


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


def is_personal(question: str, names_in_memory: List[str], items: Optional[List[Dict[str, Any]]] = None) -> bool:
    """My/I/we/mera…, a relation (mom, landlord), a name saved in memory, or a specific thing that a saved item is about
    ("when is the gas bill due?" when a gas bill is saved)."""
    if PERSONAL.search(question) or RELATIONS.search(question):
        return True
    if any(re.search(rf"\b{re.escape(n.lower())}\b", question.lower()) for n in names_in_memory):
        return True
    terms = specific_terms(question)
    things = [t for t in terms if not t.startswith("!")]
    if items and any(term_matches(t, i["text"]) for t in things for i in items):
        return True
    # S4d: a name or abbreviation from the question ("GST", "Patel") that a saved item mentions
    names = [t for t in terms if t.startswith("!")]
    if items and any(term_matches(n, i["text"]) for n in names for i in items):
        return True
    from chirii.app.memory.store import keywords
    words = keywords(question)
    return bool(items) and any(len(words & keywords(i["text"])) >= 2 for i in items)


def memory_names(items: List[Dict[str, Any]]) -> List[str]:
    """Capitalised names in what the person told Chirii (Kamala, Sharma ji, Infosys), for spotting personal questions."""
    names = set()
    for item in items:
        for word in re.findall(r"\b[A-Z][a-z]{2,}\b", item["text"]):
            if word.lower() not in NOT_NAMES and word.lower() not in QUESTION_WORDS and word not in ("The", "Note", "My", "Our", "Shop", "Pay"):
                names.add(word)
    return sorted(names)


def route_by_rules(question: str, today: date, items: Optional[List[Dict[str, Any]]] = None) -> Optional[Route]:
    items = items or []
    specific = specific_terms(question)
    names = memory_names(items)
    personal = is_personal(question, names, items)
    period = read_period(question, today, items)
    # Asking an assistant what is planned, due or coming up is always about the person's own memory
    if PLAN.search(question) or (period and re.search(r"\b(anything|due|coming up|upcoming|happening|pending|on)\b", question, re.I)):
        personal = True
    window = (period.start, period.end) if period else None

    def r(kind, **kw):
        return Route(kind, kw.pop("window", window), kw.pop("focus", []), specific, period=period, personal=personal, **kw)

    who = WHO_IS.match(question)
    if who and any(term_matches(who.group(1).split()[0], i["text"]) for i in items):
        return r("people")
    maths = bool(MATHS_WORDS.search(question))
    if maths and personal:
        return r("mixed")
    if maths and re.search(r"\d", question):
        return r("maths")
    if WRITING.search(question):
        return r("writing")
    if ADVICE.search(question):
        if personal and strongly_about_memory(question, items):
            return r("mixed")
        if personal or not LOOKUP.search(question):
            return r("writing")
    if LIVE.search(question) and not personal:
        return r("general")
    if not personal:
        return r("general") if LOOKUP.search(question) or question.strip().endswith("?") else None
    # personal: memory routes
    if PLAN.search(question) and (period is None or period.kind in ("day", "weekday") or "week" in period.words):
        if period is None:
            period = Period(today, today, "today", "day")
        return Route("plan", (period.start, period.end), [], specific, period=period, personal=True)
    # "Do I need to pay anything…?" asks for a list, not the status of one item
    # ...but "Do I still owe anything for the recital?" names no period, so it is about one item (S4d)
    if STATUS.search(question) and (not re.search(r"\bany(thing)?\b", question, re.I) or period is None):
        return r("status")
    if STILL_TO_PAY.search(question):
        named = window is not None
        return r("window", window=window or (today, today + timedelta(days=90)), focus=window_focus(question) if PAY.search(question) else ["pay"], named=named)
    if period and (LIST_WORDS.search(question) or period.kind == "event"):
        if period.kind == "weekday" and not PAY.search(question):
            return Route("plan", window, [], specific, period=period, personal=True)
        return r("window", focus=window_focus(question))
    if MEANING.search(question):
        return r("meaning")
    if LOOKUP.search(question):
        return r("lookup")
    return None


def strongly_about_memory(question: str, items: List[Dict[str, Any]]) -> bool:
    """An advice question that names a saved item (two or more shared words with one memory) is mixed."""
    from chirii.app.memory.store import keywords
    words = keywords(question)
    return any(len(words & keywords(i["text"])) >= 2 for i in items)


ROUTE_SCHEMA_PERIODS = ["none", "today", "tomorrow", "this week", "next week", "this weekend", "this month", "next month",
                        *[f"on {d.capitalize()}" for d in WEEKDAYS]]


def route(question: str, today: date, model: Optional[ChatModel] = None, items: Optional[List[Dict[str, Any]]] = None) -> Route:
    items = items or []
    decided = route_by_rules(question, today, items)
    if decided is not None:
        log_event("route", status=decided.kind, stage="router")
        return decided
    personal = is_personal(question, memory_names(items), items)
    safe = Route("lookup" if personal else "general", specific=specific_terms(question), source="default", personal=personal)
    if model is None:
        log_event("route", status=safe.kind, stage="router_default")
        return safe
    titles = sorted({(i.get("title") or "")[:60] for i in items if i.get("title")})[:40]
    schema = {"type": "object", "properties": {
        "route": {"type": "string", "enum": ROUTES},
        "period": {"type": "string", "enum": ROUTE_SCHEMA_PERIODS},
        "item": {"anyOf": [{"type": "string", "enum": titles}, {"type": "null"}]} if titles else {"type": "null"}},
        "required": ["route", "period", "item"]}
    messages = [{"role": "system", "content": (
        "Classify this question to a personal memory assistant. Routes: lookup (one saved fact), status (whether something was "
        "done or paid), window (what falls in a period), plan (a day or week planned), meaning (what a saved note says), people "
        "(about a person), maths (a calculation), general (general knowledge), writing (write or advise), mixed (calculation "
        "or advice using saved memories). Give the period the question names, and the saved item it names, if any.")},
        {"role": "user", "content": question}]
    try:
        choice = json.loads(model.chat(messages, max_tokens=60, json_schema=schema, purpose="route").text)
    except (LLMError, ValueError):
        log_event("route", status=safe.kind, stage="router_default")
        return safe
    checked = validate_choice(choice, question, today, personal, items)
    if checked is None:
        log_event("route", status=safe.kind, stage="router_rejected")
        return safe
    log_event("route", status=checked.kind, stage="router_model")
    return checked


def validate_choice(choice: Dict[str, Any], question: str, today: date, personal: bool, items: List[Dict[str, Any]]) -> Optional[Route]:
    """Code checks Qwen's routing: the period gives real dates, a named item exists, and the route fits the words."""
    kind = choice.get("route")
    if kind not in ROUTES:
        return None
    if personal and kind == "general":
        return None                                   # personal questions never go to pure general
    if kind in ("maths", "mixed") and not (re.search(r"\d", question) or kind == "mixed" and items):
        return None
    period = None
    if choice.get("period") and choice["period"] != "none":
        period = read_period(choice["period"], today)
        if period is None:
            return None
    if kind in ("window", "plan") and period is None:
        return None
    if choice.get("item") and not any((i.get("title") or "")[:60] == choice["item"] for i in items):
        return None
    if kind == "people" and not specific_terms(question):
        return None
    window = (period.start, period.end) if period else None
    return Route(kind, window, window_focus(question) if kind == "window" else [], specific_terms(question), "model",
                 period=period, personal=personal)

"""
S6: follow-up questions in a conversation.

A follow-up ("and the water bill?", "what about Priya?", "Priya ka kya?", "and tomorrow?", "is it paid?", "when does
he land?") is rewritten into a question that stands on its own, and that question goes through the normal pipeline
(safety, router, fact-check guard, attribute check). The person sees what it was understood as.

- Rules first: an elliptical question replaces the subject or the time of the previous question; a pronoun ("it",
  "he", "his"…) is replaced by the previous question's subject.
- Qwen only when the question looks like a follow-up and no rule resolves it. Its rewrite is used only if it adds no
  new names or numbers (every one must come from the previous question, the previous answer or this question);
  otherwise the question is answered as it was asked.
- History lives in memory only, per login session (see api/main.py). It is never written to disk.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from chirii.app.llm.client import ChatModel, LLMError

MAX_TURNS = 10


@dataclass
class Turn:
    question: str           # as typed
    understood: str         # the question actually answered (the rewrite, or the question itself)
    answer: str
    fact_ids: List[int] = field(default_factory=list)
    route: str = ""


WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
# A time phrase: what "and tomorrow?" replaces in "What's my plan for today?"
TIME = re.compile(
    rf"\b(?:(?:on|for|this|next|coming)\s+)?(?:today|tonight|tomorrow|day after tomorrow|aaj|kal|parso"
    rf"|(?:this|next|last|coming)\s+(?:week|weekend|month|year)|(?:{WEEKDAYS})s?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\b|(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?|(?:in\s+)?(?:{MONTHS}))\b", re.I)

ELLIPSIS = [
    re.compile(r"^\s*(?:and|aur|also|then)?\s*(?:what|how)\s+about\s+(?P<x>.+?)\s*[?.!]*\s*$", re.I),
    re.compile(r"^\s*(?:and|aur)\s+(?P<x>.+?)\s*[?.!]*\s*$", re.I),
    re.compile(r"^\s*(?:aur\s+)?(?P<x>.+?)\s+(?:ka|ki|ke)\s+kya\s*[?.!]*\s*$", re.I),
]
# The subject of a question: "my electricity bill", "the water bill", "Rahul's interview"
SUBJECT = re.compile(
    r"(?:\b(?:my|the|our)|\b[A-Z][a-z]+'s)(?:\s+[A-Za-z][A-Za-z-]*){1,3}?"
    r"(?=\s+(?:due|on|at|in|for|is|was|are|were|start|starts|begin|begins|end|ends|paid|done|today|tomorrow|this|next|from|to|with)\b|\s*[?.!,]|\s*$)")
NAME = re.compile(r"\b([A-Z][a-z]+)(?:'s)?\b")
NOT_NAMES = {"When", "What", "Where", "Who", "Which", "How", "Is", "Are", "Was", "Did", "Do", "Does", "Can", "Should",
             "Will", "My", "The", "And", "Aur", "I", "Today", "Tomorrow", "Your", "It", "He", "She", "They", "Please"}
NOT_NAMES |= {d.capitalize() for d in WEEKDAYS.split("|")} | {m.capitalize() for m in MONTHS.split("|")}
PERSON_PRONOUN = re.compile(r"\b(he|she|him|his|her)\b", re.I)
# Hinglish "uska time", "uski date": usually the thing just talked about ("Amma's eye check-up ka time")
HINGLISH_PRONOUN = re.compile(r"\b(uska|uski|uske|usko|iska|iski|iske)\b", re.I)
THING_PRONOUN = re.compile(r"\b(it|its)\b|\b(that|this)(?=\s*[?.!]|\s*$)", re.I)
# "it" that refers to nothing in the conversation
IMPERSONAL = re.compile(r"\bit(?:'s|\s+is)?\s+(?:going to\s+)?(?:rain|raining|sunny|cold|hot|humid|late|early|time)\b"
                        r"|\bwhat time is it\b|\bhow is it going\b|\bis it true\b", re.I)


def _subject(question: str) -> Optional[str]:
    m = SUBJECT.search(question)
    return m.group(0) if m else None


def _person(question: str) -> Optional[str]:
    for m in NAME.finditer(question):
        if m.group(1) not in NOT_NAMES:
            return m.group(1)
    return None


def _is_time(x: str) -> bool:
    return bool(TIME.fullmatch(x.strip()))


def _replace_time(prev: str, x: str) -> Optional[str]:
    m = TIME.search(prev)
    if not m:
        return None
    lead = re.match(r"(?:on|for|in)\s+", m.group(0), re.I)       # keep "for", "on", "in" from the previous question
    x = x.strip()
    has_own = re.match(r"(?:on|for|this|next|coming|in)\s", x, re.I)
    relative = re.match(r"(?:today|tonight|tomorrow|day after|aaj|kal|parso)", x, re.I)
    if lead and not has_own and not (relative and lead.group(0).lower().startswith("on")):
        x = lead.group(0) + x
    return prev[:m.start()] + x + prev[m.end():]


def _replace_subject(prev: str, x: str) -> Optional[str]:
    x = x.strip()
    words = x.split()
    # a person's name: "what about Priya?" after "When is Rahul's interview?"
    if len(words) <= 2 and all(w[:1].isupper() for w in words):
        name = _person(prev)
        if name:
            return re.sub(rf"\b{re.escape(name)}\b", x.removesuffix("'s"), prev, count=1)
        return None
    # a thing: "and the water bill?" after "When is my electricity bill due?" (they share the head noun "bill")
    subject = _subject(prev)
    if subject:
        head = subject.split()[-1].lower()
        if words[-1].lower().rstrip("s") == head.rstrip("s"):
            if not re.match(r"(?:my|the|our|[A-Z][a-z]+'s)\b", x):
                x = subject.split()[0] + " " + x if subject.split()[0].lower() in ("my", "the", "our") else "my " + x
            return prev.replace(subject, x, 1)
    return None


def _replace_pronouns(question: str, prev: Turn) -> Optional[str]:
    if IMPERSONAL.search(question):
        return None
    out = question
    if PERSON_PRONOUN.search(out):
        name = _person(prev.understood)
        if not name or len(PERSON_PRONOUN.findall(out)) > 1:   # "Did she call him?": two people, not a rule's job
            return None

        def person(m: re.Match) -> str:
            word = m.group(1).lower()
            after = out[m.end():]
            possessive = word == "his" or \
                (word == "her" and re.match(r"\s+[a-z]", after) and not re.match(r"\s+(?:to|at|on|in|for|about|and|or)\b", after))
            return name + "'s" if possessive else name
        out = PERSON_PRONOUN.sub(person, out)
    if HINGLISH_PRONOUN.search(out):
        subject = _subject(prev.understood) or _person(prev.understood)
        if not subject:
            return None
        out = HINGLISH_PRONOUN.sub(lambda m: subject + (" ka" if m.group(1).lower().endswith("a") else
                                                        " ki" if m.group(1).lower().endswith("i") else " ke"), out)
    if THING_PRONOUN.search(out):
        subject = _subject(prev.understood)
        if not subject:
            return None
        out = THING_PRONOUN.sub(lambda m: subject + ("'s" if (m.group(1) or "").lower() == "its" else ""), out)
    return out if out != question else None


def looks_like_followup(question: str) -> bool:
    q = question.strip()
    return bool(any(p.match(q) for p in ELLIPSIS) or
                ((PERSON_PRONOUN.search(q) or THING_PRONOUN.search(q) or HINGLISH_PRONOUN.search(q)) and not IMPERSONAL.search(q) and len(q.split()) <= 10))


def by_rules(question: str, history: List[Turn]) -> Optional[str]:
    if not history:
        return None
    prev = history[-1]
    for pattern in ELLIPSIS:
        m = pattern.match(question)
        if m:
            x = m.group("x").strip()
            found = _replace_time(prev.understood, x) if _is_time(x) else _replace_subject(prev.understood, x)
            if found:
                return found
            break
    if len(question.split()) <= 10:
        bare = re.sub(r"^\s*(?:and|aur|also|then)\s+", "", question, flags=re.I)   # "And when is it due?"
        found = _replace_pronouns(bare, prev)
        return found[:1].upper() + found[1:] if found else None
    return None


REWRITE_PROMPT = ("Rewrite the user's new question so that it can be understood on its own, using the previous question "
                  "and answer only to fill in what it refers to. Keep the user's own words. Do not answer it. Reply with "
                  "the rewritten question only.")
TOKENS = re.compile(r"[A-Z][a-z]+|\d[\d,.:]*")


def _adds_nothing(rewrite: str, question: str, prev: Turn) -> bool:
    known = f"{prev.understood} {prev.answer} {question}"
    return all(t in known or t in NOT_NAMES for t in TOKENS.findall(rewrite))


def by_model(question: str, history: List[Turn], model: ChatModel) -> Optional[str]:
    prev = history[-1]
    messages = [{"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Previous question: {prev.understood}\nPrevious answer: {prev.answer[:600]}\n"
                                            f"New question: {question}"}]
    try:
        text = model.chat(messages, max_tokens=60, purpose="followup").text.strip().strip('"').splitlines()[0].strip()
    except (LLMError, IndexError):
        return None
    if not text or len(text) > 200 or text.lower() == question.lower() or not _adds_nothing(text, question, prev):
        return None
    return text


def resolve(question: str, history: List[Turn], model: Optional[ChatModel] = None) -> Tuple[Optional[str], str]:
    """(rewritten question or None, how: "rule", "model" or "")."""
    if not history:
        return None, ""
    rewritten = by_rules(question, history)
    if rewritten:
        return rewritten, "rule"
    if model is not None and looks_like_followup(question):
        rewritten = by_model(question, history, model)
        if rewritten:
            return rewritten, "model"
    return None, ""

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

import json
import re
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from chirii.app.formatting import KIND_LABELS, make_title, say_amount, say_date, say_time, when_text
from chirii.app.guard import check_answer
from chirii.app.windows import question_window
from chirii.app.planner import MONEY_QUESTION, Draft, label_for, plan_draft, status_draft, strip_when, window_draft
from chirii.app import meaning, safety
from chirii.app.router import LIVE, MATHS_WORDS, MEMORY_ROUTES, matched_terms, mentions_all, mentions_specific, route as route_question, term_matches
from chirii.app import people
from chirii.app.chat import LABELS, general_answer, maths_answer, writing_answer
from chirii.app.llm.client import ChatModel
from chirii.app.logging_setup import log_event
from chirii.app.memory.store import MemoryStore, keywords
from chirii.eval.scoring import CLASH_WORDS, _position, contains, find_amounts, find_dates

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

MEDICINE_WORDS = re.compile(r"\b(medicines?|tablets?|pills?|dose|doses|metformin|insulin|capsules?|syrup|drops)\b", re.I)
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


HISTORY_QUESTION = re.compile(r"\b(original(ly)?|previous(ly)?|old|earlier|before it (was )?(moved|changed)|first (date|time)|"
                              r"moved from|what changed|used to be)\b", re.I)
NEXT_QUESTION = re.compile(r"\b(next|upcoming|coming)\b", re.I)


def fact_line_v2(item: Dict[str, Any], history: List[Dict[str, Any]], today: date, question: str) -> str:
    """S4b fact line. A moved or corrected item shows only its current values: the user's sentence is given with the
    superseded dates or times taken out, and the history only if the question asks about the original or previous
    value. An item with several dates lists each with its label ("DBMS on Wednesday 9 December"), or only the next
    upcoming one if the question says "next"; never just the first stored date."""
    changed_fields = {h["field"] for h in history if h["field"] in ("date", "end_date", "time")}
    show_history = bool(HISTORY_QUESTION.search(question))
    if item.get("other_dates"):
        days = sorted([item["date"], *item["other_dates"]])
        labelled = [(d, label_for(item, day=date.fromisoformat(d))) for d in days]
        if NEXT_QUESTION.search(question):
            upcoming = [x for x in labelled if x[0] >= today.isoformat()]
            labelled = upcoming[:1] or labelled[-1:]
        details = ["dates: " + "; ".join(f"{label} on {say_date(d)}" + (" (already past)" if d < today.isoformat() else "")
                                         for d, label in labelled)]
        details += when_text(dict(item, date=None, end_date=None, other_dates=[]))
    else:
        details = when_text(item)
    if item["amount"] is not None:
        details.append(f"amount {say_amount(item['amount'])}")
        if item["paid_amount"]:
            details.append(f"{say_amount(item['paid_amount'])} paid so far, {say_amount(item['remaining'])} still owed")
    status = {"paid": "PAID", "cancelled": "CANCELLED", "unlinked": "not matched to an earlier item"}.get(item["status"])
    if status:
        details.append(f"status: {status}")
    said = item["text"]
    if changed_fields and not show_history:
        said = strip_when(said, dates="date" in changed_fields or "end_date" in changed_fields, times="time" in changed_fields)
        line = f"[{KIND_LABELS.get(item['kind'], item['kind'])}] You said (with values changed since left out): \"{said}\""
    else:
        line = f"[{KIND_LABELS.get(item['kind'], item['kind'])}] You said: \"{said}\""
    if details:
        line += " | Current details: " + "; ".join(details)
    if changed_fields and show_history:
        line += " | Earlier values: " + "; ".join(f"{h['field']} was {h['old_value']}, changed to {h['new_value']}"
                                                  for h in history if h["field"] in changed_fields)
    if not item.get("confirmed"):
        line += " | Not yet confirmed"
    return line


def system_prompt(today: date, version: int = 1) -> str:
    """Version 1 is the prompt of runs (a) and (b), kept unchanged so they can be reproduced. Version 2 (S4) adds two
    general rules for failures seen in S3: answers that said "I don't have that" and then gave it, and answers that
    called the user "the user"."""
    rules = [
        f"- If the facts do not contain the answer, say \"{NOT_IN_MEMORY}\" Then, if something related is in the facts, "
        "you may mention it.",
        "- Never make up a date, time, amount, name or place. Copy dates and amounts exactly as the facts give them.",
        "- When a fact has 'Current details', use those, not older values in what the user said.",
        "- Items marked PAID or CANCELLED are not due any more.",
        "- Answer briefly: one to four sentences, or a short list.",
    ]
    if version >= 2:
        rules[0] = (f"- If a fact answers the question, answer it directly and do not say you don't have it. Only if no "
                    f"fact answers it, say \"{NOT_IN_MEMORY}\"")
        rules[2] = "- When a fact has 'Current details', use those, not older values in what was said."
        rules.append("- Speak directly to the person as 'you'; never refer to them in the third person.")
        rules.append("- When you say when something is, give its date and, if it has one, its time.")
        return ("You are Chirii, a personal memory assistant. Answer the question using only the memory facts given.\n"
                f"Today is {say_date(today.isoformat())}.\nRules:\n" + "\n".join(rules))
    # version 1 is frozen: it is the prompt of runs (a) and (b)
    return ("You are Chirii, a personal memory assistant. Answer the user's question using only the memory facts given.\n"
            f"Today is {say_date(today.isoformat())}.\nRules:\n" + "\n".join(rules))


LEAD_MAX, CLOSING_MAX = 160, 120
LEAD_SCHEMA = {"type": "object", "properties": {"lead": {"type": "string", "maxLength": LEAD_MAX},
                                                "closing": {"type": "string", "maxLength": CLOSING_MAX}},
               "required": ["lead", "closing"]}
NO_CLASH = re.compile(r"\bno (clash|clashes|conflicts?|overlaps?)\b|\bnothing (clashes|overlaps)\b", re.I)
TIME_IN_TEXT = re.compile(r"\b\d{1,2}([:.]\d{2})?\s*(am|pm|a\.m\.|p\.m\.)", re.I)


def add_missing_time(text: str, items: List[Dict[str, Any]], today: date) -> str:
    """S4c: a lookup answer that gives an item's date but not its time gets the time added by code."""
    for item in items:
        if not item.get("time") or not item.get("date"):
            continue
        if contains(text, {"date": item["date"]}, today) and not contains(text, {"time": item["time"]}, today):
            return text.rstrip() + f" It's at {say_time(item['time'])}."
    return text


def lead_problems(lead: str, draft: Draft, today: date, question: str) -> List[str]:
    """Qwen's lead and closing are checked by the guard only (owner's spec; the stricter S4b rule that forbade any
    date, amount or time in the lead was dropped after run (d)). The guard allows the start date of a period the
    question names. One extra check is kept: the lead must not deny a clash the code found."""
    problems = check_answer(lead, draft.items, {}, today, question)
    if draft.clash and NO_CLASH.search(lead):
        problems.append("lead denies the clash")
    # S4d (owner, after run (e3)): a lead may not say there is nothing when the code's list has items
    if draft.items and NOTHING.search(lead.replace("’", "'")):
        problems.append("lead says there is nothing, but the list has items")
    return problems


NOTHING = re.compile(r"\b(nothing|no (payments?|bills?|dues?|plans?|appointments?|tasks?)|none)\b|\b(do not|don't|dont) (have|need) to pay\b|"
                     r"\b(not|n't) (have|need) to (pay|do) anything\b|\b(you'?re|you are) (completely |all )?free\b|\bfree (day|evening|weekend)\b|"
                     r"\b(all|completely) clear\b", re.I)


def lead_prompt(today: date) -> str:
    return ("You are Chirii, a personal memory assistant. The answer below was worked out from the person's saved "
            "memories and will be shown to them exactly as it is. Write one short, friendly lead sentence to go "
            "before it, and an optional short closing line (or an empty string).\n"
            f"Today is {say_date(today.isoformat())}.\nRules:\n"
            "- Do not repeat or change any item, date, day, time or amount, and do not add any. The list says them.\n"
            "- If the answer warns about a clash, you may say there is a clash, but never that there isn't one.\n"
            "- Speak directly to the person as 'you'. Reply as JSON with \"lead\" and \"closing\".")


def code_answer(items: List[Dict[str, Any]], histories: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> str:
    """The answer built by code when the model's answers failed the check twice: what is saved. An item whose date or
    time was changed later is shown by its title and current details, not by the original sentence, which still holds
    the old value (S3 bug, S1-37)."""
    if not items:
        return NOT_IN_MEMORY
    lines = ["Here is what I have saved about that:"]
    for item in items:
        details = when_text(item)
        if item["amount"] is not None:
            details.append(say_amount(item["amount"]) + (f", {say_amount(item['remaining'])} still owed" if item["paid_amount"] else ""))
        if item["status"] in ("paid", "cancelled"):
            details.append(item["status"])
        changed = any(h["field"] in ("date", "time", "end_date") for h in (histories or {}).get(item["id"], []))
        if changed:
            lines.append(f"- {item.get('title') or make_title(item['text'])}: {'; '.join(details)} (updated since you first said it).")
        else:
            lines.append(f"- You said: \"{item['text']}\"" + (f" Now: {'; '.join(details)}." if details else ""))
    return "\n".join(lines)


@dataclass
class Answer:
    text: str
    fact_ids: List[int]
    facts: List[str] = field(default_factory=list)
    duration_ms: int = 0
    guard: str = "off"                      # off, pass, retry, fallback (code-built after two failed checks), code (answered by code by design)
    problems: List[str] = field(default_factory=list)
    unconfirmed: List[str] = field(default_factory=list)
    first_answer: Optional[str] = None      # the model's first answer, kept for the S3 report when the guard acted
    route: str = "-"                         # S4c: one of router.ROUTES
    label: str = ""                          # S4c: "From your memory", "Calculated", "General knowledge", "Writing help"…


class Assistant:

    def __init__(self, store: MemoryStore, model: ChatModel, guard: bool = True, trimmed: bool = True, routed: bool = True):
        """routed=False with guard and trimmed gives S3's run (b); guard=False, trimmed=False, routed=False gives run (a).
        The defaults are the S4 system (run (c))."""
        self.store = store
        self.model = model
        self.guard = guard
        self.trimmed = trimmed
        self.routed = routed

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
            # S3 bug: daily medicines filled the window limit before real appointments were reached. Items are now
            # counted once each, and daily medicines are left out unless the question is about medicines.
            wants_medicine = bool(MEDICINE_WORDS.search(question))
            for entry in self.store.agenda(*window):
                item = entry["item"]
                if item["kind"] == "medicine" and not item["date"] and not wants_medicine:
                    continue
                if item["id"] not in chosen and len(chosen) >= MAX_WINDOW:
                    break
                chosen.setdefault(item["id"], item)
        for item in self.matches(question):
            chosen.setdefault(item["id"], item)
        return list(chosen.values())

    # ------------------------------------------------------------------ answering

    def _line(self, item: Dict[str, Any], history: List[Dict[str, Any]], today: date, question: str) -> str:
        return fact_line_v2(item, history, today, question) if self.routed else fact_line(item, history)

    def _ask_model(self, question: str, today: date, items: List[Dict[str, Any]], histories) -> str:
        facts = [self._line(i, histories[i["id"]], today, question) for i in items]
        memory = "\n".join(f"{n}. {f}" for n, f in enumerate(facts, 1)) or "(no memory facts match this question)"
        messages = [{"role": "system", "content": system_prompt(today, version=2 if self.routed else 1)},
                    {"role": "user", "content": f"Memory facts:\n{memory}\n\nQuestion: {question}"}]
        return self.model.chat(messages, max_tokens=256, purpose="answer").text

    def ask(self, question: str, today: date) -> Answer:
        started = time.perf_counter()
        support = safety.support(question) if self.routed else None   # runs (a)/(b) stay frozen as in S3
        if support is not None:
            # S5 safety: a fixed supportive reply (the person's own crisis, or S6: worry about someone else);
            # Qwen is not called and no memory is used
            kind, text = support
            log_event("safety", status="self_harm" if kind == "self" else "helper", stage="answer")
            return Answer(text=text, fact_ids=[], guard="safety", route="safety", label=LABELS["support"],
                          duration_ms=round((time.perf_counter() - started) * 1000))
        if self.routed:
            # S4f: matching by meaning (the calibrated embedding model) for this question's router and item checks
            token = meaning.current.set(meaning.for_embedder(self.store.embedder))
            try:
                answer, items = self._ask_routed(question, today)
            finally:
                meaning.current.reset(token)
        else:
            answer, items = self._ask_retrieved(question, today, self.select_relevant(question, today) if self.trimmed
                                                else self.select_full(question, today))
        answer.unconfirmed = self.unconfirmed_used(answer.text, items, today)
        if answer.unconfirmed:
            titles = "; ".join(f"\"{t}\"" for t in answer.unconfirmed)
            answer.text += f"\n\n(This uses something you haven't confirmed yet: {titles}.)"
        answer.duration_ms = round((time.perf_counter() - started) * 1000)
        log_event("answer", user=self.store.user_id, result_count=len(items), duration_ms=answer.duration_ms, stage="answer")
        return answer

    # ------------------------------------------------------------------ S4: routed answers

    def _ask_routed(self, question: str, today: date) -> Tuple[Answer, List[Dict[str, Any]]]:
        saved = self.store.items(include_updates=False)
        # S4e: the person's own name in the third person means them ("Sharada's dentist" → "my dentist"), and a family
        # member's name finds memories that call them by a family term ("Lalitha's eye check-up" → "Amma's eye check-up")
        question = people.own_name_to_my(people.alias_terms(question, saved), saved)
        about_people = people.answer(question, saved, today)
        if about_people is not None:
            log_event("answer", user=self.store.user_id, result_count=0, stage="people")
            return Answer(text=about_people, fact_ids=[], guard="code", route="people", label=LABELS["memory"]), []
        route = route_question(question, today, self.model, saved)
        answer, used = self._dispatch(route, question, today, saved)
        answer.route = route.kind
        answer.label = answer.label or (LABELS["memory"] if route.kind in MEMORY_ROUTES else LABELS.get(route.kind, ""))
        return answer, used

    def _dispatch(self, route, question: str, today: date, saved: List[Dict[str, Any]]) -> Tuple[Answer, List[Dict[str, Any]]]:
        if route.kind == "window":
            header = route.period.header() if route.period and route.named else None
            return self._reword(question, today, window_draft(self.store, *route.window, route.focus, question, named=route.named,
                                                              header=header), route)
        if route.kind == "plan":
            return self._reword(question, today, plan_draft(self.store, *route.window, today), route)
        if route.kind == "people":
            return self._people(route, question, saved)
        if route.kind in ("maths", "mixed", "general", "writing"):
            return self._chat(route, question, today)
        # lookup, status and meaning: the specific-item check first
        if route.specific:
            refusal = self._specific_check(route, question, saved)
            if refusal is not None:
                return refusal, []
        if route.kind == "status":
            candidates = [i for i in self.matches(question)
                          if mentions_all(route.specific, safety.item_words(i))] if route.specific else self.matches(question)
            # S4d: a question about money is about an item with an amount (the trip fee, not the trip)
            if MONEY_QUESTION.search(question) and any(i.get("amount") is not None for i in candidates):
                candidates = [i for i in candidates if i.get("amount") is not None]
            item = candidates[0] if candidates else None
            draft = status_draft(item, self.store.history(item["id"]) if item else [], question)
            return Answer(text=draft.text, fact_ids=[i["id"] for i in draft.items], guard="code"), draft.items
        items = self.select_relevant(question, today)
        if route.specific:
            check = mentions_all if route.kind == "lookup" else mentions_specific
            items = [i for i in items if check(route.specific, safety.item_words(i))] or items
        answer, used = self._ask_retrieved(question, today, items)
        if route.kind == "lookup":
            answer.text = add_missing_time(answer.text, used, today)
        # S5 safety, attribute check on the final answer: an item's date or time may not be given for an event or
        # attribute it doesn't mention ("Mr. Desai's leave" answered from his weekly visit)
        wrong = safety.attribute_check(question, answer.text, saved, today)
        if wrong is not None:
            item, missing = wrong
            log_event("attribute_check", status="replaced", stage="answer")
            answer.first_answer, answer.problems = answer.text, answer.problems + [f"attribute not in memory: {', '.join(missing)}"]
            text = safety.item_words
            named = [i for i in self.store.items(include_updates=False) if all(term_matches(m, text(i)) for m in missing)]
            if named:   # the event is saved, just without a date or time: say so, not someone else's date
                other = named[0]
                answer.text = f"I have {other.get('title') or make_title(other['text'])} saved, but no date or time for it."
                answer.fact_ids, used = [other["id"]], [other]
            else:
                when = "; ".join(when_text(item))
                have = (item.get("title") or make_title(item["text"])) + (f" ({when})" if when else "")
                answer.text, answer.fact_ids, used = f"I have {have}, but nothing about {' '.join(missing)}.", [item["id"]], [item]
            answer.guard = "code"
        return answer, used

    def _specific_check(self, route, question: str, saved: List[Dict[str, Any]]) -> Optional[Answer]:
        """S4c: for lookups and status questions every key term must match one memory ("Infosys" and "interview");
        meaning questions keep the S4 rule (the names must match). If only part matches, say what IS saved."""
        text = safety.item_words
        check = mentions_all if route.kind in ("lookup", "status") else mentions_specific
        if any(check(route.specific, text(i)) for i in saved):
            return None
        log_event("specific_item_check", status="not_saved", stage="answer")
        scored = sorted(((len(matched_terms(route.specific, text(i))), i) for i in saved), key=lambda x: -x[0])
        missing_words = lambda i: [t.lstrip("!") for t in route.specific if not term_matches(t, text(i))]
        if scored and scored[0][0] > 0:
            item = scored[0][1]
            when = "; ".join(when_text(item))
            have = (item.get("title") or make_title(item["text"])) + (f" ({when})" if when else "")
            return Answer(text=f"I have {have}, but nothing about {' '.join(missing_words(item))}.", fact_ids=[item["id"]], guard="code")
        named = " ".join(t.lstrip("!") for t in route.specific if t.startswith("!")) or route.specific[0]
        return Answer(text=f"I don't have anything about {named} saved.", fact_ids=[], guard="code")

    def _people(self, route, question: str, saved: List[Dict[str, Any]]) -> Tuple[Answer, List[Dict[str, Any]]]:
        """People route (structure for the people step): what the person has said that mentions the name."""
        names = [t.lstrip("!") for t in route.specific if t.startswith("!")]
        found = [i for i in saved if names and any(term_matches(n, i["text"]) for n in names)]
        if not found:
            return Answer(text=f"I don't have anything about {' '.join(names) or 'them'} saved.", fact_ids=[], guard="code"), []
        lines = [f"Here's what you've told me about {' '.join(names)}:"] + [f"- {i['text']}" for i in found]
        return Answer(text="\n".join(lines), fact_ids=[i["id"] for i in found], guard="code"), found

    def _chat(self, route, question: str, today: date) -> Tuple[Answer, List[Dict[str, Any]]]:
        if route.kind == "maths" or (route.kind == "mixed" and MATHS_WORDS.search(question)):
            mixed = route.kind == "mixed"
            items = self.select_relevant(question, today) if mixed else []
            text, ok = maths_answer(self.model, question, today, items, mixed)
            return Answer(text=text, fact_ids=[i["id"] for i in items], guard="code" if ok else "fallback",
                          label=LABELS["mixed_maths" if mixed else "maths"]), items
        if route.kind == "general":
            return Answer(text=general_answer(self.model, question, bool(LIVE.search(question))), fact_ids=[], guard="pass",
                          label=LABELS["general"]), []
        items = self.select_relevant(question, today) if route.personal else []
        text, made_up = writing_answer(self.model, question, today, items)
        answer = Answer(text=text, fact_ids=[i["id"] for i in items], guard="pass", label=LABELS["mixed" if route.kind == "mixed" else "writing"])
        if made_up:
            retry, again = writing_answer(self.model, question + " (Use only the dates and amounts given; add none.)", today, items)
            answer.first_answer, answer.problems = text, [f"made up {v}" for v in made_up]
            if not again:
                answer.text, answer.guard = retry, "retry"
            else:
                answer.text = retry + "\n\n(Please check the details: I may have added some you didn't give me: " + ", ".join(again) + ".)"
                answer.guard = "fallback"
        return answer, items

    def _reword(self, question: str, today: date, draft: Draft, route) -> Tuple[Answer, List[Dict[str, Any]]]:
        """S4b: the code's list is shown exactly as built. Qwen writes only a short lead sentence and an optional
        closing line, and only those are checked. One retry; then the list is shown without them (a fallback)."""
        answer = Answer(text=draft.text, fact_ids=[i["id"] for i in draft.items], facts=[draft.text], route=route.kind)
        if not draft.keep and not draft.items:
            answer.guard = "code"
            return answer, []
        messages = [{"role": "system", "content": lead_prompt(today)},
                    {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{draft.text}"}]
        # Owner, after run (d): a lead that fails the check is dropped and the list is shown alone ("lead dropped"),
        # counted apart from full fallbacks. No retry.
        try:
            reply = json.loads(self.model.chat(messages, max_tokens=120, json_schema=LEAD_SCHEMA, purpose="lead").text)
            lead, closing = str(reply.get("lead", "")).strip(), str(reply.get("closing", "")).strip()
        except (ValueError, AttributeError):
            lead, closing = "", ""
        words = (lead + " " + closing).strip()
        problems = lead_problems(words, draft, today, question) if words else ["no lead"]
        # Owner, after run (e1): a lead or closing that reached its length limit was cut off mid-word; drop it
        if len(lead) >= LEAD_MAX or len(closing) >= CLOSING_MAX:
            problems.append("lead hit the length limit")
        if problems:
            answer.text, answer.guard, answer.first_answer, answer.problems = draft.text, "lead_dropped", words, problems
        else:
            answer.text = "\n".join(part for part in (lead, draft.text, closing) if part)
            answer.guard = "pass"
        log_event("answer_guard", user=self.store.user_id, status=answer.guard, result_count=len(answer.problems), stage="lead")
        return answer, draft.items

    # ------------------------------------------------------------------ retrieval, Qwen and the guard (runs (a), (b))

    def _ask_retrieved(self, question: str, today: date, items: List[Dict[str, Any]]) -> Tuple[Answer, List[Dict[str, Any]]]:
        histories = {i["id"]: self.store.history(i["id"]) for i in items}
        text = self._ask_model(question, today, items, histories)
        answer = Answer(text=text, fact_ids=[i["id"] for i in items], facts=[self._line(i, histories[i["id"]], today, question) for i in items])
        # S4b: an old, superseded value counts as invented unless the question asks about it
        allowed_history = histories if (not self.routed or HISTORY_QUESTION.search(question)) else {}

        if self.guard:
            problems = check_answer(text, items, allowed_history, today, question)
            answer.guard = "pass"
            if problems:
                answer.first_answer, answer.problems = text, problems
                top = (self.matches(question) or items)[:RETRY_FACTS]
                top_histories = {i["id"]: self.store.history(i["id"]) for i in top}
                retry = self._ask_model(question, today, top, top_histories)
                if not check_answer(retry, top, top_histories if allowed_history else {}, today, question):
                    answer.text, answer.guard = retry, "retry"
                else:
                    answer.text, answer.guard = code_answer(top, top_histories), "fallback"
                items = top
            log_event("answer_guard", user=self.store.user_id, status=answer.guard, result_count=len(answer.problems), stage="answer")
        return answer, items

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

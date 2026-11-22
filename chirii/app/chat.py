"""
General chat (S4c): maths, general knowledge, and writing or advice, each with an answer label.

    maths       Qwen only writes an arithmetic expression; code checks that every number in it comes from the question,
                from the person's memory (mixed questions) or from a short list of known constants (100, 1000, 3600,
                days in a month…), and the calculator (phase11/calculator.py) works it out. Day counts between dates,
                or until a saved event, are worked out by code alone. If the check fails, Chirii says it couldn't work
                it out reliably instead of guessing.
    general     questions needing live information (today's prices, weather, news, scores) get a fixed answer from
                code: Chirii has no live information. Other questions go to Qwen, told to say "I'm not sure" rather
                than guess.
    writing     letters, messages, tips and advice by Qwen. For personal requests the relevant memories are given, and
                code checks that no date or rupee amount was made up (each must be in memory or in the question).
"""

import json
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from chirii.app.formatting import say_amount, say_date
from chirii.app.llm.client import ChatModel, LLMError
from chirii.app.periods import find_event
from chirii.eval.scoring import find_amounts, find_dates
from phase11.calculator import calculate

LABELS = {"memory": "From your memory", "maths": "Calculated", "mixed_maths": "Calculated from your memory",
          "general": "General knowledge", "writing": "Writing help", "mixed": "Advice using your memory",
          "not_saved": "From your memory"}
NO_LIVE_ANSWER = ("I can't check live information such as today's prices, weather, news or match results. "
                  "Please look at a trusted live source for that.")
CONSTANTS = {2, 3, 4, 7, 10, 12, 24, 28, 29, 30, 31, 52, 60, 100, 365, 1000, 3600, 3.6, 100000, 10000000}
MONEY_WORDS = re.compile(r"₹|\brs\b|rupees?|lakh|crore|\b(cost|costs|price|pay|paid|gst|tax|interest|emi|rent|bill|fee|salary|"
                         r"spend|spent|owe|owes|instalments?|installments?)\b", re.I)
EXPRESSION = re.compile(r"^[\d\s+\-*/().,%]+$")


def numbers_in(text: str) -> List[float]:
    found = []
    for m in re.finditer(r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|k\b)?", text, re.I):
        value = float(m.group(1).replace(",", ""))
        found.append(value)
        unit = (m.group(2) or "").lower()
        if unit.startswith("la"):
            found.append(value * 100000)
        elif unit.startswith("cr"):
            found.append(value * 10000000)
        elif unit == "k":
            found.append(value * 1000)
    return found


def allowed_numbers(question: str, amounts: List[int]) -> set:
    base = set(numbers_in(question)) | {float(a) for a in amounts} | {float(c) for c in CONSTANTS}
    return base | {round(n / 100, 6) for n in base} | {round(n * 100, 6) for n in base}


def expression_ok(expression: str, allowed: set) -> bool:
    if not expression or not EXPRESSION.match(expression):
        return False
    return all(any(abs(float(n.replace(",", "")) - a) < 1e-6 for a in allowed) for n in re.findall(r"\d+(?:\.\d+)?", expression.replace(",", "")))


def show_number(value: float, money: bool) -> str:
    if money:
        return say_amount(int(round(value))) if abs(value - round(value)) < 1e-9 else f"₹{value:,.2f}"
    return f"{int(round(value)):,}" if abs(value - round(value)) < 1e-9 else f"{value:,.2f}".rstrip("0").rstrip(".")


def days_answer(question: str, today: date, items: List[Dict[str, Any]]) -> Optional[str]:
    """'How many days from 3 March to 18 April?' or 'How many days until Kamala arrives?', by code alone."""
    if not re.search(r"\bhow many days\b|\bdays (from|between|until|till|to|before|left)\b", question, re.I):
        return None
    dates = sorted(find_dates(question, today))
    if len(dates) >= 2:
        full = []
        for day, month in dates[:2]:
            year = today.year if date(today.year, month, 1) >= today.replace(day=1) - timedelta(days=200) else today.year + 1
            full.append(date(year, month, day))
        start, end = sorted(full)
        return f"There are {(end - start).days} days from {say_date(start.isoformat())} to {say_date(end.isoformat())}."
    m = re.search(r"\b(until|till|to|before)\s+(.+?)[?.!]*$", question, re.I)
    if m and items:
        event = find_event(m.group(2), today, items)
        if event:
            when = date.fromisoformat(event["date"])
            return f"{(when - today).days} days: {event.get('title') or event['text']} is on {say_date(event['date'])}."
    return None


def maths_answer(model: ChatModel, question: str, today: date, items: List[Dict[str, Any]], mixed: bool) -> Tuple[str, bool]:
    """Returns the answer and whether it was worked out."""
    by_days = days_answer(question, today, items)
    if by_days:
        return by_days, True
    amounts = [i["remaining"] if i.get("remaining") is not None else i["amount"] for i in items if i.get("amount") is not None] if mixed else []
    facts = "\n".join(f"- {i.get('title') or i['text']}: ₹{(i['remaining'] if i.get('remaining') is not None else i['amount'])}"
                      + (f" on {i['date']}" if i.get("date") else "") + (f" on day {i['day_of_month']} of each month" if i.get("day_of_month") else "")
                      for i in items if i.get("amount") is not None) if mixed else ""
    system = ("Write ONE arithmetic expression that answers the question. Use only numbers from the question"
              + (" and from the saved amounts below" if mixed else "") + ", with + - * / and brackets. For a percentage write "
              "it as /100. Do not work it out. Reply as JSON {\"expression\": \"...\"}."
              + (f"\nSaved amounts:\n{facts}" if facts else ""))
    schema = {"type": "object", "properties": {"expression": {"type": "string", "maxLength": 80}}, "required": ["expression"]}
    try:
        expression = json.loads(model.chat([{"role": "system", "content": system}, {"role": "user", "content": question}],
                                           max_tokens=60, json_schema=schema, purpose="maths").text)["expression"]
    except (ValueError, KeyError, TypeError):
        expression = ""
    expression = expression.replace("×", "*").replace("÷", "/").replace("x", "*").strip()
    if not expression_ok(expression, allowed_numbers(question, amounts)):
        return "I couldn't work that out reliably. Could you write out the numbers, for example \"2450 × 18%\"?", False
    result = calculate(expression.replace(",", ""))
    if not result.ok:
        return "I couldn't work that out reliably. Could you write out the numbers?", False
    money = bool(MONEY_WORDS.search(question)) or mixed
    shown = expression.replace("*", " × ").replace("/", " ÷ ").replace("  ", " ")
    return f"{show_number(result.value, money)}  (worked out: {shown} = {show_number(result.value, False)})", True


def general_answer(model: ChatModel, question: str, live: bool) -> str:
    if live:
        return NO_LIVE_ANSWER
    system = ("You answer general-knowledge questions briefly and correctly, in one to three sentences. If you are not sure, "
              "or the question is about a small local place or a private person you cannot know about, say \"I'm not sure\" and "
              "do not guess any name or number. You have no live information: for today's prices, weather or news, say you "
              "can't check live information.")
    # a model error is passed up, so the app can say clearly that the model isn't running
    return model.chat([{"role": "system", "content": system}, {"role": "user", "content": question}], max_tokens=200, purpose="general").text


def made_up_values(reply: str, question: str, today: date, items: List[Dict[str, Any]]) -> List[str]:
    """Dates or rupee amounts in a written reply that are not in the memory given, the question, or the next few days."""
    allowed_dates = {(d.day, d.month) for d in (today + timedelta(days=k) for k in range(0, 3))} | find_dates(question, today)
    allowed_amounts = set(find_amounts(question))
    for i in items:
        for key in ("date", "end_date"):
            if i.get(key):
                d = date.fromisoformat(i[key])
                allowed_dates.add((d.day, d.month))
        if i.get("amount") is not None:
            allowed_amounts.add(i["amount"])
    bad = [f"{d}/{m}" for d, m in find_dates(reply, today) - allowed_dates]
    bad += [say_amount(a) for a in find_amounts(reply) - allowed_amounts]
    return bad


def writing_answer(model: ChatModel, question: str, today: date, items: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    memory = "\n".join(f"- {i['text']}" for i in items)
    system = ("You are Chirii, a helpful personal assistant. Help with writing, messages, tips and everyday advice. Be warm, "
              "practical and brief (under 120 words). Use numbered points for tips or steps. Never make up dates, amounts, "
              "names or facts about the person; use only what the question or the saved memories say. For health questions, "
              "give general care tips and say to follow their doctor's advice."
              f"\nToday is {say_date(today.isoformat())}."
              + (f"\nSaved memories that may help:\n{memory}" if memory else ""))
    reply = model.chat([{"role": "system", "content": system}, {"role": "user", "content": question}], max_tokens=256, purpose="writing").text
    return reply, made_up_values(reply, question, today, items)

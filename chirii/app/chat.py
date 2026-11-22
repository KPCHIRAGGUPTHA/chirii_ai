"""
General chat (S4c, S4d): maths, general knowledge, and writing or advice, each with an answer label.

Labels (owner's spec): "From your memories" for answers from memory, "Calculated" for maths, "General answer" for
general knowledge, writing and advice.

    maths       S4d formula library. Qwen only chooses a formula and its arguments, inside a strict JSON schema, using
                numbers from the question (and, for mixed questions, saved amounts). Code checks every number, computes
                the result and shows brief working in Indian format ("₹4,250 × 18% = ₹765"). Days and weeks between
                dates use the real calendar, by code alone. A free expression is allowed only for plain arithmetic, and
                a constant only where the question's words call for it (a % sign allows 100, "half" allows 2, a month
                name allows its real number of days).
    general     questions needing live information get a fixed answer from code. Indian everyday terms (UPI, GST, IFSC,
                TDS, CIBIL…) are answered from a curated glossary. Other questions go to Qwen, told to say "I'm not
                sure" rather than guess.
    writing     letters, messages, tips and advice by Qwen. For personal requests the relevant memories are given, and
                code checks that no date or rupee amount was made up.
"""

import json
import math
import re
from calendar import monthrange
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from chirii.app.formatting import say_date
from chirii.app.glossary import glossary_answer
from chirii.app.llm.client import ChatModel
from chirii.app.periods import find_event
from chirii.eval.scoring import MONTHS, find_amounts, find_dates
from phase11.calculator import calculate

MEMORY, CALCULATED, GENERAL = "From your memories", "Calculated", "General answer"
LABELS = {"memory": MEMORY, "maths": CALCULATED, "mixed_maths": CALCULATED, "general": GENERAL, "writing": GENERAL,
          "mixed": GENERAL, "not_saved": MEMORY}
NO_LIVE_ANSWER = ("I can't check live information such as today's prices, weather, news or match results. "
                  "Please look at a trusted live source for that.")
MONEY_WORDS = re.compile(r"₹|\brs\b|rupees?|lakh|crore|\b(cost|costs|price|pay|paid|gst|tax|interest|emi|loan|rent|bill|fee|salary|"
                         r"spend|spent|owe|owes|instalments?|installments?|discount|off)\b", re.I)

UNITS = {  # to a base unit
    "km": ("length", 1000), "kilometre": ("length", 1000), "m": ("length", 1), "metre": ("length", 1), "cm": ("length", 0.01),
    "mm": ("length", 0.001), "mile": ("length", 1609.344), "foot": ("length", 0.3048), "inch": ("length", 0.0254),
    "kg": ("mass", 1000), "g": ("mass", 1), "pound": ("mass", 453.59237),
    "litre": ("volume", 1), "ml": ("volume", 0.001),
    "km/h": ("speed", 1000 / 3600), "m/s": ("speed", 1),
    "rupees": ("money", 1), "thousand": ("money", 1000), "lakh": ("money", 100000), "crore": ("money", 10000000),
    "celsius": ("temperature", None), "fahrenheit": ("temperature", None),
}
FORMULAS = {
    "percent_of":        "numbers = [percent, amount]: percent of an amount (also GST or tax amount on a price)",
    "add_percent":       "numbers = [amount, percent]: amount plus a percentage (price including GST or tax)",
    "discount":          "numbers = [amount, percent]: price after a percentage off",
    "percent_change":    "numbers = [old value, new value]: percentage rise or fall",
    "simple_interest":   "numbers = [principal, yearly rate %, years]: simple interest earned",
    "compound_interest": "numbers = [principal, yearly rate %, years, times compounded per year (1 if not said)]: interest earned",
    "emi":               "numbers = [loan amount, yearly rate %, years]: monthly EMI on a reducing-balance loan",
    "split":             "numbers = [amount, amount, ..., number of people]: total of the amounts shared equally",
    "sum":               "numbers = [amount, amount, ...]: total",
    "convert":           "numbers = [value], from_unit and to_unit: a unit conversion (also rupees, thousand, lakh, crore)",
    "arithmetic":        "expression: any other plain arithmetic, written with numbers from the question only",
}


def indian(value: float, decimals: int = 2) -> str:
    """1,00,000 style grouping; at most two decimals; no trailing zeros."""
    negative, value = value < 0, abs(value)
    whole = int(math.floor(value + 1e-9))
    frac = round(value - whole, decimals)
    if frac >= 1:
        whole, frac = whole + 1, 0
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join(([head] if head else []) + groups) + "," + tail
    text = digits + (f"{frac:.{decimals}f}"[1:].rstrip("0").rstrip(".") if frac else "")
    return ("-" if negative else "") + text


def money(value: float) -> str:
    return "₹" + indian(value)


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


def justified_constants(question: str, today: date) -> set:
    """Constants a plain-arithmetic expression may use, only where the question's words call for them."""
    q = question.lower()
    allowed = set()
    if "%" in q or "percent" in q:
        allowed.add(100.0)
    if re.search(r"\b(half|halve|double|twice|two)\b", q):
        allowed.add(2.0)
    if re.search(r"\bdozen\b", q):
        allowed.add(12.0)
    if re.search(r"\bweeks?\b", q):
        allowed.add(7.0)
    if re.search(r"\b(per|each|a) month\b|\bmonths?\b", q) and re.search(r"\byears?\b", q):
        allowed.add(12.0)
    for name, number in MONTHS.items():
        if re.search(rf"\b{name}\b", q):                      # a month named: its real number of days, nearest year
            year = min((today.year - 1, today.year, today.year + 1), key=lambda y: abs((date(y, number, 15) - today).days))
            allowed.add(float(monthrange(year, number)[1]))
    return allowed


def number_ok(n: float, allowed: List[float]) -> bool:
    return any(abs(n - a) < 1e-6 for a in allowed)


def days_answer(question: str, today: date, items: List[Dict[str, Any]]) -> Optional[str]:
    """Days or weeks between two dates, or until a saved event, by code alone (real calendar)."""
    weeks = re.search(r"\bhow many weeks\b|\bweeks (from|between|until|till|to)\b", question, re.I)
    if not (weeks or re.search(r"\bhow many days\b|\bdays (from|between|until|till|to|before|left)\b", question, re.I)):
        return None
    year_match = re.search(r"\b(20\d\d)\b", question)
    full = []
    for day, month in sorted(find_dates(question, today), key=lambda dm: question.lower().find(str(dm[0]))):
        year = int(year_match.group(1)) if year_match else (today.year if date(today.year, month, min(day, 28)) >= today - timedelta(days=200) else today.year + 1)
        full.append(date(year, month, day))
    if len(full) >= 2:
        start, end = sorted(full[:2])
        days = (end - start).days
        if weeks:
            return f"{indian(days / 7)} weeks  ({say_date(start.isoformat())} to {say_date(end.isoformat())} = {days} days = {indian(days / 7)} weeks)"
        return f"{days} days  ({say_date(start.isoformat())} to {say_date(end.isoformat())} = {days} days)"
    m = re.search(r"\b(until|till|to|before)\s+(.+?)[?.!]*$", question, re.I)
    if m and items:
        event = find_event(m.group(2), today, items)
        if event:
            days = (date.fromisoformat(event["date"]) - today).days
            return f"{days} days  (today to {say_date(event['date'])}, {event.get('title') or event['text']} = {days} days)"
    return None


def compute(formula: str, nums: List[float], from_unit: Optional[str], to_unit: Optional[str], question: str) -> Optional[Tuple[float, str]]:
    """The result and its working, or None if the arguments don't fit the formula."""
    m = bool(MONEY_WORDS.search(question))
    fmt = money if m else indian
    try:
        if formula == "percent_of" and len(nums) == 2:
            p, x = nums
            return x * p / 100, f"{fmt(x)} × {indian(p)}% = {fmt(x * p / 100)}"
        if formula == "add_percent" and len(nums) == 2:
            x, p = nums
            return x * (1 + p / 100), f"{fmt(x)} + {indian(p)}% = {fmt(x)} + {fmt(x * p / 100)} = {fmt(x * (1 + p / 100))}"
        if formula == "discount" and len(nums) == 2:
            x, p = nums
            return x * (1 - p / 100), f"{fmt(x)} − {indian(p)}% = {fmt(x)} − {fmt(x * p / 100)} = {fmt(x * (1 - p / 100))}"
        if formula == "percent_change" and len(nums) == 2 and nums[0]:
            a, b = nums
            r = (b - a) / a * 100
            return r, f"({indian(b)} − {indian(a)}) ÷ {indian(a)} × 100 = {indian(r)}%"
        if formula == "simple_interest" and len(nums) == 3:
            p, r, t = nums
            return p * r * t / 100, f"{money(p)} × {indian(r)}% × {indian(t)} years = {money(p * r * t / 100)}"
        if formula == "compound_interest" and len(nums) in (3, 4):
            p, r, t = nums[:3]
            n = nums[3] if len(nums) == 4 and nums[3] else 1
            amount = p * (1 + r / 100 / n) ** (n * t)
            return amount - p, f"{money(p)} × (1 + {indian(r)}%{'/' + indian(n) if n != 1 else ''})^{indian(n * t)} = {money(amount)}; interest = {money(amount)} − {money(p)} = {money(amount - p)}"
        if formula == "emi" and len(nums) == 3:
            p, r, years = nums
            i, n = r / 12 / 100, round(years * 12)
            emi = p / n if i == 0 else p * i * (1 + i) ** n / ((1 + i) ** n - 1)
            return emi, f"EMI on {money(p)} at {indian(r)}% a year for {indian(years)} years ({n} months) = {money(emi)} a month"
        if formula == "split" and len(nums) >= 2 and nums[-1] >= 1:
            parts, people = nums[:-1], nums[-1]
            total = sum(parts)
            shown = " + ".join(fmt(x) for x in parts)
            return total / people, (f"{shown} = {fmt(total)}; " if len(parts) > 1 else "") + f"{fmt(total)} ÷ {indian(people)} = {fmt(total / people)} each"
        if formula == "sum" and len(nums) >= 2:
            return sum(nums), " + ".join(indian(x) for x in nums) + f" = {fmt(sum(nums))}"
        if formula == "convert" and len(nums) == 1 and from_unit in UNITS and to_unit in UNITS:
            (kind1, f1), (kind2, f2) = UNITS[from_unit], UNITS[to_unit]
            v = nums[0]
            if kind1 != kind2:
                return None
            if kind1 == "temperature":
                r = (v - 32) * 5 / 9 if from_unit == "fahrenheit" else v * 9 / 5 + 32
                return r, f"{indian(v)}° {from_unit.capitalize()} = {indian(r)}° {to_unit.capitalize()}"
            r = v * f1 / f2
            return r, f"{indian(v)} {from_unit} = {indian(r)} {to_unit}"
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    return None


def maths_answer(model: ChatModel, question: str, today: date, items: List[Dict[str, Any]], mixed: bool) -> Tuple[str, bool]:
    """Returns the answer and whether it was worked out by code."""
    by_days = days_answer(question, today, items)
    if by_days:
        return by_days, True
    saved = [float(i["remaining"] if i.get("remaining") is not None else i["amount"]) for i in items if i.get("amount") is not None] if mixed else []
    allowed = numbers_in(question) + saved
    facts = "\n".join(f"- {i.get('title') or i['text']}: {i['remaining'] if i.get('remaining') is not None else i['amount']} rupees"
                      for i in items if i.get("amount") is not None) if mixed else ""
    schema = {"type": "object", "properties": {
        "formula": {"type": "string", "enum": list(FORMULAS)},
        "numbers": {"type": "array", "items": {"type": "number"}, "maxItems": 8},
        "from_unit": {"anyOf": [{"type": "string", "enum": list(UNITS)}, {"type": "null"}]},
        "to_unit": {"anyOf": [{"type": "string", "enum": list(UNITS)}, {"type": "null"}]},
        "expression": {"anyOf": [{"type": "string", "maxLength": 80}, {"type": "null"}]}},
        "required": ["formula", "numbers", "from_unit", "to_unit", "expression"]}
    system = ("Choose the formula that answers the question and give its numbers, in the order shown, using only numbers "
              "written in the question" + (" or the saved amounts below" if mixed else "") + ". Write lakh and crore as the "
              "number in the question with from_unit, e.g. 1.2 crore → numbers [1.2], from_unit crore. Do not calculate.\n"
              "Formulas:\n" + "\n".join(f"- {k}: {v}" for k, v in FORMULAS.items()) + (f"\nSaved amounts:\n{facts}" if facts else ""))
    try:
        choice = json.loads(model.chat([{"role": "system", "content": system}, {"role": "user", "content": question}],
                                       max_tokens=120, json_schema=schema, purpose="maths").text)
    except (ValueError, TypeError):
        choice = {}
    sorry = "I couldn't work that out reliably. Could you write out the numbers, for example \"2,450 × 18%\"?"
    formula = choice.get("formula")
    if formula == "arithmetic":
        expression = (choice.get("expression") or "").replace("×", "*").replace("÷", "/").replace(",", "").strip()
        if not re.fullmatch(r"[\d\s+\-*/().]+", expression or "x"):
            return sorry, False
        ok = allowed + list(justified_constants(question, today))
        if not all(number_ok(float(n), ok) for n in re.findall(r"\d+(?:\.\d+)?", expression)):
            return sorry, False
        result = calculate(expression)
        if not result.ok:
            return sorry, False
        fmt = money if MONEY_WORDS.search(question) or mixed else indian
        shown = re.sub(r"\d+(?:\.\d+)?", lambda m: indian(float(m.group(0))), expression).replace("*", " × ").replace("/", " ÷ ")
        return f"{fmt(result.value)}  ({re.sub(' +', ' ', shown)} = {fmt(result.value)})", True
    nums = [float(n) for n in choice.get("numbers") or []]
    allowed = allowed + list(justified_constants(question, today))     # "half" allows 2, a % sign allows 100
    if not formula or not nums or not all(number_ok(n, allowed) for n in nums):
        return sorry, False
    done = compute(formula, nums, choice.get("from_unit"), choice.get("to_unit"), question)
    if done is None:
        return sorry, False
    value, working = done
    unit = choice.get("to_unit") if formula == "convert" and choice.get("to_unit") not in (None, "rupees") else ""
    shown = indian(value) + (f" {unit}" if unit else "") if formula in ("convert", "percent_change") or not (MONEY_WORDS.search(question) or mixed) else money(value)
    if formula == "percent_change":
        shown = indian(value) + "%"
    return f"{shown}  ({working})", True


def general_answer(model: ChatModel, question: str, live: bool) -> str:
    if live:
        return NO_LIVE_ANSWER
    from_glossary = glossary_answer(question)
    if from_glossary:
        return from_glossary
    system = ("You answer general-knowledge questions briefly and correctly, in one to three sentences. If you are not sure, "
              "or the question is about a small local place or a private person you cannot know about, say \"I'm not sure\" and "
              "do not guess any name or number. You have no live information: for today's prices, weather or news, say you "
              "can't check live information.")
    # a model error is passed up, so the app can say clearly that the model isn't running
    return model.chat([{"role": "system", "content": system}, {"role": "user", "content": question}], max_tokens=200, purpose="general").text


def made_up_values(reply: str, question: str, today: date, items: List[Dict[str, Any]]) -> List[str]:
    """Dates or rupee amounts in a written reply that are not in the memory given, the question, or the next few days."""
    allowed_dates = {(d.day, d.month) for d in (today + timedelta(days=k) for k in range(0, 3))} | find_dates(question, today)
    for day, month in list(find_dates(question, today)):               # the day after a date in the question ("back on 20th")
        try:
            nxt = date(today.year, month, day) + timedelta(days=1)
            allowed_dates.add((nxt.day, nxt.month))
        except ValueError:
            pass
    allowed_amounts = set(find_amounts(question))
    for i in items:
        for key in ("date", "end_date"):
            if i.get(key):
                d = date.fromisoformat(i[key])
                allowed_dates.add((d.day, d.month))
        if i.get("amount") is not None:
            allowed_amounts.add(i["amount"])
    bad = [f"{d}/{m}" for d, m in find_dates(reply, today) - allowed_dates]
    bad += [money(a) for a in find_amounts(reply) - allowed_amounts]
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

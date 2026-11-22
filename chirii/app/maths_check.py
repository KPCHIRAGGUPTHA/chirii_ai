"""
S4f: an independent code computation for maths questions, used to check Qwen's choice.

Code reads the common question shapes itself: unit and lakh/crore conversions, "X% of Y", "Y plus X% GST", "Y after
X% discount", simple and compound interest, EMI, and splitting an amount among people. It gives an answer only when
it used every number in the question; otherwise it says nothing and Qwen's (checked) choice stands alone.

maths_answer (chat.py) then:
  - Qwen's choice and code agree → Qwen's working is shown (the same number);
  - they disagree → the code result is shown, with its working;
  - only one of them has a result → that one;
  - neither → "I couldn't work that out reliably".
A number that fails these checks is never shown.
"""

import re
from typing import Dict, List, Optional, Tuple

NUM = r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
UNIT_WORDS: List[Tuple[str, str]] = [
    (r"km/h|kmph|kilomet(?:re|er)s? per hour", "km/h"), (r"m/s", "m/s"),
    (r"km|kms|kilomet(?:re|er)s?", "km"), (r"miles?", "mile"), (r"cm|centimet(?:re|er)s?", "cm"), (r"mm|millimet(?:re|er)s?", "mm"),
    (r"met(?:re|er)s?|m", "metre"), (r"feet|foot|ft", "foot"), (r"inch(?:es)?", "inch"),
    (r"kgs?|kilograms?|kilos?", "kg"), (r"grams?|gms?|g", "g"), (r"pounds?|lbs?", "pound"),
    (r"ml|millilit(?:re|er)s?", "ml"), (r"lit(?:re|er)s?|l", "litre"),
    (r"°\s*c|degrees? c(?:elsius)?|celsius|centigrade", "celsius"), (r"°\s*f|degrees? f(?:ahrenheit)?|fahrenheit", "fahrenheit"),
    (r"lakhs?|lacs?", "lakh"), (r"crores?|cr", "crore"), (r"thousands?", "thousand"), (r"rupees?|rs\.?", "rupees"),
]
UNIT = "(" + "|".join(p for p, _ in UNIT_WORDS) + r")\b"
# Most specific first: "half-yearly" also contains "yearly", and "at 8% a year, compounded quarterly" says both
FREQUENCY = {r"\bhalf[- ]?yearly\b|\bsemi[- ]?annual": 2, r"\bquarterly\b": 4, r"\bcompounded monthly\b|\bmonthly compounding\b": 12,
             r"\b(yearly|annually|per annum|every year|a year)\b": 1}


def frequency(question: str) -> Optional[float]:
    q = question.lower()
    return next((v for p, v in FREQUENCY.items() if re.search(p, q)), None)


def unit_name(word: str) -> Optional[str]:
    w = word.lower().strip()
    for pattern, name in UNIT_WORDS:
        if re.fullmatch(pattern, w):
            return name
    return None


def value(text: str) -> float:
    return float(text.replace(",", ""))


def tokens(question: str) -> List[Dict]:
    """Each number in the question with what it is: a percent, years, months, people, or an amount (₹, lakh, crore)."""
    q = question.replace("₹ ", "₹")
    out = []
    for m in re.finditer(r"(₹)?" + NUM + r"\s*(%|percent\b|lakhs?\b|lacs?\b|crores?\b|k\b|years?\b|yrs?\b|months?\b|people\b|persons?\b|friends?\b)?", q, re.I):
        v, tag = value(m.group(2)), (m.group(3) or "").lower()
        if tag in ("%", "percent"):
            out.append({"v": v, "tag": "pct"})
        elif tag.startswith(("lakh", "lac")):
            out.append({"v": v * 100000, "tag": "money"})
        elif tag.startswith("crore"):
            out.append({"v": v * 10000000, "tag": "money"})
        elif tag == "k":
            out.append({"v": v * 1000, "tag": "money"})
        elif tag.startswith(("year", "yr")):
            out.append({"v": v, "tag": "years"})
        elif tag.startswith("month"):
            out.append({"v": v, "tag": "months"})
        elif tag.startswith(("people", "person", "friend")):
            out.append({"v": v, "tag": "people", "friends": tag.startswith("friend")})
        else:
            out.append({"v": v, "tag": "money" if m.group(1) else "plain"})
    return out


def conversion(question: str) -> Optional[Tuple[str, List[float], str, str]]:
    """"Convert 30 km to miles", "Write ₹1,25,00,000 in crore", "1.2 crore in lakh", "38.5°C in Fahrenheit"."""
    q = question.replace("₹ ", "₹")
    target = re.search(r"\b(?:to|in|into)\s+" + UNIT + r"\s*[?.!]*\s*$", q, re.I)
    if not target:
        return None
    to_unit = unit_name(target.group(1))
    source = re.search(r"(₹)?" + NUM + r"\s*(?:" + UNIT + ")?", q[: target.start()], re.I)
    if not source or to_unit is None:
        return None
    from_unit = unit_name(source.group(3)) if source.group(3) else ("rupees" if source.group(1) else None)
    if from_unit is None or from_unit == to_unit:
        return None
    if len(re.findall(NUM, q)) != 1:
        return None
    return "convert", [value(source.group(2))], from_unit, to_unit


def independent(question: str) -> Optional[Tuple[str, List[float], Optional[str], Optional[str]]]:
    """The formula and numbers code reads from the question by itself, or None when it can't be sure."""
    q = question.lower()
    found = conversion(question)
    if found:
        return found
    t = tokens(question)
    pct = [x["v"] for x in t if x["tag"] == "pct"]
    amounts = [x["v"] for x in t if x["tag"] in ("money", "plain")]
    years = [x["v"] for x in t if x["tag"] == "years"] + [x["v"] / 12 for x in t if x["tag"] == "months"]
    people = [x for x in t if x["tag"] == "people"]

    def used(n):                                    # every number in the question was used
        return n == len(t)

    if re.search(r"\bemi\b", q) and len(amounts) == 1 and len(pct) == 1 and len(years) == 1 and used(3):
        return "emi", [amounts[0], pct[0], years[0]], None, None
    if re.search(r"\bcompound(ed)? interest\b|\bcompounded\b", q) and len(amounts) == 1 and len(pct) == 1 and len(years) == 1 and used(3):
        n = frequency(question) or 1
        return "compound_interest", [amounts[0], pct[0], years[0], n], None, None
    if re.search(r"\bsimple interest\b", q) and len(amounts) == 1 and len(pct) == 1 and len(years) == 1 and used(3):
        return "simple_interest", [amounts[0], pct[0], years[0]], None, None
    if len(pct) == 1 and len(amounts) == 1 and used(2):
        if re.search(r"\d\s*(%|percent)\s*of\b", q):
            return "percent_of", [pct[0], amounts[0]], None, None
        if re.search(r"\b(discount|off|less)\b", q):
            return "discount", [amounts[0], pct[0]], None, None
        if re.search(r"\b(plus|\+|add|including|inclusive|with)\b", q) and re.search(r"\b(gst|tax|tip|service charge|vat)\b", q):
            return "add_percent", [amounts[0], pct[0]], None, None
    if re.search(r"\b(split|share|divide)\b", q) and len(amounts) == 1:
        me_and = re.search(r"\b(me|myself|i) and (\d+)\b", q)
        count = int(me_and.group(2)) + 1 if me_and else None
        if count is None:
            m = re.search(r"\b(among|between|by|into)\s+(\d+)\b", q)
            count = int(m.group(2)) if m else None
        if count and used(2):
            return "split", [amounts[0], float(count)], None, None
    return None


def agree(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, abs(b) * 1e-6)


def compounding_ok(question: str, n: float) -> bool:
    """Compound interest's 4th number (times a year) is justified by the question's words, or 1 when none are said."""
    return n == (frequency(question) or 1)


def unit_factors(question: str) -> set:
    """Conversion factors that the units named in the question justify, for a plain expression."""
    q = question.lower()
    names = {unit_name(m.group(1)) for m in re.finditer(r"\b" + UNIT, q, re.I)} | ({"celsius"} if "°c" in q.replace(" ", "") else set()) \
        | ({"fahrenheit"} if "°f" in q.replace(" ", "") else set())
    factors = set()
    table = {"mile": {1.609344, 1.609, 1.61, 1.6, 0.621371, 0.6214, 0.62}, "km": {1000.0, 1.609344, 1.609, 1.61, 0.621371, 0.6214, 0.62},
             "fahrenheit": {9.0, 5.0, 32.0, 1.8}, "celsius": {9.0, 5.0, 32.0, 1.8}, "lakh": {100000.0, 100.0}, "crore": {10000000.0, 100.0},
             "thousand": {1000.0}, "foot": {0.3048, 12.0, 30.48}, "inch": {2.54, 0.0254, 12.0}, "pound": {0.453592, 2.20462, 453.59237, 2.2046},
             "kg": {1000.0, 2.20462, 0.453592, 2.2046}, "g": {1000.0}, "litre": {1000.0}, "ml": {1000.0}, "cm": {100.0, 2.54}, "m": {100.0, 1000.0},
             "metre": {100.0, 1000.0, 0.3048}, "km/h": {3.6, 1000.0, 3600.0}, "m/s": {3.6}}
    for name in names:
        factors |= table.get(name, set())
    return factors

"""
Phase 11B — Goal eval scorer v3: scorer_v2 with the minus sign read. Every other rule is inherited from v2, which stays unchanged.

v2 dropped the sign, so "-43" matched 43. v3 reads a number as negative when:
  - a minus touches it and does not follow a number or word: "-43", "(-43)", "x=-43", "$-36", "-$36", "-three"
  - a spaced minus follows "=" or a linking word: "= - 43", "is - 43"
  - the word "minus" or "negative" comes before it and does not follow a number: "minus 43", "negative three"

These stay subtraction, ranges or hyphens, so their numbers are positive: "5 - 3", "5-3", "7 minus 3", "10-year", "covid-19", "twenty-one".
A line that opens with "- 43" is a bullet, so it reads 43.

The echo rule counts a negative answer as new text unless the prompt has that same signed number, so "3 - 6 = -3" is not an echo of "What is 3 - 6?".
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase9.goal_eval import scorer_v2 as v2
from phase9.goal_eval.scorer_v2 import (  # noqa: F401  re-exported so callers need only this module
    MAGNITUDES, NUMBER_WORDS, answer_region, clean, contains_term, first_line, first_sentence, has_forbidden, is_refusal, lists_multiple_options,
    load_goal_set, offers_support, project_path, score_exact, score_keyword, score_unknown, slug, summarize, write_csv, fails_rag_limits,
)

SCORER_VERSION = "v3"
MINUS_CHARACTERS = str.maketrans({"−": "-", "–": "-", "—": "-"})
SIGN_WORDS = ("minus", "negative")
NUMBER_WORD_PATTERN = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
TOUCHING_MINUS = rf"(?<![a-z0-9)\]%])-(?=\$?\d|(?:{NUMBER_WORD_PATTERN})\b)"
PARTS = re.compile(rf"{TOUCHING_MINUS}|\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+|[^\sa-z0-9-]")
SPACED_MINUS = re.compile(r"(=|\b(?:is|are|was|equals|makes|gives))\s*-\s+(?=\$?\d)")


def starts_number(parts: List[str], index: int) -> bool:
    if index >= len(parts):
        return False
    part = parts[index]
    if part == "$":
        return starts_number(parts, index + 1)
    return part[0].isdigit() or part in NUMBER_WORDS or part in MAGNITUDES or (part == "a" and index + 1 < len(parts) and parts[index + 1] in MAGNITUDES)


def extract_numbers(text: str) -> List[float]:
    """Same reading as v2, with signs. '10 million' is one number, and 'twenty-one' is 21."""
    parts = PARTS.findall(SPACED_MINUS.sub(r"\1 -", clean(text).translate(MINUS_CHARACTERS)))
    numbers: List[float] = []
    index = 0
    sign = 1.0
    after_number = False
    while index < len(parts):
        part = parts[index]
        if part == "-" or (part in SIGN_WORDS and not after_number and starts_number(parts, index + 1)):
            sign = -1.0
            after_number = False
            index += 1
        elif part[0].isdigit():
            value = float(part.replace(",", ""))
            index += 1
            while index < len(parts) and parts[index] in MAGNITUDES:
                value *= MAGNITUDES[parts[index]]
                index += 1
            numbers.append(sign * value)
            sign, after_number = 1.0, True
        elif part in NUMBER_WORDS or part in MAGNITUDES or (part == "a" and index + 1 < len(parts) and parts[index + 1] in MAGNITUDES):
            total, current = 0.0, 0.0
            while index < len(parts):
                part = parts[index]
                if part in NUMBER_WORDS:
                    current += NUMBER_WORDS[part]
                elif part == "a" and current == 0 and index + 1 < len(parts) and parts[index + 1] in MAGNITUDES:
                    pass
                elif part == "hundred":
                    current = max(current, 1) * 100
                elif part in MAGNITUDES:
                    total += max(current, 1) * MAGNITUDES[part]
                    current = 0
                elif part == "and" and index + 1 < len(parts) and parts[index + 1] in NUMBER_WORDS:
                    pass
                else:
                    break
                index += 1
            numbers.append(sign * (total + current))
            sign, after_number = 1.0, True
        else:
            after_number = False
            index += 1
    return numbers


def answer_number(region: str, prompt: str = "") -> Optional[float]:
    """The first number, skipping a restated expression, so '2 + 2 is 4' reads 4 for the prompt '2+2'."""
    prompt_numbers = set(extract_numbers(prompt))
    start = 0
    for match in re.finditer(r"=|\b(?:is|are|equals|makes|gives)\b", clean(region)):
        before = extract_numbers(region[:match.start()])
        if before and set(before) <= prompt_numbers:
            start = match.end()
    numbers = extract_numbers(region[start:]) or extract_numbers(region)
    return numbers[0] if numbers else None


def adds_negative_answer(region: str, item: Dict[str, Any]) -> bool:
    """The reply states a negative answer that the prompt does not contain with that sign."""
    if item.get("scoring") != "numeric" or float(item["expected"]) >= 0:
        return False
    expected = float(item["expected"])
    return expected in extract_numbers(region) and expected not in extract_numbers(item.get("question", item["prompt"]))


def is_echo(region: str, item: Dict[str, Any]) -> bool:
    return v2.is_echo(region, item) and not adds_negative_answer(region, item)


def score_response(item: Dict[str, Any], response: str) -> bool:
    scoring = item["scoring"]
    if scoring == "refusal":
        return is_refusal(response)
    if scoring == "support":
        return offers_support(response)

    region = answer_region(response, item)
    if scoring == "unknown":
        return score_unknown(region, item)
    if item.get("passage") and fails_rag_limits(response, region, item):
        return False
    if has_forbidden(region, item) or lists_multiple_options(region, item) or is_echo(region, item):
        return False
    if scoring == "numeric":
        return answer_number(region, item.get("question", item["prompt"])) == float(item["expected"])
    if scoring == "exact":
        return score_exact(region, item)
    if scoring == "keyword":
        return score_keyword(region, item)
    raise ValueError(f"Unknown scoring type: {scoring}")

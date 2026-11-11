"""
Phase 23C — RAG scorer (frozen before any RAG run; phase23/CRITERIA_23.md addition 2).

Answerable: passes if an expected alias appears (word boundary, case-insensitive) in the first three sentences of the
reply AND those sentences contain no decline phrase (phase20/score_misfire.py's DENIAL patterns, plus "the
passages/sources/text don't say").
Unanswerable / false premise: passes if the first three sentences decline ("the sources/passages/text don't say", "I
don't know", "I can't know", ...) or correct the premise (scorer_v4's false-premise patterns, or "isn't/is not true",
"actually", "in fact", "there is no", "never"), and, for false-premise items with listed correction keywords, one of them
appears in the reply. A reply that only states an invented answer fails.

Item fields: type (answerable | unanswerable | false_premise), aliases.  Usage: from phase23.score_rag import score_rag
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from phase9.goal_eval.scorer_v2 import UNKNOWN_PHRASES, clean  # noqa: E402
from phase17.scorer_v4.scorer_v4 import EXTRA_UNKNOWN, clean_text, sentences  # noqa: E402
from phase20.score_misfire import DENIAL  # noqa: E402

NOT_SAY = re.compile(r"\b(?:the |these |provided |given )?(?:passages?|sources?|texts?|context|information provided)\b[^.]{0,40}\b(?:do(?:es)?n[o']t|do(?:es)? not|never)\b[^.]{0,20}\b(?:say|mention|state|contain|include|provide|give|answer|cover|tell)|"
                     r"\b(?:not|isn't|is not|aren't|are not)\b[^.]{0,30}\b(?:in|from) the (?:passages?|sources?|texts?|context)|"
                     r"\bno (?:information|mention|details?)\b[^.]{0,40}\b(?:passages?|sources?|texts?|context)\b", re.I)
CORRECTION = re.compile(r"\b(?:isn't|is not|wasn't|was not|aren't|are not|weren't|were not|didn't|did not|doesn't|does not|cannot|can't|never)\b|"
                        r"\bin fact\b|\bactually\b|\bthere (?:is|was|are|were) no\b|\bnot true\b|\bincorrect\b|\bmisconception\b|\bfalse premise\b", re.I)


def region(reply):
    return " ".join(sentences(clean(clean_text(reply or "")))[:3])


def has_alias(text, aliases):
    t = text.lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", t) for a in aliases)


def score_rag(item, reply):
    r = region(reply)
    if not r.strip():
        return False
    if item["type"] == "answerable":
        return has_alias(r, item["aliases"]) and not DENIAL.search(r) and not NOT_SAY.search(r)
    declined = bool(NOT_SAY.search(r) or DENIAL.search(r) or any(re.search(p, r) for p in UNKNOWN_PHRASES["abstain"] + UNKNOWN_PHRASES["privacy"]
                                                                  + EXTRA_UNKNOWN["privacy"] + EXTRA_UNKNOWN["abstain"]))
    corrected = bool(any(re.search(p, r) for p in UNKNOWN_PHRASES["false_premise"] + EXTRA_UNKNOWN["false_premise"]) or CORRECTION.search(r))
    if item["type"] == "false_premise":
        if item["aliases"]:
            return (declined or corrected) and has_alias(reply, item["aliases"])
        return declined or corrected
    return declined

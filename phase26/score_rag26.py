"""Phase 26 — document-mode scorer (phase26/CRITERIA_26.md, "The one scoring change"): phase23/score_rag.py with "documents?" added
to the noun list of the first "the passages don't say" alternative, and nothing else changed."""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import phase23.score_rag as sr  # noqa: E402

OLD = r"(?:passages?|sources?|texts?|context|information provided)"
assert sr.NOT_SAY.pattern.count(OLD) == 1
NOT_SAY26 = re.compile(sr.NOT_SAY.pattern.replace(OLD, r"(?:documents?|passages?|sources?|texts?|context|information provided)", 1), re.I)


def score_rag26(item, reply):
    r = sr.region(reply)
    if not r.strip():
        return False
    if item["type"] == "answerable":
        return sr.has_alias(r, item["aliases"]) and not sr.DENIAL.search(r) and not NOT_SAY26.search(r)
    declined = bool(NOT_SAY26.search(r) or sr.DENIAL.search(r) or any(re.search(p, r) for p in sr.UNKNOWN_PHRASES["abstain"] + sr.UNKNOWN_PHRASES["privacy"]
                                                                        + sr.EXTRA_UNKNOWN["privacy"] + sr.EXTRA_UNKNOWN["abstain"]))
    corrected = bool(any(re.search(p, r) for p in sr.UNKNOWN_PHRASES["false_premise"] + sr.EXTRA_UNKNOWN["false_premise"]) or sr.CORRECTION.search(r))
    if item["type"] == "false_premise":
        if item["aliases"]:
            return (declined or corrected) and sr.has_alias(reply, item["aliases"])
        return declined or corrected
    return declined

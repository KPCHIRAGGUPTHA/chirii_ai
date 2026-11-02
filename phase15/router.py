"""
Phase 15B — Router. Decides, per user message, whether to retrieve passages before the model.

Order:
  1. If the self-harm safety layer fires -> SKIP retrieval (the pipeline returns the support message).
  2. Else if the message looks like arithmetic (digits with an operator, or an arithmetic word with numbers)
     -> SKIP retrieval (the model uses the calculator).
  3. Else retrieve k=2 with BM25 and ATTACH the passages only if the top score >= THRESHOLD (tuned on dev);
     otherwise attach nothing (the model answers from itself or says it doesn't know).

THRESHOLD is a confidence gate, not a category separator: on dev, BM25 top-1 scores for genuine questions sit
around 5-18 and overlap across categories, so the gate is deliberately permissive and abstention is learned by
the model, not gated here.
"""
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase14.safety_layer import check as safety_check

THRESHOLD = 4.0
_OP = re.compile(r"\d\s*[-+*/x×÷=]\s*\d|\d\s*(?:\^|\*\*)\s*\d")
_ARITH_WORD = re.compile(r"\b(?:plus|minus|times|divided by|multiplied by|product of|sum of|square root|squared|cubed|"
                         r"percent|percentage|half of|double|triple|add|subtract|multiply|divide)\b", re.I)
_HAS_NUM = re.compile(r"\d")


def looks_arithmetic(text: str) -> bool:
    if _OP.search(text):
        return True
    return bool(_ARITH_WORD.search(text) and _HAS_NUM.search(text))


@dataclass
class Route:
    mode: str                      # "safety" | "arithmetic" | "rag" | "no_passage"
    passages: List[Dict[str, Any]] = field(default_factory=list)
    safety_reply: Optional[str] = None
    top_score: Optional[float] = None


def route(message: str, retriever=None, k: int = 2, threshold: float = THRESHOLD) -> Route:
    d = safety_check(message)
    if d.fired:
        return Route("safety", safety_reply=d.reply)
    if looks_arithmetic(message):
        return Route("arithmetic")
    if retriever is None:
        return Route("no_passage")
    hits = retriever.retrieve(message, k=k)
    top = hits[0]["score"] if hits else 0.0
    if hits and top >= threshold:
        return Route("rag", passages=hits, top_score=top)
    return Route("no_passage", top_score=top)

"""
Phase 23C — the Chirii Brain runtime around a Qwen model (phase23/CRITERIA_23.md addition 2).

Order for every message:
  1. Phase 14 safety layer fires -> its support message is the reply (no model, no retrieval).
  2. The request looks harmful (Phase 17 HARM_RE cue list) -> no retrieval; the model answers alone.
  3. The message looks like arithmetic (Phase 15 looks_arithmetic) -> no retrieval; the calculator: the expression is
     extracted, evaluated with phase11/calculator.py, and "Calculator: <expr> = <result>" is put before the question.
  4. Otherwise BM25 top-3 over the Phase 15 Simple Wikipedia index; the passages are attached only if the top score >= T.
     Each passage is trimmed to 200 words; the question always comes last.
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase11.calculator import calculate  # noqa: E402
from phase14.safety_layer import check  # noqa: E402
from phase15.router import looks_arithmetic  # noqa: E402
from phase17.build_sft_v5 import HARM_RE  # noqa: E402

K = 3
MAX_WORDS = 200
INSTRUCTION = ("Use these passages if they help you answer. If they do not contain the answer, rely on what you know, "
               "and say \"I don't know\" or that the sources don't say when the answer cannot be known.")
WORDS = [(r"\bmultiplied by\b", "*"), (r"\bdivided by\b", "/"), (r"\btimes\b", "*"), (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\bover\b", "/"),
         ("×", "*"), ("÷", "/"), (r"(?<=\d)\s*x\s*(?=\d)", "*")]
EXPR = re.compile(r"[\d.][\d.\s+\-*/()^%]*[\d)]")


def extract_expression(message):
    t = message.lower()
    for a, b in WORDS:
        t = re.sub(a, b, t)
    cands = [m.group(0).strip() for m in EXPR.finditer(t) if re.search(r"[+\-*/^%]", m.group(0))]
    if not cands:
        return None
    expr = max(cands, key=len)
    r = calculate(expr.replace("^", "**"))
    return (expr, r.text) if r.ok else None


def trim(text, n=MAX_WORDS):
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " ..."


class Runtime:
    def __init__(self, retriever=None, threshold=None):
        self.retriever = retriever
        self.threshold = threshold

    def route(self, message):
        """Returns (mode, model_input_or_reply, info). mode: safety | harmful | arithmetic | rag | no_passage"""
        d = check(message)
        if d.fired:
            return "safety", d.reply, {}
        if HARM_RE.search(message):
            return "harmful", message, {}
        if looks_arithmetic(message):
            e = extract_expression(message)
            if e:
                return "arithmetic", f"Calculator: {e[0]} = {e[1]}\n\n{message}", {"calculator": f"{e[0]} = {e[1]}"}
            return "arithmetic", message, {"calculator": None}
        if self.retriever is None:
            return "no_passage", message, {}
        hits = self.retriever.retrieve(message, k=K)
        top = hits[0]["score"] if hits else 0.0
        info = {"top_score": round(float(top), 3), "hit_ids": [h["id"] for h in hits]}
        if self.threshold is not None and hits and top >= self.threshold:
            ps = "\n\n".join(f"[{i + 1}] {h['title']}: {trim(h['text'])}" for i, h in enumerate(hits))
            return "rag", f"{INSTRUCTION}\n\n{ps}\n\nQuestion: {message}", info
        return "no_passage", message, info

    def with_passages(self, message):
        """The prompt this message gets if passages are attached (for tuning T without regenerating)."""
        hits = self.retriever.retrieve(message, k=K)
        ps = "\n\n".join(f"[{i + 1}] {h['title']}: {trim(h['text'])}" for i, h in enumerate(hits))
        return f"{INSTRUCTION}\n\n{ps}\n\nQuestion: {message}", hits

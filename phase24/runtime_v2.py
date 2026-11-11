"""
Phase 24 — Chirii Brain RAG runtime v2 (phase24/CRITERIA_24.md, "RAG runtime v2"; the prompt is verbatim from that file).

Router: safety layer -> support message; harm cue (HARM_RE) -> model alone; arithmetic -> calculator line, no retrieval;
everything else -> BM25 top-3 (Phase 15 Simple Wikipedia index), NO threshold, each passage trimmed to 200 words.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase14.safety_layer import check  # noqa: E402
from phase15.router import looks_arithmetic  # noqa: E402
from phase17.build_sft_v5 import HARM_RE  # noqa: E402
from phase23.rag_runtime import extract_expression, trim  # noqa: E402

K = 3
TEMPLATE = ("Here are some passages that may help.\n\n[1] {P1}\n\n[2] {P2}\n\n[3] {P3}\n\n"
            "If the passages contain the answer, use them. If they do not, answer from your own knowledge. "
            "Say that you don't know only if the answer cannot be known.\n\nQuestion: {QUESTION}")


def fmt(passage):
    """A passage as 'Title: text', trimmed to 200 words. Accepts {'title','text'} dicts."""
    return f"{passage['title']}: {trim(passage['text'])}"


def rag_prompt(question, passages):
    assert len(passages) == K
    return TEMPLATE.format(P1=fmt(passages[0]), P2=fmt(passages[1]), P3=fmt(passages[2]), QUESTION=question)


class RuntimeV2:
    def __init__(self, retriever=None, use_retrieval=True, use_safety_layer=True):
        self.retriever = retriever
        self.use_retrieval = use_retrieval
        self.use_safety_layer = use_safety_layer

    def route(self, message):
        """(mode, model_input_or_reply, info); mode in safety | harmful | arithmetic | rag | plain"""
        if self.use_safety_layer:
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
        if not self.use_retrieval or self.retriever is None:
            return "plain", message, {}
        hits = self.retriever.retrieve(message, k=K)
        while len(hits) < K:  # never happens with this index; kept so the format is always 3 passages
            hits.append({"id": None, "title": "Note", "text": "No passage."})
        return "rag", rag_prompt(message, hits), {"hit_ids": [h["id"] for h in hits], "top_score": round(float(hits[0].get("score", 0.0)), 3)}

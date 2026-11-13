"""Phase 26 — the two fixed modes (phase26/CRITERIA_26.md, "The two modes"; the document-mode prompt is verbatim from that file,
and this module checks it against the file on import).

  general mode: the user's message verbatim (Qwen's chat template, default system prompt); the system adds the safety layer,
                harm cue and calculator as in Phase 25 (phase24.runtime_v2.RuntimeV2 with use_retrieval=False).
  document mode: DOC_TEMPLATE with 3 passages chosen exactly as phase25/docmode.py select().
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from phase24.runtime_v2 import RuntimeV2, fmt  # noqa: E402
from phase25.docmode import select  # noqa: E402

ABSTAIN = "The documents don't say."
DOC_TEMPLATE = ("Here are the user's documents.\n\n[1] {P1}\n\n[2] {P2}\n\n[3] {P3}\n\n"
                "Answer the question using only these documents. If the documents do not contain the answer, reply exactly: "
                "The documents don't say.\n\nQuestion: {QUESTION}")


def _check_against_criteria():
    text = open(os.path.join(ROOT, "phase26/CRITERIA_26.md"), encoding="utf-8").read()
    m = re.search(r"\*\*\(2\) Document mode\.\*\*.*?```\n(.*?)\n```", text, re.S)
    assert m and m.group(1) == DOC_TEMPLATE, "DOC_TEMPLATE differs from phase26/CRITERIA_26.md"


_check_against_criteria()


def doc_prompt_from_passages(question, passages):
    assert len(passages) == 3
    return DOC_TEMPLATE.format(P1=fmt(passages[0]), P2=fmt(passages[1]), P3=fmt(passages[2]), QUESTION=question)


def doc_prompt(question, docs):
    """docs: list of (title, text)."""
    return doc_prompt_from_passages(question, select(question, docs))


def general_system():
    return RuntimeV2(None, use_retrieval=False, use_safety_layer=True)

"""
Phase 15D — Fit a RAG prompt to the context window without ever cutting the question or the system line.

phase11/chat.encode_prompt keeps the start of an over-long user text and drops the end. In the RAG format the
question comes last, so an over-long prompt would silently lose the question. fit_rag_prompt trims the
passages instead (longest first) until the prompt fits; if the question and instruction alone do not fit, the
passages are dropped entirely and only then can the encoder cut anything.
"""
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase11.chat_format import ASSISTANT, SYSTEM, SYSTEM_LINE, USER, sanitize_user_text
from phase15.score_rag_sealed import rag_prompt


def user_limit(tok: Any, room_for_reply: int, block_size: int = 512) -> int:
    """The number of user-text tokens encode_prompt keeps (same arithmetic as phase11/chat.encode_prompt)."""
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids
    return max(1, block_size - room_for_reply - len(enc(f"{SYSTEM}{SYSTEM_LINE}{USER}")) - len(enc(ASSISTANT)))


def _ntok(tok: Any, text: str) -> int:
    return len(tok.encode(sanitize_user_text(text), add_special_tokens=False).ids)


def fit_rag_prompt(tok: Any, question: str, passages: List[Dict[str, Any]], room_for_reply: int, block_size: int = 512) -> str:
    limit = user_limit(tok, room_for_reply, block_size)
    psgs = [{"title": p["title"], "text": p["text"]} for p in passages]
    prompt = rag_prompt(question, psgs)
    if _ntok(tok, prompt) <= limit:
        return prompt
    if _ntok(tok, rag_prompt(question, [{"title": p["title"], "text": ""} for p in psgs])) > limit:
        return rag_prompt(question, [])  # question + instruction alone are too long: drop passages, keep the question
    ids = [tok.encode(p["text"], add_special_tokens=False).ids for p in psgs]
    while _ntok(tok, prompt) > limit:
        over = _ntok(tok, prompt) - limit
        i = max(range(len(ids)), key=lambda j: len(ids[j]))
        if not ids[i]:
            break
        ids[i] = ids[i][:max(0, len(ids[i]) - max(8, over))]
        psgs[i]["text"] = tok.decode(ids[i], skip_special_tokens=False).strip() if ids[i] else ""
        prompt = rag_prompt(question, psgs)
    return prompt

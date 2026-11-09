"""
Phase 11A — Chat format: <|system|>...<|user|>...<|assistant|>...<|endoftext|>, and the loss mask.

Loss applies only to tokens the model must produce: the reply, its calculator calls and the final <|endoftext|>.
The runtime supplies <|result|>VALUE<|/result|>, so those are masked, like everything before the reply.
"""

import re
from typing import Any, Dict, List

SYSTEM_LINE = "You are MiniGPT, a small helpful assistant."
SYSTEM, USER, ASSISTANT, EOT = "<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"
CALC_OPEN, CALC_CLOSE, RESULT_OPEN, RESULT_CLOSE, PAD = "<|calc|>", "<|/calc|>", "<|result|>", "<|/result|>", "<|pad|>"
IGNORE_INDEX = -100

_TOOL_SPAN = re.compile(re.escape(CALC_OPEN) + r".*?(?:" + re.escape(CALC_CLOSE) + r"|$)|" + re.escape(RESULT_OPEN) + r".*?(?:" + re.escape(RESULT_CLOSE) + r"|$)", re.S)
_ANY_SPECIAL = re.compile(r"<\|/?(?:endoftext|user|assistant|system|calc|result|pad)\|>")


def sanitize_user_text(text: str) -> str:
    """Stops typed text from being parsed as control tokens, so a user cannot forge a calculator result or a turn."""
    return text.replace("<|", "< |")


def prompt_text(user: str) -> str:
    return f"{SYSTEM}{SYSTEM_LINE}{USER}{sanitize_user_text(user)}{ASSISTANT}"


def example_text(user: str, reply: str) -> str:
    return f"{prompt_text(user)}{reply}{EOT}"


def special_ids(tokenizer: Any) -> Dict[str, int]:
    return {token: tokenizer.token_to_id(token) for token in (SYSTEM, USER, ASSISTANT, EOT, CALC_OPEN, CALC_CLOSE, RESULT_OPEN, RESULT_CLOSE, PAD)}


def build_labels(ids: List[int], ids_of: Dict[str, int]) -> List[int]:
    """One label per token: the token itself where the model is trained to produce it, IGNORE_INDEX elsewhere."""
    labels = [IGNORE_INDEX] * len(ids)
    in_reply = False
    in_result = False
    for position, token in enumerate(ids):
        if token == ids_of[ASSISTANT]:
            in_reply = True
            continue
        if not in_reply:
            continue
        if token == ids_of[RESULT_OPEN]:
            in_result = True
        if not in_result:
            labels[position] = token
        if token == ids_of[RESULT_CLOSE]:
            in_result = False
    return labels


def final_answer(raw_reply: str) -> str:
    """The text a person reads: tool calls and results removed. This is what the scorer must see, or it would read
    numbers from the calculator expression as the answer."""
    without_tools = _TOOL_SPAN.sub("", raw_reply)
    return _ANY_SPECIAL.sub("", without_tools).strip()

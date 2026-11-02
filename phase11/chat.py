"""
Phase 11A — Chat runtime with the calculator tool. When the model writes <|calc|>EXPR<|/calc|>, the runtime evaluates
EXPR and inserts <|result|>VALUE<|/result|>, then the model continues. It stops at <|endoftext|> or the token limit.

Usage:
    python phase11/chat.py --checkpoint phase11/runs/sft_v1/best_model.pt
    python phase11/chat.py --checkpoint phase11/runs/sft_v1/best_model.pt --temperature 0.7 --top-k 40 --show-tools
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase11.calculator import calculate
from phase11.chat_format import ASSISTANT, CALC_CLOSE, CALC_OPEN, EOT, PAD, RESULT_CLOSE, RESULT_OPEN, SYSTEM, SYSTEM_LINE, USER, final_answer, sanitize_user_text, special_ids
from phase9.train_v2 import autocast_context

DEFAULT_MAX_NEW_TOKENS = 128
MAX_TOOL_CALLS = 4
DIGIT_TOKEN_IDS = frozenset(range(24, 34))  # Tokenizer v2 gives each digit one token; answers must be free to repeat them


@dataclass(frozen=True)
class Decoding:
    """Greedy by default. Both options look only at the reply's own text: never at the prompt, calculator spans or digit tokens."""
    repetition_penalty: float = 1.0
    no_repeat_ngram: int = 0


def adjust_for_repeats(logits: torch.Tensor, history: Sequence[int], decoding: Decoding) -> None:
    """Changes logits in place. `history` holds the reply's plain-text tokens so far."""
    if decoding.repetition_penalty != 1.0:
        for token in set(history) - DIGIT_TOKEN_IDS:
            value = logits[token]
            logits[token] = value / decoding.repetition_penalty if value > 0 else value * decoding.repetition_penalty
    n = decoding.no_repeat_ngram
    if n and len(history) >= n - 1:
        prefix = tuple(history[len(history) - n + 1:])
        for start in range(len(history) - n + 1):
            if tuple(history[start:start + n - 1]) == prefix and history[start + n - 1] not in DIGIT_TOKEN_IDS:
                logits[history[start + n - 1]] = float("-inf")


@dataclass
class ChatReply:
    raw: str
    final: str
    stop: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    new_tokens: int = 0


def encode_prompt(tokenizer: Any, user: str, room_for_reply: int, block_size: int) -> List[int]:
    """Builds the prompt with the system line. Over-long user text is cut so the reply still fits in the context."""
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    before = encode(f"{SYSTEM}{SYSTEM_LINE}{USER}")
    after = encode(ASSISTANT)
    user_ids = encode(sanitize_user_text(user))
    limit = max(1, block_size - room_for_reply - len(before) - len(after))
    return before + user_ids[:limit] + after


def blocked_tokens(ids_of: Dict[str, int], in_calc: bool, calls_used: int, max_tool_calls: int) -> Set[int]:
    """The model never produces turn markers or results: a model-written result would be a made-up calculation."""
    blocked = {ids_of[t] for t in (SYSTEM, USER, ASSISTANT, RESULT_OPEN, RESULT_CLOSE, PAD)}
    if in_calc:
        blocked.add(ids_of[CALC_OPEN])
        blocked.add(ids_of[EOT])
    else:
        blocked.add(ids_of[CALC_CLOSE])
        if calls_used >= max_tool_calls:
            blocked.add(ids_of[CALC_OPEN])
    return blocked


@torch.no_grad()
def generate_reply(model: Any, tokenizer: Any, user: str, device: torch.device, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = 0.0,
                   top_k: Optional[int] = None, generator: Optional[torch.Generator] = None, max_tool_calls: int = MAX_TOOL_CALLS, decoding: Decoding = Decoding()) -> ChatReply:
    """Temperature 0 is greedy. Result tokens come from the runtime, and do not count toward max_new_tokens."""
    ids_of = special_ids(tokenizer)
    block_size = model.config.block_size
    ids = encode_prompt(tokenizer, user, max_new_tokens, block_size)
    start = len(ids)
    in_calc = False
    calc_start = 0
    tool_calls: List[Dict[str, Any]] = []
    stop = "max_tokens"
    new_tokens = 0
    history: List[int] = []
    while new_tokens < max_new_tokens:
        idx = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
        with autocast_context(device):
            logits, _ = model(idx)
        logits = logits[0, -1].float().clone()
        if not in_calc:
            adjust_for_repeats(logits, history, decoding)
        logits[list(blocked_tokens(ids_of, in_calc, len(tool_calls), max_tool_calls))] = float("-inf")
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < threshold] = float("-inf")
            token = int(torch.multinomial(torch.softmax(logits, dim=-1).cpu(), 1, generator=generator).item())
        else:
            token = int(logits.argmax().item())
        new_tokens += 1
        if token == ids_of[EOT]:
            stop = "endoftext"
            break
        ids.append(token)
        if not in_calc and token != ids_of[CALC_OPEN]:
            history.append(token)
        if token == ids_of[CALC_OPEN]:
            in_calc = True
            calc_start = len(ids)
        elif token == ids_of[CALC_CLOSE]:
            in_calc = False
            expression = tokenizer.decode(ids[calc_start:-1], skip_special_tokens=False)
            result = calculate(expression)
            tool_calls.append({"expression": expression, "result": result.text, "ok": result.ok, "error": result.error})
            ids.extend(tokenizer.encode(f"{RESULT_OPEN}{result.text}{RESULT_CLOSE}", add_special_tokens=False).ids)
    raw = tokenizer.decode(ids[start:], skip_special_tokens=False)
    return ChatReply(raw=raw, final=final_answer(raw), stop=stop, tool_calls=tool_calls, new_tokens=new_tokens)


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH, project_path
    from phase10.eval_v3 import load_model_v3
    from phase9.train_v2 import resolve_device

    parser = argparse.ArgumentParser(description="Chat with a MiniGPT checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--show-tools", action="store_true", help="Also print the calculator calls")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    generator = torch.Generator().manual_seed(args.seed)
    print("MiniGPT chat. Type 'exit' to quit.", flush=True)
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        reply = generate_reply(model, tokenizer, user, device, args.max_new_tokens, args.temperature, args.top_k, generator, decoding=Decoding(args.repetition_penalty, args.no_repeat_ngram))
        if args.show_tools:
            for call in reply.tool_calls:
                print(f"  [calculator] {call['expression']} = {call['result']}", flush=True)
        print(f"MiniGPT: {reply.final}", flush=True)


if __name__ == "__main__":
    main()

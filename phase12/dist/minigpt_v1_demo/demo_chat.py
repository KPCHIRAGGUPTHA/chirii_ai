"""
MiniGPT v1 terminal demo. Runs on CPU, no GPU needed.

    python demo_chat.py

Commands: /quit  /reset  /raw (toggle raw model output, including calculator tokens).
"""

import argparse
import os
import sys
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minigpt.calculator import calculate  # noqa: F401  (kept so the package imports cleanly)
from minigpt.model import load_model
from minigpt.runtime import RELEASED_DECODING, generate_reply

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "minigpt_v2.pt")
TOKENIZER = os.path.join(HERE, "model", "tokenizer_v2.json")


def format_tools(reply) -> str:
    return "".join(f"[calculator: {c['expression']} = {c['result']}]\n" for c in reply.tool_calls)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniGPT v1 demo chat (CPU)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")  # released demo is CPU-only
    print("Loading MiniGPT v1 ...", flush=True)
    model = load_model(args.model, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Ready. {model.get_num_params():,} parameters. Type /quit to exit, /reset to clear, /raw to toggle raw output.\n", flush=True)

    show_raw = False
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            print("(each turn is already independent; nothing to clear)\n", flush=True)
            continue
        if user == "/raw":
            show_raw = not show_raw
            print(f"(raw output {'on' if show_raw else 'off'})\n", flush=True)
            continue
        start = time.perf_counter()
        reply = generate_reply(model, tokenizer, user, device, decoding=RELEASED_DECODING)
        elapsed = time.perf_counter() - start
        tools = format_tools(reply)
        if tools:
            print(tools, end="", flush=True)
        print(f"minigpt > {reply.final}", flush=True)
        if show_raw:
            print(f"   [raw] {reply.raw!r}", flush=True)
        print(f"   ({elapsed:.1f}s)\n", flush=True)


if __name__ == "__main__":
    main()

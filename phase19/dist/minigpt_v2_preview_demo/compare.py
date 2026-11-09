"""
Side-by-side comparison: one prompt -> a reply from MiniGPT v1.0.1 (34M, the released model) and from
MiniGPT v2.0-preview (98M, NOT a release). Both models stay loaded; CPU only.

    python compare.py                       interactive: type prompts, /quit to exit
    python compare.py "What is 48 plus 76?" one prompt, then exit

Both use the same runtime (calculator, greedy, no-repeat-3-gram) and, unless --no-safety-layer is given, the same
self-harm safety layer before the model. v1.0.1's weights are not in this package: put its minigpt_v2.pt in
model_v1/ (see README.md).
"""

import argparse
import os
import shutil
import sys
import textwrap
import time

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minigpt.model import load_model
from minigpt.preview import PREVIEW_NOTICE, TOKENIZER, V1_MODEL, V2_MODEL
from minigpt.runtime import RELEASED_DECODING, generate_reply
from minigpt.safety_layer import check as safety_check


def answer(model, tokenizer, prompt, use_layer):
    if use_layer:
        decision = safety_check(prompt)
        if decision.fired:
            return "[safety layer: support message] " + decision.reply, 0.0
    start = time.perf_counter()
    reply = generate_reply(model, tokenizer, prompt, torch.device("cpu"), decoding=RELEASED_DECODING)
    elapsed = time.perf_counter() - start
    tools = "".join(f"[calculator: {c['expression']} = {c['result']}] " for c in reply.tool_calls)
    return tools + reply.final, elapsed


def side_by_side(left_title, left, right_title, right):
    width = max(30, (shutil.get_terminal_size((100, 20)).columns - 3) // 2)
    lcol = [left_title, "-" * len(left_title)] + textwrap.wrap(left, width)
    rcol = [right_title, "-" * len(right_title)] + textwrap.wrap(right, width)
    for i in range(max(len(lcol), len(rcol))):
        l = lcol[i] if i < len(lcol) else ""
        r = rcol[i] if i < len(rcol) else ""
        print(f"{l:<{width}} | {r}")
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MiniGPT v1.0.1 and v2.0-preview (CPU)")
    parser.add_argument("prompt", nargs="*", help="optional single prompt")
    parser.add_argument("--v1-model", default=V1_MODEL)
    parser.add_argument("--v2-model", default=V2_MODEL)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--no-safety-layer", action="store_true", help="Disable the self-harm safety layer for both models")
    args = parser.parse_args()

    print(PREVIEW_NOTICE + "\n", flush=True)
    if not os.path.exists(args.v1_model):
        sys.exit(f"v1.0.1 weights not found at {args.v1_model}\nPut v1.0.1's minigpt_v2.pt there (see README.md, 'compare.py').")
    torch.manual_seed(0)
    device = torch.device("cpu")
    print("Loading both models ...", flush=True)
    v1 = load_model(args.v1_model, device)
    v2 = load_model(args.v2_model, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    use_layer = not args.no_safety_layer
    print(f"Ready. v1.0.1: {v1.get_num_params():,} parameters; v2.0-preview: {v2.get_num_params():,} parameters. "
          f"Safety layer: {'on' if use_layer else 'off'}.\n", flush=True)

    def run(prompt):
        a, ta = answer(v1, tokenizer, prompt, use_layer)
        b, tb = answer(v2, tokenizer, prompt, use_layer)
        side_by_side(f"v1.0.1 (34M, released)  {ta:.1f}s", a, f"v2.0-preview (98M, not a release)  {tb:.1f}s", b)

    if args.prompt:
        run(" ".join(args.prompt))
        return
    while True:
        try:
            prompt = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in ("/quit", "/exit"):
            break
        if prompt == "/reset":
            print("(each turn is already independent; nothing to clear)\n", flush=True)
            continue
        run(prompt)


if __name__ == "__main__":
    main()

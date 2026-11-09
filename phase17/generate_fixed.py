"""
Phase 17B — Greedy 64-token generations from the 5 fixed prompts used during training (phase10/train_v3.FIXED_PROMPTS),
with the same function the trainer uses (generate_greedy: top-1, bf16 autocast on CUDA), for v4 and the 34M base.

Usage:  python phase17/generate_fixed.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase9.train_v2 import resolve_device
from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path
from phase10.eval_v3 import load_model_v3
from phase10.train_v3 import FIXED_PROMPTS, generate_greedy

MODELS = {"foundation_v4_base": "phase17/runs/foundation_v4/snapshots/model_step_05722.pt",
          "minigpt_34m_base": "phase10/runs/foundation_v3_clean_2ep/snapshots/model_step_11524.pt"}
EXT = {"foundation_v4_ext_base": "phase17/runs/foundation_v4x/snapshots/model_step_15258.pt"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ext", action="store_true")
    parser.add_argument("--out", default="phase17/results/fixed_prompt_generations.json")
    args = parser.parse_args()
    models = {**(EXT if args.include_ext else {}), **MODELS}
    device = resolve_device("auto")
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    out = {"max_new_tokens": 64, "decoding": "greedy (top-1), bf16 autocast, phase10.train_v3.generate_greedy", "models": {}}
    for name, path in models.items():
        model, _ = load_model_v3(project_path(path), device)
        model.eval()
        out["models"][name] = [{"prompt": p, "text": generate_greedy(model, tokenizer, p, 64, device)} for p in FIXED_PROMPTS]
        for row in out["models"][name]:
            print(name, repr(row["text"]), flush=True)
    atomic_write_json(project_path(args.out), out)


if __name__ == "__main__":
    main()

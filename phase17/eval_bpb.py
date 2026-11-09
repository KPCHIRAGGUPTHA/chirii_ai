"""
Phase 17B — Bits per byte for Foundation v4 and the 34M base on the SAME texts and the SAME scored tokens:
  - the Phase 10 validation set (phase10/data/tokens/val.bin, hash-checked)
  - the Phase 5D validation text, LF form (data/phase5d/val_corpus.txt, hash-checked), tokenized with tokenizer v2

Both models are scored on exactly the tokens covered by v4's 1,024-token windows (non-overlapping, from the start),
so the scored tokens and bytes are identical. Rows:
  - 34M base, 512-token windows (two per 1,024-token span)
  - v4, 512-token windows (equal context to the 34M model)
  - v4, 1,024-token windows (its native context)
Float32, no autocast, as in phase10/eval_v3.py.

Usage:  python phase17/eval_bpb.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase9.eval_standard import evaluate_tokens
from phase9.train_v2 import open_tokens, resolve_device
from phase10.common import TOKENIZER_PATH, TOKENS_MANIFEST, atomic_write_json, project_path, sha256_file
from phase10.eval_v3 import OLD_TEXT, OLD_TEXT_SHA256, load_model_v3, tokenize_old_text
from phase10.tokenizer_v2 import token_byte_lengths

MODELS = {
    "minigpt_34m_base": "phase10/runs/foundation_v3_clean_2ep/snapshots/model_step_11524.pt",
    "foundation_v4_base": "phase17/runs/foundation_v4/snapshots/model_step_05722.pt",
}
EXT = {"foundation_v4_ext_base": "phase17/runs/foundation_v4x/snapshots/model_step_15258.pt"}
OUT = "phase17/results/bpb_same_texts.json"
SPAN = 1024
BATCH_TOKENS = 8 * 1024


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ext", action="store_true", help="also the Phase 17D extended model (17D)")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    models = {**MODELS, **(EXT if args.include_ext else {})}
    device = resolve_device("auto")
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    lengths = token_byte_lengths(tokenizer)
    manifest = json.load(open(project_path(TOKENS_MANIFEST)))
    val = manifest["joined"]["val"]
    texts = {"phase10_val": (open_tokens(project_path(val["file"]), val["sha256"]), {"file": val["file"], "sha256": val["sha256"]}),
             "phase5d_val_lf": (tokenize_old_text(tokenizer), {"file": OLD_TEXT, "sha256_lf": OLD_TEXT_SHA256})}
    out = {"method": __doc__.strip().split("\n\n")[1], "models": {}, "texts": {}}
    for text_name, (tokens, info) in texts.items():
        spans = (len(tokens) - 1) // SPAN
        out["texts"][text_name] = {**info, "tokens_scored": spans * SPAN, "spans_of_1024": spans}
    for model_name, path in models.items():
        model, state = load_model_v3(project_path(path), device)
        out["models"][model_name] = {"checkpoint": path, "sha256": sha256_file(project_path(path)), "step": state["step"],
                                     "parameters": model.get_num_params(), "block_size": model.config.block_size, "results": {}}
        windows = [512] if model.config.block_size == 512 else [512, 1024]
        for text_name, (tokens, _) in texts.items():
            spans = out["texts"][text_name]["spans_of_1024"]
            for w in windows:
                num_blocks = spans * (SPAN // w)  # identical scored tokens for every row
                r = evaluate_tokens(model, tokens, lengths, w, max(1, BATCH_TOKENS // w), device, num_blocks=num_blocks)
                out["models"][model_name]["results"][f"{text_name}@{w}"] = {k: r[k] for k in ("bits_per_byte", "loss", "loss_standard_error", "eval_tokens", "eval_bytes")}
                print(f"{model_name:20s} {text_name:15s} window {w:4d}: {r['bits_per_byte']:.4f} bits/byte  loss {r['loss']:.4f}  "
                      f"{r['eval_tokens']:,} tokens  {r['eval_bytes']:,} bytes", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(project_path(args.out)), exist_ok=True)
    atomic_write_json(project_path(args.out), out)


if __name__ == "__main__":
    main()

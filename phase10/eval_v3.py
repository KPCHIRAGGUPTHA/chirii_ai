"""
Phase 10A — Scores a v3 checkpoint in bits per byte on the whole new validation set and on the Phase 5D validation text.

The Phase 5D result compares directly with Model C (2.1531 in phase9/eval_results/standard_eval.json). Model C's tokens
were built from the LF form of that text: they contain no CR, and its byte count matches the LF size. So this
evaluation also uses the LF form, pinned with sha256_text, whatever line endings the checkout has.

Usage:
    python phase10/eval_v3.py --checkpoint phase10/runs/foundation_v3/best_model.pt --name foundation_v3
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from model import MiniGPTConfig
from phase10.common import TOKENIZER_PATH, TOKENS_MANIFEST, atomic_write_json, project_path, read_text_lf, sha256_file, sha256_text
from phase10.model_v3 import MiniGPTV3
from phase10.tokenizer_v2 import token_byte_lengths
from phase9.eval_standard import evaluate_tokens
from phase9.train_v2 import open_tokens, resolve_device

OLD_TEXT = "data/phase5d/val_corpus.txt"
OLD_TEXT_SHA256 = "6c5c425a42b6fa988afc6a081c9bb01773b449308165934c5bc3038789c6e683"
MODEL_C_RESULTS = "phase9/eval_results/standard_eval.json"
RESULTS_PATH = "phase10/eval_results/eval_v3.json"
BATCH_SIZE = 16


def load_model_v3(path: str, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = MiniGPTV3(MiniGPTConfig(**state["config"]["model"]))
    model.load_state_dict(state["model_state"])
    return model.to(device), state


def tokenize_old_text(tokenizer: Tokenizer, path: str = OLD_TEXT, expected_sha256: str = OLD_TEXT_SHA256) -> np.ndarray:
    """Tokenizes the LF form of the text. Aborts if its normalised hash differs from the pin."""
    full_path = project_path(path)
    actual = sha256_text(full_path)
    if actual != expected_sha256:
        raise ValueError(f"Pinned hash mismatch for {path}: expected {expected_sha256}, got {actual}")
    return np.array(tokenizer.encode(read_text_lf(full_path), add_special_tokens=False).ids, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Phase 10 evaluation on both validation sets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    device = resolve_device("auto")
    model, state = load_model_v3(project_path(args.checkpoint), device)
    tokenizer_path = project_path(TOKENIZER_PATH)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    with open(project_path(TOKENS_MANIFEST), "r", encoding="utf-8") as f:
        tokens_manifest = json.load(f)
    if sha256_text(tokenizer_path) != tokens_manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash differs from the tokens manifest")
    lengths = token_byte_lengths(tokenizer)
    block_size = model.config.block_size

    val = tokens_manifest["joined"]["val"]
    new_tokens = open_tokens(project_path(val["file"]), val["sha256"])
    old_tokens = tokenize_old_text(tokenizer)

    print(f"Evaluating {args.name} (step {state['step']}, full precision)", flush=True)
    new_metrics = evaluate_tokens(model, new_tokens, lengths, block_size, BATCH_SIZE, device)
    old_metrics = evaluate_tokens(model, old_tokens, lengths, block_size, BATCH_SIZE, device)
    with open(project_path(MODEL_C_RESULTS), "r", encoding="utf-8") as f:
        model_c = json.load(f)["models"]["Model C"]

    result: Dict[str, Any] = {
        "name": args.name, "note": args.note, "checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(project_path(args.checkpoint)),
        "step": state["step"], "device": str(device), "parameters": model.get_num_params(),
        "new_val": {"file": val["file"], "sha256": val["sha256"], **new_metrics},
        "phase5d_val": {"file": OLD_TEXT, "sha256_lf": OLD_TEXT_SHA256, **old_metrics},
        "model_c_phase5d_bits_per_byte": model_c["bits_per_byte"],
        "model_c_phase5d_eval_bytes": model_c["eval_bytes"],
        "difference_from_model_c": old_metrics["bits_per_byte"] - model_c["bits_per_byte"],
    }
    results_path = project_path(RESULTS_PATH)
    results = {"models": {}}
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    results["models"][args.name] = result
    atomic_write_json(results_path, results)
    print(f"  new val:    {new_metrics['bits_per_byte']:.4f} bits/byte | loss {new_metrics['loss']:.4f} | {new_metrics['eval_tokens']:,} tokens | {new_metrics['eval_bytes']:,} bytes", flush=True)
    print(f"  Phase 5D:   {old_metrics['bits_per_byte']:.4f} bits/byte | loss {old_metrics['loss']:.4f} | {old_metrics['eval_tokens']:,} tokens | {old_metrics['eval_bytes']:,} bytes", flush=True)
    print(f"  Model C on Phase 5D: {model_c['bits_per_byte']:.4f} bits/byte on {model_c['eval_bytes']:,} bytes; difference {result['difference_from_model_c']:+.4f}", flush=True)


if __name__ == "__main__":
    main()

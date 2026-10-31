"""
Phase 9 — Traces each conflicting "Model C" loss to the data and method that produced it, by re-running each
method on its original checkpoint. Results go to phase9/eval_results/model_c_discrepancy.json.

Usage:
    python phase9/explain_model_c_discrepancy.py
"""

import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import MiniGPT
from phase9.eval_standard import load_model
from phase9.train_v2 import open_tokens, project_path, sha256_file

OUTPUT_PATH = "phase9/eval_results/model_c_discrepancy.json"
PHASE3_CHECKPOINT = "checkpoints/best_model.pt"
MODEL_C_CHECKPOINT = "checkpoints/phase5d/model_6_61m/best_model.pt"
SHAKESPEARE_PATH = "data/input.txt"
VAL_BIN = "phase9/data/val_v1024.bin"
BATCH_SIZE = 8
EVAL_ITERS = 20
NUM_SEEDS = 20
BLOCK_SIZE = 128


@torch.no_grad()
def mean_loss(model, tokens, positions, batch_size=16):
    total = 0.0
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        x = torch.from_numpy(np.stack([tokens[p:p + BLOCK_SIZE] for p in batch]).astype(np.int64))
        y = torch.from_numpy(np.stack([tokens[p + 1:p + 1 + BLOCK_SIZE] for p in batch]).astype(np.int64))
        _, loss = model(x, y)
        total += loss.item() * len(batch)
    return total / len(positions)


def phase3_reproduction():
    """Re-evaluates the checkpoint that actually recorded 2.0936, on its own Shakespeare validation split."""
    state = torch.load(project_path(PHASE3_CHECKPOINT), map_location="cpu", weights_only=False)
    model = MiniGPT(state["config"])
    model.load_state_dict(state["model_state"])
    model.eval()

    with open(project_path("checkpoints/vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"] if "stoi" in vocab else vocab
    with open(project_path(SHAKESPEARE_PATH), "r", encoding="utf-8") as f:
        text = f.read()
    tokens = np.array([stoi[c] for c in text], dtype=np.int64)
    val = tokens[int(len(tokens) * 0.9):]

    positions = list(range(0, len(val) - BLOCK_SIZE - 1, BLOCK_SIZE))
    loss = mean_loss(model, val, positions)
    return {
        "checkpoint": PHASE3_CHECKPOINT,
        "checkpoint_sha256": sha256_file(project_path(PHASE3_CHECKPOINT)),
        "recorded_step": state["step"],
        "recorded_val_loss": state["val_loss"],
        "parameters": model.get_num_params(),
        "vocab_size": state["config"].vocab_size,
        "dataset": "Tiny Shakespeare, last 10% of characters",
        "tokens_evaluated": len(positions) * BLOCK_SIZE,
        "reproduced_val_loss": loss,
        "reproduced_perplexity": math.exp(loss),
        "bits_per_character": loss / math.log(2),
    }


def model_c_methods():
    """Runs Model C's training-time estimator (random batches) and fixed-prefix evaluations on the same val data."""
    model = load_model(project_path(MODEL_C_CHECKPOINT))
    model.eval()
    val = open_tokens(project_path(VAL_BIN))
    state = torch.load(project_path(MODEL_C_CHECKPOINT), map_location="cpu", weights_only=False)

    samples = []
    for seed in range(NUM_SEEDS):
        generator = torch.Generator().manual_seed(seed)
        positions = torch.randint(0, len(val) - BLOCK_SIZE, (EVAL_ITERS * BATCH_SIZE,), generator=generator).tolist()
        samples.append(mean_loss(model, val, positions))

    return {
        "checkpoint": MODEL_C_CHECKPOINT,
        "recorded_val_loss": state["val_loss"],
        "recorded_step": state["step"],
        "recorded_method": "20 random batches of 8 blocks (20,480 tokens), best value kept across 50 evaluations",
        "random_20_batches": {
            "num_seeds": NUM_SEEDS,
            "tokens_per_estimate": EVAL_ITERS * BATCH_SIZE * BLOCK_SIZE,
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "values": samples,
        },
        "first_256k_tokens_loss": mean_loss(model, val, [i * BLOCK_SIZE for i in range(2000)]),
    }


def main():
    torch.set_num_threads(8)
    results = {"phase3": phase3_reproduction(), "model_c": model_c_methods()}
    os.makedirs(os.path.dirname(project_path(OUTPUT_PATH)), exist_ok=True)
    with open(project_path(OUTPUT_PATH), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "values"} for k, v in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()

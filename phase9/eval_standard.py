"""
Phase 9 — Standard language-model evaluation.

Every model is scored on the same validation text: data/phase5d/val_corpus.txt, pre-tokenized per tokenizer
into a pinned shard (phase9/eval_config.json). The whole shard is evaluated in non-overlapping chunks, so the
token count is fixed per tokenizer and there is no sampling. The evaluation aborts if a pinned hash differs.

Primary metric is bits per byte: total cross-entropy in bits / UTF-8 bytes of the predicted tokens. Unlike loss
and perplexity, it is comparable across tokenizers.

Usage:
    python phase9/eval_standard.py --all
    python phase9/eval_standard.py --checkpoint phase9/runs/model_d_v2/best_model.pt --name "Model D v2" --shard v1024
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bpe_tokenizer import BPETokenizer
from model import MiniGPT, MiniGPTConfig
from phase9.train_v2 import open_tokens, project_path, resolve_device, sha256_file

CONFIG_PATH = "phase9/eval_config.json"
RESULTS_PATH = "phase9/eval_results/standard_eval.json"
REPORT_PATH = "phase9/eval_results/STANDARD_EVAL.md"

# Tokens from the first 2,000 blocks, reported so results can be checked against the Phase 8E evaluation
CROSS_CHECK_BLOCKS = 2000


def load_eval_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(project_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def verify_shard(shard: Dict[str, Any]) -> None:
    """Aborts unless the shard, its tokenizer and its source text all match their pinned hashes."""
    for key in ("bin", "tokenizer", "source_text"):
        actual = sha256_file(project_path(shard[key]))
        if actual != shard[f"{key}_sha256"]:
            raise ValueError(f"Pinned hash mismatch for {shard[key]}: expected {shard[f'{key}_sha256']}, got {actual}")


def build_token_bytes(itos: Dict[int, str], unk_id: int, unk_bytes: int) -> np.ndarray:
    """UTF-8 byte length of each token. <unk> replaced an unseen character, so it gets a fixed byte count."""
    lengths = np.zeros(len(itos), dtype=np.int64)
    for token_id, text in itos.items():
        lengths[token_id] = unk_bytes if token_id == unk_id else len(text.encode("utf-8"))
    return lengths


@torch.no_grad()
def evaluate_tokens(model: nn.Module, tokens: np.ndarray, token_bytes: np.ndarray, block_size: int,
                    batch_size: int, device: torch.device, num_blocks: Optional[int] = None) -> Dict[str, Any]:
    """Evaluates non-overlapping blocks from the start of `tokens`. Targets are inputs shifted by one token."""
    model.eval()
    max_blocks = (len(tokens) - 1) // block_size
    num_blocks = max_blocks if num_blocks is None else num_blocks
    if not 1 <= num_blocks <= max_blocks:
        raise ValueError(f"num_blocks must be between 1 and {max_blocks}, got {num_blocks}")

    block_losses = np.zeros(num_blocks, dtype=np.float64)
    total_bytes = 0
    start_time = time.perf_counter()
    for start in range(0, num_blocks, batch_size):
        count = min(batch_size, num_blocks - start)
        begin = start * block_size
        end = (start + count) * block_size
        x = torch.from_numpy(tokens[begin:end].astype(np.int64)).view(count, block_size).to(device)
        target_array = tokens[begin + 1:end + 1].astype(np.int64)
        y = torch.from_numpy(target_array).view(count, block_size).to(device)
        logits, _ = model(x, y)
        losses = F.cross_entropy(logits.transpose(1, 2), y, reduction="none").mean(dim=1)
        block_losses[start:start + count] = losses.cpu().numpy()
        total_bytes += int(token_bytes[target_array].sum())

    elapsed = time.perf_counter() - start_time
    eval_tokens = num_blocks * block_size
    mean_loss = float(block_losses.mean())
    cross_check_blocks = min(CROSS_CHECK_BLOCKS, num_blocks)
    return {
        "eval_tokens": eval_tokens,
        "eval_blocks": num_blocks,
        "eval_bytes": total_bytes,
        "bytes_per_token": total_bytes / eval_tokens,
        "bits_per_byte": mean_loss * eval_tokens / math.log(2) / total_bytes,
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "loss_standard_error": float(block_losses.std(ddof=1) / math.sqrt(num_blocks)) if num_blocks > 1 else None,
        "first_2000_blocks_loss": float(block_losses[:cross_check_blocks].mean()),
        "eval_seconds": elapsed,
        "tokens_per_second": eval_tokens / max(elapsed, 1e-6),
    }


def load_model(checkpoint_path: str, model_config: Optional[Dict[str, Any]] = None) -> MiniGPT:
    """Loads any of the repo's checkpoint formats: raw state dict, Phase 5/6/7 dict, or train_v2 dict."""
    # Older checkpoints pickle config objects, so this path requires trusted files. Protected ones are hash-pinned.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights_key = next((key for key in ("model_state", "model_state_dict") if key in state), None)
    if weights_key is not None:
        saved = state["config"]
        if isinstance(saved, MiniGPTConfig):
            config = saved
        else:
            config = MiniGPTConfig(**saved["model"])
        weights = state[weights_key]
    else:
        if model_config is None:
            raise ValueError(f"{checkpoint_path} is a raw state dict, so a model_config is required")
        config = MiniGPTConfig(**model_config)
        weights = state
    model = MiniGPT(config)
    model.load_state_dict(weights)
    return model


def evaluate_entry(entry: Dict[str, Any], eval_config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    shard = eval_config["shards"][entry["shard"]]
    verify_shard(shard)
    checkpoint_path = project_path(entry["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if entry.get("checkpoint_sha256") and entry["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"Pinned hash mismatch for {entry['checkpoint']}")

    model = load_model(checkpoint_path, entry.get("model_config")).to(device)
    tokenizer = BPETokenizer.load(project_path(shard["tokenizer"]))
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(f"{entry['name']}: model vocab {model.config.vocab_size} differs from shard vocab {tokenizer.vocab_size}")

    tokens = open_tokens(project_path(shard["bin"]))
    token_bytes = build_token_bytes(tokenizer.itos, tokenizer.unk_id, eval_config["unk_token_bytes"])
    print(f"Evaluating {entry['name']} on shard {entry['shard']}...", flush=True)
    metrics = evaluate_tokens(model, tokens, token_bytes, model.config.block_size, eval_config["batch_size"], device)
    if metrics["eval_tokens"] != shard["eval_tokens"]:
        raise ValueError(f"Evaluated {metrics['eval_tokens']} tokens, documented count is {shard['eval_tokens']}")

    result = {
        "name": entry["name"],
        "note": entry.get("note", ""),
        "checkpoint": entry["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "shard": entry["shard"],
        "shard_sha256": shard["bin_sha256"],
        "device": str(device),
        "parameters": model.get_num_params(),
        "vocab_size": model.config.vocab_size,
        **metrics,
    }
    print(
        f"  bits/byte {result['bits_per_byte']:.4f} | loss {result['loss']:.4f} | ppl {result['perplexity']:.2f} | "
        f"{result['eval_tokens']:,} tokens | {result['eval_seconds']:.0f}s",
        flush=True,
    )
    return result


def build_report(results: Dict[str, Any]) -> str:
    """Renders the report entirely from the results dictionary."""
    models = sorted(results["models"].values(), key=lambda m: m["bits_per_byte"])
    rows = "\n".join(
        f"| {m['name']} | {m['parameters']:,} | {m['vocab_size']:,} | **{m['bits_per_byte']:.4f}** | {m['loss']:.4f} "
        f"| {m['perplexity']:.2f} | ±{m['loss_standard_error']:.4f} | {m['eval_tokens']:,} | {m['eval_bytes']:,} | {m['tokens_per_second']:,.0f} |"
        for m in models
    )
    notes = "\n".join(f"- **{m['name']}**: {m['note']}" for m in models if m["note"])
    shards = "\n".join(
        f"- `{name}`: `{s['bin']}` (SHA-256 `{s['bin_sha256']}`), {s['eval_tokens']:,} tokens evaluated"
        for name, s in results["shards"].items()
    )
    return f"""# Phase 9 Standard Evaluation

Generated by `phase9/eval_standard.py`. Rows are sorted by bits per byte; lower is better.

| Model | Parameters | Vocab | Bits per Byte | Loss | Perplexity | Loss SE | Tokens | Bytes | Tok/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Loss and perplexity are per token, so they can only be compared between models that share a tokenizer. Bits per byte can be compared across tokenizers. Loss SE is the standard error across blocks; it ignores correlation between neighbouring blocks.

{notes}

## Method

- **Text**: `{results['source_text']}` (SHA-256 `{results['source_text_sha256']}`)
- **Chunks**: non-overlapping blocks from the start of the shard; the last partial block is dropped.
- **Unknown tokens**: counted as {results['unk_token_bytes']} bytes each.
{shards}
"""


def save_results(results: Dict[str, Any], results_path: str = RESULTS_PATH, report_path: str = REPORT_PATH) -> None:
    os.makedirs(os.path.dirname(project_path(results_path)), exist_ok=True)
    with open(project_path(results_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(project_path(report_path), "w", encoding="utf-8") as f:
        f.write(build_report(results).strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 9 standard evaluation")
    parser.add_argument("--all", action="store_true", help="Evaluate every model listed in eval_config.json")
    parser.add_argument("--checkpoint", help="Evaluate one checkpoint")
    parser.add_argument("--name", help="Display name for --checkpoint")
    parser.add_argument("--shard", default="v1024", help="Shard name for --checkpoint")
    parser.add_argument("--note", default="", help="Note shown in the report for --checkpoint")
    args = parser.parse_args()
    if not args.all and not (args.checkpoint and args.name):
        parser.error("Pass --all, or --checkpoint with --name")

    eval_config = load_eval_config()
    device = resolve_device("auto")
    if device.type == "cpu":
        torch.set_num_threads(eval_config["cpu_threads"])

    results_path = project_path(RESULTS_PATH)
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        first_shard = next(iter(eval_config["shards"].values()))
        results = {
            "source_text": first_shard["source_text"],
            "source_text_sha256": first_shard["source_text_sha256"],
            "unk_token_bytes": eval_config["unk_token_bytes"],
            "shards": eval_config["shards"],
            "models": {},
        }

    if args.all:
        entries = eval_config["models"]
    else:
        entries = [{"name": args.name, "checkpoint": args.checkpoint, "shard": args.shard, "note": args.note}]
    for entry in entries:
        results["models"][entry["name"]] = evaluate_entry(entry, eval_config, device)
        save_results(results)
    print(f"Wrote {RESULTS_PATH} and {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()

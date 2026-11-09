"""
Phase 11A — Supervised fine-tuning from a pretrained v3 checkpoint. Loss is only on assistant tokens, with calculator
results masked (phase11/chat_format.py). Uses train_v3's model, bf16 autocast, fused AdamW, compile path and checkpoints.

Batches are computed from the seed before training starts, so a resume needs only the step number.
Output in out_dir: checkpoint.pt, best_model.pt, final_model.pt, metrics.json, steps.jsonl, generations.jsonl.

Usage:
    python phase11/sft_train.py --config phase11/configs/sft_v1.json --auto-resume
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from model import MiniGPTConfig
from phase10.common import atomic_write_json, project_path, sha256_file, sha256_text
from phase10.model_v3 import MiniGPTV3
from phase10.train_v3 import compile_model, query_gpu_temperature, save_best_model, save_checkpoint, truncate_jsonl
from phase11.chat import generate_reply
from phase11.chat_format import IGNORE_INDEX, PAD, build_labels, example_text, special_ids
from phase9.train_v2 import accumulate_gradients, autocast_context, build_param_groups, get_learning_rate, query_gpu_utilization, resolve_device

DEFAULT_TRAINING = {
    "epochs": 2,
    "max_steps": None,
    "micro_batch_size": 16,
    "grad_accum_steps": 2,
    "learning_rate": 2e-4,
    "min_lr": 2e-5,
    "warmup_steps": 50,
    "weight_decay": 0.0,
    "betas": [0.9, 0.95],
    "grad_clip": 1.0,
    "seed": 11,
    "pad_lengths": [128, 256, 512],
    "eval_interval": 250,
    "eval_max_examples": 2000,
    "eval_batch_size": 16,
    "log_interval": 20,
    "checkpoint_interval_minutes": 30,
    "checkpoint_every_steps": 0,
    "stop_after_step": None,
    "compile": True,
    "cpu_threads": 8,
    "generate_max_new_tokens": 64,
}
SAMPLE_PROMPTS = ["What is 17 plus 26?", "Who are you?", "What is the capital of Italy?", "Who will win the 2099 Velmora Cup?", "How do I break into a stranger's car?", "How do I kill a Python process?"]
RESUME_LOCKED = ("epochs", "max_steps", "micro_batch_size", "grad_accum_steps", "learning_rate", "min_lr", "warmup_steps", "weight_decay", "betas", "grad_clip", "seed", "pad_lengths")

Example = Tuple[List[int], List[int]]


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for section in ("init_checkpoint", "data", "out_dir"):
        if section not in config:
            raise ValueError(f"Config {path} is missing required key '{section}'")
    training = dict(DEFAULT_TRAINING)
    training.update(config.get("training", {}))
    training.update(overrides or {})
    config["training"] = training
    return config


def read_rows(path: str, expected_sha256: Optional[str] = None) -> List[Dict[str, Any]]:
    if expected_sha256 and sha256_text(path) != expected_sha256:
        raise ValueError(f"Hash differs from the pin for {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode_examples(tokenizer: Tokenizer, rows: Sequence[Dict[str, Any]], block_size: int) -> List[Example]:
    """Each example gives inputs and targets shifted by one. Targets are masked outside what the model must produce."""
    ids_of = special_ids(tokenizer)
    encodings = tokenizer.encode_batch_fast([example_text(row["prompt"], row["text"]) for row in rows], add_special_tokens=False)
    examples = []
    for row, encoding in zip(rows, encodings):
        ids = encoding.ids
        if len(ids) - 1 > block_size:
            raise ValueError(f"Example {row.get('id')} has {len(ids)} tokens, over the block size")
        labels = build_labels(ids, ids_of)
        if all(label == IGNORE_INDEX for label in labels[1:]):
            raise ValueError(f"Example {row.get('id')} has no token to train on")
        examples.append((ids[:-1], labels[1:]))
    return examples


def encode_replay(rows: Sequence[Dict[str, Any]], block_size: int) -> List[Example]:
    """Pretraining-text replay rows ({"replay_ids": [...]}): inputs and targets shifted by one, loss on every token."""
    examples = []
    for row in rows:
        ids = row["replay_ids"]
        if len(ids) - 1 > block_size or len(ids) < 2:
            raise ValueError(f"Replay row {row.get('id')} has {len(ids)} tokens")
        examples.append((ids[:-1], ids[1:]))
    return examples


def make_batches(lengths: Sequence[int], batch_size: int, epochs: int, seed: int) -> List[List[int]]:
    """Shuffles each epoch and groups similar lengths to save padding, in a fixed order for the seed. Drops each epoch's partial batch."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    all_batches: List[List[int]] = []
    for _ in range(epochs):
        order = rng.permutation(len(lengths))
        batches: List[List[int]] = []
        pool_size = batch_size * 100
        for start in range(0, len(order), pool_size):
            pool = order[start:start + pool_size]
            pool = pool[np.argsort(lengths[pool], kind="stable")]
            for begin in range(0, len(pool) - batch_size + 1, batch_size):
                batches.append(pool[begin:begin + batch_size].tolist())
        rng.shuffle(batches)
        all_batches.extend(batches)
    return all_batches


def collate(examples: Sequence[Example], indices: Sequence[int], pad_id: int, pad_lengths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pads to the smallest fixed length that fits, so a compiled model sees few shapes. Attention is causal, so padding cannot affect earlier tokens."""
    longest = max(len(examples[i][0]) for i in indices)
    length = next((value for value in pad_lengths if value >= longest), longest)
    x = torch.full((len(indices), length), pad_id, dtype=torch.long)
    y = torch.full((len(indices), length), IGNORE_INDEX, dtype=torch.long)
    for row, index in enumerate(indices):
        inputs, targets = examples[index]
        x[row, :len(inputs)] = torch.tensor(inputs, dtype=torch.long)
        y[row, :len(targets)] = torch.tensor(targets, dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate_masked(model: nn.Module, examples: Sequence[Example], pad_id: int, pad_lengths: Sequence[int], batch_size: int, device: torch.device) -> float:
    """Mean loss per trained token, over examples in file order."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    for start in range(0, len(order), batch_size):
        x, y = collate(examples, order[start:start + batch_size], pad_id, pad_lengths)
        with autocast_context(device):
            _, loss = model(x.to(device), y.to(device))
        tokens = int((y != IGNORE_INDEX).sum())
        total += loss.item() * tokens
        count += tokens
    if was_training:
        model.train()
    return total / max(count, 1)


def train(config: Dict[str, Any], resume: bool = False) -> Dict[str, Any]:
    training = config["training"]
    out_dir = project_path(config["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    metrics_path = os.path.join(out_dir, "metrics.json")
    steps_path = os.path.join(out_dir, "steps.jsonl")
    generations_path = os.path.join(out_dir, "generations.jsonl")

    device = resolve_device(config.get("device", "auto"))
    if device.type == "cpu":
        torch.set_num_threads(training["cpu_threads"])
    torch.manual_seed(training["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["seed"])
        torch.cuda.reset_peak_memory_stats()

    data = config["data"]
    tokenizer_path = project_path(data["tokenizer"])
    if data.get("tokenizer_sha256") and sha256_text(tokenizer_path) != data["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash differs from its pin")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = special_ids(tokenizer)[PAD]

    init_path = project_path(config["init_checkpoint"])
    if config.get("init_checkpoint_sha256") and sha256_file(init_path) != config["init_checkpoint_sha256"]:
        raise ValueError("Pretrained checkpoint hash differs from its pin")
    init_state = torch.load(init_path, map_location="cpu", weights_only=True)
    model_cfg = init_state["config"]["model"]
    config["model"] = model_cfg
    raw_model = MiniGPTV3(MiniGPTConfig(**model_cfg))
    raw_model.load_state_dict(init_state["model_state"])
    raw_model.to(device)
    del init_state
    if tokenizer.get_vocab_size() != model_cfg["vocab_size"]:
        raise ValueError("Tokenizer and model vocabulary sizes differ")

    block_size = model_cfg["block_size"]
    train_examples = encode_examples(tokenizer, read_rows(project_path(data["train"]), data.get("train_sha256")), block_size)
    if data.get("replay"):
        # Phase 17E: pretraining-text replay. Pre-tokenized windows; every token carries loss. Pinned by SHA-256.
        replay_examples = encode_replay(read_rows(project_path(data["replay"]["file"]), data["replay"]["sha256"]), block_size)
        print(f"Replay: {len(replay_examples):,} windows, {sum(len(e[1]) for e in replay_examples):,} loss tokens", flush=True)
        train_examples = train_examples + replay_examples
    val_rows = read_rows(project_path(data["val"]), data.get("val_sha256"))[:training["eval_max_examples"]]
    val_examples = encode_examples(tokenizer, val_rows, block_size)

    batches = make_batches([len(e[0]) for e in train_examples], training["micro_batch_size"], training["epochs"], training["seed"])
    accum = training["grad_accum_steps"]
    planned_steps = len(batches) // accum
    max_steps = min(planned_steps, training["max_steps"]) if training["max_steps"] else planned_steps
    if max_steps < 1:
        raise ValueError("Not enough examples for one step")

    optimizer = torch.optim.AdamW(build_param_groups(raw_model, training["weight_decay"]), lr=training["learning_rate"], betas=tuple(training["betas"]), fused=device.type == "cuda")
    metrics: Dict[str, Any] = {
        "config": config, "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "autocast": "bf16" if device.type == "cuda" else "disabled", "fused_adamw": device.type == "cuda", "parameters": raw_model.get_num_params(),
        "train_examples": len(train_examples), "val_examples": len(val_examples), "max_steps": max_steps, "evals": [], "resumes": [], "train_seconds": 0.0,
        "max_grad_norm": 0.0, "non_finite_count": 0, "completed_steps": 0, "status": "running",
    }
    start_step = 0
    best_val_loss = float("inf")
    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Cannot resume: {checkpoint_path} does not exist")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        for key in RESUME_LOCKED:
            if state["config"]["training"][key] != training[key]:
                raise ValueError(f"Cannot resume: training setting '{key}' differs from the checkpoint")
        raw_model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_step = state["step"]
        best_val_loss = state["best_val_loss"]
        metrics = state["metrics"]
        metrics["status"] = "running"
        metrics["resumes"].append({"from_step": start_step, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        truncate_jsonl(steps_path, start_step)
        truncate_jsonl(generations_path, start_step)
        print(f"Resumed from step {start_step}", flush=True)
    else:
        for path in (steps_path, generations_path):
            if os.path.exists(path):
                os.remove(path)

    train_model, compile_status = compile_model(raw_model, training, block_size, device)
    metrics["compile"] = compile_status
    raw_model.train()
    print(f"Device: {metrics['device_name']} | Autocast: {metrics['autocast']} | Compile: {compile_status} | Parameters: {metrics['parameters']:,}", flush=True)
    print(f"Examples: {len(train_examples):,} train, {len(val_examples):,} val | Steps: {max_steps:,} | {training['micro_batch_size']} x {accum} examples per step", flush=True)

    last_checkpoint = time.monotonic()
    placeholder_generator = torch.Generator()
    with open(steps_path, "a", encoding="utf-8") as steps_file:
        for step in range(start_step + 1, max_steps + 1):
            step_start = time.perf_counter()
            lr = get_learning_rate(step, max_steps, training["warmup_steps"], training["learning_rate"], training["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            micro_batches = [collate(train_examples, batches[(step - 1) * accum + i], pad_id, training["pad_lengths"]) for i in range(accum)]
            optimizer.zero_grad(set_to_none=True)
            loss = accumulate_gradients(train_model, micro_batches, device)
            grad_norm = nn.utils.clip_grad_norm_(raw_model.parameters(), training["grad_clip"]).item()
            steps_file.write(json.dumps({"step": step, "loss": loss, "grad_norm": grad_norm, "lr": lr}) + "\n")
            steps_file.flush()
            if not (math.isfinite(loss) and math.isfinite(grad_norm)):
                metrics["non_finite_count"] += 1
                metrics["status"] = "failed_non_finite"
                metrics["failed_step"] = step
                atomic_write_json(metrics_path, metrics)
                raise RuntimeError(f"Non-finite value at step {step}: loss {loss}, grad norm {grad_norm}")
            optimizer.step()
            metrics["train_seconds"] += time.perf_counter() - step_start
            metrics["completed_steps"] = step
            metrics["max_grad_norm"] = max(metrics["max_grad_norm"], grad_norm)

            is_last = step == max_steps
            stopping = step == training["stop_after_step"]
            if step % training["log_interval"] == 0 or step == 1 or is_last or stopping:
                gpu = query_gpu_utilization() if device.type == "cuda" else None
                print(f"step {step:5d}/{max_steps} | loss {loss:.4f} | grad {grad_norm:.3f} | lr {lr:.6f} | gpu {gpu}%", flush=True)
            if step % training["eval_interval"] == 0 or is_last or stopping:
                val_loss = evaluate_masked(raw_model, val_examples, pad_id, training["pad_lengths"], training["eval_batch_size"], device)
                entry = {"step": step, "val_loss": val_loss, "gpu_temp_c": query_gpu_temperature() if device.type == "cuda" else None}
                metrics["evals"].append(entry)
                print(f"  >>> step {step} val loss {val_loss:.4f}", flush=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    metrics["best_step"] = step
                    metrics["best_val_loss"] = val_loss
                    save_best_model(os.path.join(out_dir, "best_model.pt"), raw_model, config, step, entry)
                raw_model.eval()
                with open(generations_path, "a", encoding="utf-8") as f:
                    for prompt in SAMPLE_PROMPTS:
                        reply = generate_reply(raw_model, tokenizer, prompt, device, training["generate_max_new_tokens"])
                        f.write(json.dumps({"step": step, "prompt": prompt, "raw": reply.raw}, ensure_ascii=False) + "\n")
                        print(f"      {prompt!r} -> {reply.raw!r}", flush=True)
                raw_model.train()
                atomic_write_json(metrics_path, metrics)
            timed_out = time.monotonic() - last_checkpoint >= training["checkpoint_interval_minutes"] * 60
            by_steps = training["checkpoint_every_steps"] and step % training["checkpoint_every_steps"] == 0
            if is_last or stopping or timed_out or by_steps:
                if is_last:
                    metrics["status"] = "completed"
                    save_best_model(os.path.join(out_dir, "final_model.pt"), raw_model, config, step, metrics["evals"][-1])
                elif stopping:
                    metrics["status"] = "stopped_early"
                save_checkpoint(checkpoint_path, raw_model, optimizer, placeholder_generator, config, step, best_val_loss, metrics)
                atomic_write_json(metrics_path, metrics)
                last_checkpoint = time.monotonic()
            if stopping:
                break
    atomic_write_json(metrics_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 11 SFT trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    overrides = {key: value for key, value in {"max_steps": args.max_steps, "stop_after_step": args.stop_after_step}.items() if value is not None}
    config = load_config(args.config, overrides)
    if args.out_dir:
        config["out_dir"] = args.out_dir
    if args.init_checkpoint:
        config["init_checkpoint"] = args.init_checkpoint
    out_dir = project_path(config["out_dir"])
    has_checkpoint = os.path.exists(os.path.join(out_dir, "checkpoint.pt"))
    metrics_path = os.path.join(out_dir, "metrics.json")
    if args.auto_resume and has_checkpoint and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            if json.load(f).get("status") == "completed":
                print("Already completed; nothing to do", flush=True)
                return
    summary = train(config, resume=args.resume or (args.auto_resume and has_checkpoint))
    print(f"Finished: status={summary['status']} steps={summary['completed_steps']} best_val_loss={summary.get('best_val_loss')}", flush=True)


if __name__ == "__main__":
    main()

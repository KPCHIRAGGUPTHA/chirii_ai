"""
Phase 10A — Trainer v3. Built on phase9/train_v2.py and reusing its sampling, weight-decay groups, learning-rate
schedule, averaged micro-batch loss and bf16 autocast. Adds SDPA attention, fused AdamW, torch.compile with a clean
fallback, per-step grad norms, evaluation in bits per byte on a pinned subset, fixed-prompt generation, and timed
checkpoints with exact resume after a kill.

Output in out_dir: checkpoint.pt, best_model.pt (weights only), metrics.json, steps.jsonl (one line per step),
generations.jsonl.

Usage:
    python phase10/train_v3.py --config phase10/configs/foundation_v3_clean_2ep.json --auto-resume
    python phase10/train_v3.py --config phase10/configs/foundation_v3.json --seed 1 --stop-after-step 300 --eval-interval 50 --out-dir phase10/runs/stability_seed1
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import MiniGPTConfig
from phase10.common import atomic_write_json, project_path, sha256_text
from phase10.model_v3 import MiniGPTV3
from phase9.eval_standard import evaluate_tokens
from phase9.train_v2 import (
    accumulate_gradients,
    autocast_context,
    build_param_groups,
    get_batch,
    get_learning_rate,
    open_tokens,
    query_gpu_utilization,
    resolve_device,
    sample_positions,
)

DEFAULT_TRAINING = {
    "max_steps": 5341,
    "micro_batch_size": 64,
    "grad_accum_steps": 4,
    "learning_rate": 8e-4,
    "min_lr": 8e-5,
    "warmup_steps": 200,
    "weight_decay": 0.1,
    "betas": [0.9, 0.95],
    "grad_clip": 1.0,
    "seed": 1,
    "eval_interval": 250,
    "eval_batch_size": 16,
    "log_interval": 50,
    "checkpoint_interval_minutes": 30,
    "checkpoint_every_steps": 0,
    "snapshot_every_steps": 0,
    "stop_after_step": None,
    "compile": True,
    "cpu_threads": 8,
    "target_tokens_per_sec": 40000,
    "generate_max_new_tokens": 64,
}

FIXED_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "2 + 2 =",
    "The main reason plants need sunlight is",
    "def add(a, b):\n",
]

# Settings a resume must not change, or the run would no longer be one continuous training run
RESUME_LOCKED = ("max_steps", "micro_batch_size", "grad_accum_steps", "learning_rate", "min_lr", "warmup_steps",
                 "weight_decay", "betas", "grad_clip", "seed")


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for section in ("model", "data", "out_dir"):
        if section not in config:
            raise ValueError(f"Config {path} is missing required key '{section}'")
    training = dict(DEFAULT_TRAINING)
    training.update(config.get("training", {}))
    training.update(overrides or {})
    config["training"] = training
    return config


def compile_model(model: nn.Module, training: Dict[str, Any], block_size: int, device: torch.device) -> Tuple[nn.Module, str]:
    """Returns the model to train with and what happened. Any compile failure falls back to the plain model."""
    if not training["compile"]:
        return model, "disabled"
    try:
        compiled = torch.compile(model)
        # Compilation is lazy, so run one real training pass now to surface any failure before training starts
        x = torch.zeros((training["micro_batch_size"], block_size), dtype=torch.long, device=device)
        with autocast_context(device):
            _, loss = compiled(x, x)
        loss.backward()
        model.zero_grad(set_to_none=True)
        return compiled, "enabled"
    except Exception as error:
        model.zero_grad(set_to_none=True)
        print(f"torch.compile failed, using the plain model: {type(error).__name__}: {error}", flush=True)
        return model, f"failed: {type(error).__name__}"


@torch.no_grad()
def generate_greedy(model: nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    """Top-1 generation, so the text changes only when the weights do. Uses no random numbers."""
    block_size = model.config.block_size
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with autocast_context(device):
            logits, _ = model(idx[:, -block_size:])
        idx = torch.cat((idx, logits[:, -1, :].float().argmax(dim=-1, keepdim=True)), dim=1)
    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=False)


def query_gpu_temperature() -> Optional[int]:
    """Returns GPU temperature in Celsius from nvidia-smi, or None if it cannot be read."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return int(output.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def truncate_jsonl(path: str, last_step: int) -> None:
    """After a kill, drops lines from steps that the resumed run will repeat. A half-written last line is dropped too."""
    if not os.path.exists(path):
        return
    kept = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry["step"] <= last_step:
                kept.append(line)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(kept)
    os.replace(tmp_path, path)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                    config: Dict[str, Any], step: int, best_val_loss: float, metrics: Dict[str, Any]) -> None:
    """Writes atomically, so a kill during a save leaves the previous checkpoint intact."""
    state = {
        "format": "phase10_train_v3",
        "step": step,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_rng_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "best_val_loss": best_val_loss,
        "metrics": metrics,
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def save_best_model(path: str, model: nn.Module, config: Dict[str, Any], step: int, val: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    torch.save({"format": "phase10_train_v3_weights", "step": step, "config": config, "model_state": model.state_dict(), "val": val}, tmp_path)
    os.replace(tmp_path, path)


def load_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Verifies every pin that is set and returns train tokens, the validation subset, token byte sizes and the tokenizer."""
    data = config["data"]
    train_tokens = open_tokens(project_path(data["train_bin"]), data.get("train_sha256"))
    val_tokens = open_tokens(project_path(data["val_bin"]), data.get("val_sha256"))
    subset = val_tokens[:data["val_subset_tokens"] + 1]
    if len(subset) != data["val_subset_tokens"] + 1:
        raise ValueError(f"Validation data has only {len(val_tokens):,} tokens")
    if data.get("val_subset_sha256") and hashlib.sha256(subset.tobytes()).hexdigest() != data["val_subset_sha256"]:
        raise ValueError("Validation subset hash differs from its pin")

    if data.get("tokenizer"):
        from tokenizers import Tokenizer
        from phase10.tokenizer_v2 import token_byte_lengths
        tokenizer_path = project_path(data["tokenizer"])
        if data.get("tokenizer_sha256") and sha256_text(tokenizer_path) != data["tokenizer_sha256"]:
            raise ValueError("Tokenizer hash differs from its pin")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        if tokenizer.get_vocab_size() != config["model"]["vocab_size"]:
            raise ValueError("Tokenizer and model vocabulary sizes differ")
        return train_tokens, subset, token_byte_lengths(tokenizer), tokenizer
    # Tests run without a tokenizer: each token counts as one byte and generation is skipped
    return train_tokens, subset, np.ones(config["model"]["vocab_size"], dtype=np.int64), None


def train(config: Dict[str, Any], resume: bool = False) -> Dict[str, Any]:
    training = config["training"]
    model_cfg = config["model"]
    out_dir = project_path(config["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    best_path = os.path.join(out_dir, "best_model.pt")
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
    generator = torch.Generator().manual_seed(training["seed"])

    train_tokens, val_subset, token_bytes, tokenizer = load_data(config)

    raw_model = MiniGPTV3(MiniGPTConfig(**model_cfg)).to(device)
    init = config.get("init_checkpoint")
    if init and not resume:
        # Continued pretraining (Phase 17D): start from earlier weights with a fresh optimizer. Pinned by SHA-256.
        from phase10.common import sha256_file
        init_path = project_path(init["file"])
        if sha256_file(init_path) != init["sha256"]:
            raise ValueError(f"init_checkpoint {init['file']} differs from its pin")
        init_state = torch.load(init_path, map_location="cpu", weights_only=True)
        if init_state["config"]["model"] != model_cfg:
            raise ValueError("init_checkpoint model config differs from this config")
        raw_model.load_state_dict(init_state["model_state"])
        print(f"Initialised from {init['file']} (step {init_state.get('step')})", flush=True)
    optimizer = torch.optim.AdamW(
        build_param_groups(raw_model, training["weight_decay"]),
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
        fused=device.type == "cuda",
    )

    block_size = model_cfg["block_size"]
    tokens_per_step = training["micro_batch_size"] * training["grad_accum_steps"] * block_size
    start_step = 0
    best_val_loss = float("inf")
    metrics = {
        "config": config,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "autocast": "bf16" if device.type == "cuda" else "disabled",
        "fused_adamw": device.type == "cuda",
        "parameters": raw_model.get_num_params(),
        "train_tokens_available": len(train_tokens),
        "tokens_per_step": tokens_per_step,
        "token_budget": training["max_steps"] * tokens_per_step,
        "evals": [],
        "resumes": [],
        "train_seconds": 0.0,
        "wall_seconds": 0.0,
        "max_grad_norm": 0.0,
        "max_grad_norm_step": None,
        "max_grad_norm_after_warmup": 0.0,
        "max_grad_norm_after_warmup_step": None,
        "non_finite_count": 0,
        "completed_steps": 0,
        "status": "running",
    }

    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Cannot resume: {checkpoint_path} does not exist")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if state["config"]["model"] != model_cfg:
            raise ValueError("Cannot resume: checkpoint model config differs from this config")
        for key in RESUME_LOCKED:
            if state["config"]["training"][key] != training[key]:
                raise ValueError(f"Cannot resume: training setting '{key}' differs from the checkpoint")
        raw_model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["sampler_rng_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_state"]])
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

    print(f"Device: {metrics['device_name']} | Autocast: {metrics['autocast']} | Fused AdamW: {metrics['fused_adamw']} | Compile: {compile_status} | Parameters: {metrics['parameters']:,}", flush=True)
    print(f"Train tokens: {len(train_tokens):,} | Steps: {training['max_steps']:,} | Tokens per step: {tokens_per_step:,} ({training['micro_batch_size']} x {training['grad_accum_steps']} x {block_size}) | Budget: {metrics['token_budget']:,}", flush=True)

    def add_time(key: str, since: float) -> float:
        now = time.perf_counter()
        metrics[key] += now - since
        return now

    def record_memory() -> None:
        if device.type == "cuda":
            metrics["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            metrics["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    wall_mark = time.perf_counter()
    last_checkpoint = time.monotonic()
    window_tokens = 0
    window_seconds = 0.0

    with open(steps_path, "a", encoding="utf-8") as steps_file:
        for step in range(start_step + 1, training["max_steps"] + 1):
            step_start = time.perf_counter()
            lr = get_learning_rate(step, training["max_steps"], training["warmup_steps"], training["learning_rate"], training["min_lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            micro_batches = [
                get_batch(train_tokens, sample_positions(len(train_tokens), block_size, training["micro_batch_size"], generator), block_size)
                for _ in range(training["grad_accum_steps"])
            ]
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
            now = time.perf_counter()
            elapsed = now - step_start
            metrics["train_seconds"] += elapsed
            window_seconds += elapsed
            window_tokens += tokens_per_step
            metrics["completed_steps"] = step
            if grad_norm > metrics["max_grad_norm"]:
                metrics["max_grad_norm"], metrics["max_grad_norm_step"] = grad_norm, step
            if step > training["warmup_steps"] and grad_norm > metrics["max_grad_norm_after_warmup"]:
                metrics["max_grad_norm_after_warmup"], metrics["max_grad_norm_after_warmup_step"] = grad_norm, step

            is_last = step == training["max_steps"]
            stopping = step == training["stop_after_step"]

            if step % training["log_interval"] == 0 or step == 1 or is_last or stopping:
                gpu_util = query_gpu_utilization() if device.type == "cuda" else None
                record_memory()
                print(
                    f"step {step:5d}/{training['max_steps']} | loss {loss:.4f} | grad {grad_norm:.3f} | lr {lr:.6f} | "
                    f"{window_tokens / max(window_seconds, 1e-6):,.0f} tok/s | gpu {gpu_util}% | peak {metrics.get('gpu_peak_allocated_gb')} GB",
                    flush=True,
                )
                window_tokens = 0
                window_seconds = 0.0

            if step % training["eval_interval"] == 0 or is_last or stopping:
                with autocast_context(device):
                    val = evaluate_tokens(raw_model, val_subset, token_bytes, block_size, training["eval_batch_size"], device)
                raw_model.train()
                gpu_temp = query_gpu_temperature() if device.type == "cuda" else None
                entry = {"step": step, "val_loss": val["loss"], "val_bits_per_byte": val["bits_per_byte"], "val_tokens": val["eval_tokens"], "val_bytes": val["eval_bytes"], "gpu_temp_c": gpu_temp}
                metrics["evals"].append(entry)
                print(f"  >>> step {step} val loss {val['loss']:.4f} | {val['bits_per_byte']:.4f} bits/byte | gpu temp {gpu_temp} C", flush=True)
                if val["loss"] < best_val_loss:
                    best_val_loss = val["loss"]
                    metrics["best_step"] = step
                    metrics["best_val_loss"] = val["loss"]
                    metrics["best_val_bits_per_byte"] = val["bits_per_byte"]
                    save_best_model(best_path, raw_model, config, step, entry)
                if tokenizer is not None:
                    raw_model.eval()
                    with open(generations_path, "a", encoding="utf-8") as f:
                        for prompt in FIXED_PROMPTS:
                            text = generate_greedy(raw_model, tokenizer, prompt, training["generate_max_new_tokens"], device)
                            f.write(json.dumps({"step": step, "prompt": prompt, "text": text}, ensure_ascii=False) + "\n")
                            print(f"      {text!r}", flush=True)
                    raw_model.train()
                wall_mark = add_time("wall_seconds", wall_mark)
                record_memory()
                atomic_write_json(metrics_path, metrics)

            # Weights only, one file per step, never overwritten by a later snapshot
            snapshot_every = training["snapshot_every_steps"]
            if snapshot_every and (step % snapshot_every == 0 or is_last):
                snapshot_dir = os.path.join(out_dir, "snapshots")
                os.makedirs(snapshot_dir, exist_ok=True)
                last_eval = metrics["evals"][-1] if metrics["evals"] and metrics["evals"][-1]["step"] == step else None
                save_best_model(os.path.join(snapshot_dir, f"model_step_{step:05d}.pt"), raw_model, config, step, last_eval)
                print(f"  snapshot saved at step {step}", flush=True)

            timed_out = time.monotonic() - last_checkpoint >= training["checkpoint_interval_minutes"] * 60
            by_steps = training["checkpoint_every_steps"] and step % training["checkpoint_every_steps"] == 0
            if is_last or stopping or timed_out or by_steps:
                if is_last:
                    metrics["status"] = "completed"
                elif stopping:
                    metrics["status"] = "stopped_early"
                wall_mark = add_time("wall_seconds", wall_mark)
                record_memory()
                save_checkpoint(checkpoint_path, raw_model, optimizer, generator, config, step, best_val_loss, metrics)
                atomic_write_json(metrics_path, metrics)
                last_checkpoint = time.monotonic()
                print(f"  checkpoint saved at step {step}", flush=True)
            if stopping:
                break

    metrics["tokens_per_sec_training"] = metrics["completed_steps"] * tokens_per_step / max(metrics["train_seconds"], 1e-6)
    metrics["tokens_per_sec_overall"] = metrics["completed_steps"] * tokens_per_step / max(metrics["wall_seconds"], 1e-6)
    atomic_write_json(metrics_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 10 trainer v3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint.pt; fails if there is none")
    parser.add_argument("--auto-resume", action="store_true", help="Continue from checkpoint.pt if it exists, else start fresh")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop-after-step", type=int, default=None, help="Stop early but keep the full learning-rate schedule")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    overrides = {key: value for key, value in {
        "seed": args.seed, "stop_after_step": args.stop_after_step, "eval_interval": args.eval_interval,
        "checkpoint_every_steps": args.checkpoint_every_steps, "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
    }.items() if value is not None}
    config = load_config(args.config, overrides)
    if args.out_dir:
        config["out_dir"] = args.out_dir

    out_dir = project_path(config["out_dir"])
    has_checkpoint = os.path.exists(os.path.join(out_dir, "checkpoint.pt"))
    metrics_path = os.path.join(out_dir, "metrics.json")
    if args.auto_resume and has_checkpoint and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            if json.load(f).get("status") == "completed":
                print("Already completed; nothing to do", flush=True)
                return
    summary = train(config, resume=args.resume or (args.auto_resume and has_checkpoint))
    print(f"Finished: status={summary['status']} steps={summary['completed_steps']} best_val_loss={summary.get('best_val_loss')} "
          f"train tok/s={summary['tokens_per_sec_training']:,.0f} overall tok/s={summary['tokens_per_sec_overall']:,.0f}", flush=True)


if __name__ == "__main__":
    main()

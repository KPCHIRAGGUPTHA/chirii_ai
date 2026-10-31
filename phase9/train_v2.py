"""
Phase 9 — Corrected pretraining trainer (replaces phase8/train_model_scaling.py for new runs).
Random sampling over all training tokens, AdamW with selective weight decay, averaged micro-batch
losses, auto device detection, bf16 autocast on CUDA, JSON config in, JSON metrics out, checkpoint + resume.

Usage:
    python phase9/train_v2.py --config phase9/configs/model_d_v2.json
    python phase9/train_v2.py --config phase9/configs/model_d_v2.json --max-steps 2 --out-dir phase9/runs/dry
    python phase9/train_v2.py --config phase9/configs/model_d_v2.json --resume
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import MiniGPT, MiniGPTConfig

DEFAULT_TRAINING = {
    "max_steps": 2500,
    "micro_batch_size": 8,
    "grad_accum_steps": 4,
    "learning_rate": 1e-3,
    "min_lr": 1e-4,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "grad_clip": 1.0,
    "seed": 42,
    "eval_interval": 100,
    "eval_tokens": 64000,
    "eval_batch_size": 16,
    "log_interval": 50,
    "checkpoint_interval": 100,
    "cpu_threads": 8,
    "warn_min_tokens_per_sec": 10000,
    "warn_min_gpu_util_pct": 50,
    "stop_after_step": None,
}


def project_path(path: str) -> str:
    """Resolves a repo-relative path; absolute paths pass through unchanged."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reads a JSON config and fills training defaults. `overrides` replaces top-level training keys."""
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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def open_tokens(path: str, expected_sha256: Optional[str] = None) -> np.memmap:
    """Opens a raw uint16 token file as a read-only memmap, so it is never fully loaded into RAM."""
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def sample_positions(num_tokens: int, block_size: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform random start positions over the whole token file, including the final valid block."""
    if num_tokens <= block_size:
        raise ValueError(f"Token file ({num_tokens}) must be longer than block_size ({block_size})")
    return torch.randint(0, num_tokens - block_size, (batch_size,), generator=generator)


def get_batch(tokens: np.ndarray, positions: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copies the sampled windows out of the memmap as (input, target) tensors."""
    x = np.stack([tokens[p : p + block_size] for p in positions.tolist()]).astype(np.int64)
    y = np.stack([tokens[p + 1 : p + 1 + block_size] for p in positions.tolist()]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def build_param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Decays Linear weights only; biases, LayerNorm and embeddings (including the tied lm_head) get no decay."""
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
        elif isinstance(module, nn.Linear) and module.bias is not None:
            no_decay_ids.add(id(module.bias))

    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if id(param) in no_decay_ids else decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_learning_rate(step: int, max_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay, matching Model C's schedule. `step` is 1-indexed."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if max_steps <= warmup_steps:
        return min_lr
    ratio = min(1.0, (step - warmup_steps) / (max_steps - warmup_steps))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (peak_lr - min_lr)


def resolve_device(requested: str = "auto") -> torch.device:
    """auto selects CUDA when available and CPU otherwise."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def autocast_context(device: torch.device):
    """bf16 autocast on CUDA; a disabled context on CPU. bf16 needs no GradScaler."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.amp.autocast("cpu", enabled=False)


def accumulate_gradients(model: nn.Module, micro_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    """Backpropagates each micro-batch at 1/N weight and returns the mean micro-batch loss."""
    total = 0.0
    for x, y in micro_batches:
        with autocast_context(device):
            _, loss = model(x.to(device), y.to(device))
        (loss / len(micro_batches)).backward()
        total += loss.item()
    return total / len(micro_batches)


@torch.no_grad()
def evaluate_loss(model: nn.Module, tokens: np.ndarray, block_size: int, num_tokens: int, batch_size: int, device: torch.device) -> float:
    """Mean loss over the first `num_tokens` tokens in non-overlapping chunks, for checkpoint selection only."""
    was_training = model.training
    model.eval()
    num_blocks = min(num_tokens, len(tokens) - 1) // block_size
    if num_blocks < 1:
        raise ValueError("Validation data is shorter than one block")
    total = 0.0
    for start in range(0, num_blocks, batch_size):
        count = min(batch_size, num_blocks - start)
        positions = torch.arange(start, start + count) * block_size
        x, y = get_batch(tokens, positions, block_size)
        with autocast_context(device):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item() * count
    if was_training:
        model.train()
    return total / num_blocks


def query_gpu_utilization() -> Optional[int]:
    """Returns GPU utilization percent from nvidia-smi, or None if it cannot be read."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return int(output.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def throughput_warnings(tokens_per_sec: float, gpu_util_pct: Optional[int], training: Dict[str, Any], device: torch.device) -> List[str]:
    """Flags a badly underused GPU. CPU runs never warn."""
    if device.type != "cuda":
        return []
    warnings = []
    if tokens_per_sec < training["warn_min_tokens_per_sec"]:
        warnings.append(f"Throughput {tokens_per_sec:,.0f} tok/s is below {training['warn_min_tokens_per_sec']:,} tok/s")
    if gpu_util_pct is not None and gpu_util_pct < training["warn_min_gpu_util_pct"]:
        warnings.append(f"GPU utilization {gpu_util_pct}% is below {training['warn_min_gpu_util_pct']}%")
    return warnings


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                    config: Dict[str, Any], step: int, best_val_loss: float, metrics: Dict[str, Any]) -> None:
    """Writes atomically, so an interrupted save cannot corrupt the previous checkpoint."""
    state = {
        "format": "phase9_train_v2",
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


def train(config: Dict[str, Any], resume: bool = False) -> Dict[str, Any]:
    training = config["training"]
    model_cfg = config["model"]
    out_dir = project_path(config["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.pt")
    best_path = os.path.join(out_dir, "best_model.pt")
    metrics_path = os.path.join(out_dir, "metrics.json")

    device = resolve_device(config.get("device", "auto"))
    if device.type == "cpu":
        torch.set_num_threads(training["cpu_threads"])
    torch.manual_seed(training["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["seed"])
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(training["seed"])

    train_tokens = open_tokens(project_path(config["data"]["train_bin"]), config["data"].get("train_sha256"))
    val_tokens = open_tokens(project_path(config["data"]["val_bin"]), config["data"].get("val_sha256"))

    model = MiniGPT(MiniGPTConfig(**model_cfg)).to(device)
    optimizer = torch.optim.AdamW(
        build_param_groups(model, training["weight_decay"]),
        lr=training["learning_rate"],
        betas=tuple(training["betas"]),
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
        "parameters": model.get_num_params(),
        "train_tokens_available": len(train_tokens),
        "tokens_per_step": tokens_per_step,
        "token_budget": training["max_steps"] * tokens_per_step,
        "steps": [],
        "evals": [],
        "warnings": [],
        "status": "running",
    }

    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Cannot resume: {checkpoint_path} does not exist")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if state["config"]["model"] != model_cfg:
            raise ValueError("Cannot resume: checkpoint model config differs from this config")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["sampler_rng_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_state"]])
        start_step = state["step"]
        best_val_loss = state["best_val_loss"]
        metrics = state["metrics"]
        metrics["status"] = "running"
        print(f"Resumed from step {start_step}", flush=True)

    print(f"Device: {metrics['device_name']} | Autocast: {metrics['autocast']} | Parameters: {metrics['parameters']:,}", flush=True)
    print(f"Train tokens: {len(train_tokens):,} | Steps: {training['max_steps']:,} | Token budget: {metrics['token_budget']:,}", flush=True)

    model.train()
    window_start = time.perf_counter()
    window_tokens = 0
    for step in range(start_step + 1, training["max_steps"] + 1):
        lr = get_learning_rate(step, training["max_steps"], training["warmup_steps"], training["learning_rate"], training["min_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        micro_batches = [
            get_batch(train_tokens, sample_positions(len(train_tokens), block_size, training["micro_batch_size"], generator), block_size)
            for _ in range(training["grad_accum_steps"])
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = accumulate_gradients(model, micro_batches, device)
        if not math.isfinite(loss):
            metrics["status"] = "failed_non_finite_loss"
            atomic_write_json(metrics_path, metrics)
            raise RuntimeError(f"Non-finite loss {loss} at step {step}")
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"]).item()
        optimizer.step()
        window_tokens += tokens_per_step

        if step % training["log_interval"] == 0 or step == 1 or step == training["max_steps"]:
            elapsed = time.perf_counter() - window_start
            tokens_per_sec = window_tokens / max(elapsed, 1e-6)
            gpu_util = query_gpu_utilization() if device.type == "cuda" else None
            gpu_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
            entry = {
                "step": step,
                "train_loss": round(loss, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": round(lr, 6),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "gpu_util_pct": gpu_util,
                "gpu_peak_mem_gb": round(gpu_mem_gb, 3) if gpu_mem_gb is not None else None,
            }
            metrics["steps"].append(entry)
            print(
                f"step {step:5d}/{training['max_steps']} | loss {loss:.4f} | grad {grad_norm:.3f} | lr {lr:.6f} | "
                f"{tokens_per_sec:,.0f} tok/s | gpu {gpu_util}% | gpu mem {entry['gpu_peak_mem_gb']} GB",
                flush=True,
            )
            # Step 1 includes startup costs, so it is not a fair throughput sample
            if step > 1:
                for warning in throughput_warnings(tokens_per_sec, gpu_util, training, device):
                    print(f"[THROUGHPUT WARNING] step {step}: {warning}", flush=True)
                    metrics["warnings"].append({"step": step, "message": warning})
            window_start = time.perf_counter()
            window_tokens = 0

        is_last = step == training["max_steps"]
        if step % training["eval_interval"] == 0 or is_last:
            val_loss = evaluate_loss(model, val_tokens, block_size, training["eval_tokens"], training["eval_batch_size"], device)
            metrics["evals"].append({"step": step, "val_loss": round(val_loss, 4), "val_ppl": round(math.exp(val_loss), 2)})
            print(f"  >>> step {step} val loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f})", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                metrics["best_step"] = step
                save_checkpoint(best_path, model, optimizer, generator, config, step, best_val_loss, metrics)

        if step % training["checkpoint_interval"] == 0 or is_last:
            metrics["best_val_loss"] = round(best_val_loss, 4)
            metrics["completed_steps"] = step
            metrics["status"] = "completed" if is_last else "running"
            save_checkpoint(checkpoint_path, model, optimizer, generator, config, step, best_val_loss, metrics)
            atomic_write_json(metrics_path, metrics)

        # Smoke tests stop early while keeping the full run's LR schedule, and can be resumed afterwards
        if step == training["stop_after_step"]:
            metrics["best_val_loss"] = round(best_val_loss, 4)
            metrics["completed_steps"] = step
            metrics["status"] = "stopped_early"
            save_checkpoint(checkpoint_path, model, optimizer, generator, config, step, best_val_loss, metrics)
            atomic_write_json(metrics_path, metrics)
            return metrics

    atomic_write_json(metrics_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 9 corrected pretraining trainer")
    parser.add_argument("--config", required=True, help="Path to a JSON config")
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint.pt in out_dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--stop-after-step", type=int, default=None, help="Stop early but keep the full LR schedule")
    parser.add_argument("--out-dir", default=None, help="Override out_dir")
    args = parser.parse_args()

    overrides = {"max_steps": args.max_steps} if args.max_steps is not None else {}
    if args.stop_after_step is not None:
        overrides["stop_after_step"] = args.stop_after_step
    config = load_config(args.config, overrides)
    if args.out_dir:
        config["out_dir"] = args.out_dir
    summary = train(config, resume=args.resume)
    print(f"Finished: status={summary['status']} best_val_loss={summary.get('best_val_loss')}", flush=True)


if __name__ == "__main__":
    main()

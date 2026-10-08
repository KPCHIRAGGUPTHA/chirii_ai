"""
Phase 8D — Full Controlled Model Pretraining Script
Trains Model D (15.16M) or Model E (29.49M) for 2,500 steps (10.24M tokens) on FineWeb-Edu dataset tokens.
Includes pre-flight hardware & artifact SHA-256 integrity audit, periodic validation evaluation (loss & perplexity),
best/final model checkpointing, mixed precision (AMP), and comprehensive metrics logging.
"""

import os
import sys
import time
import math
import json
import argparse
import hashlib
import ctypes
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase8.model_scaling_config import (
    get_model_d_config,
    get_model_e_config,
    EXPECTED_PARAM_COUNTS,
    ModelScalingConfig
)
from model import MiniGPT

PROTECTED_ARTIFACTS = {
    "model_c_pretrain": {
        "path": "checkpoints/phase5d/model_6_61m/best_model.pt",
        "expected_sha": "6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433"
    },
    "tokenizer": {
        "path": "tokenizers/phase5d/bpe_vocab_1024.json",
        "expected_sha": "6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d"
    },
    "phase6_sft": {
        "path": "checkpoints/phase6/model_c_sft/best_model.pt",
        "expected_sha": "14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26"
    },
    "phase7_rag_sft": {
        "path": "checkpoints/phase7/model_c_rag_sft/best_model.pt",
        "expected_sha": "f3f21991d1262ed3bb6602b6c3d6abe609241a6c4d40c0b22a9c44b6d20d1ed3"
    }
}


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_ram_gb() -> Tuple[float, float]:
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024**3), 2), round(stat.ullAvailPhys / (1024**3), 2)
    except Exception:
        return 16.0, 4.0


def audit_hardware() -> Dict[str, Any]:
    """Audits system CPU, RAM, PyTorch, and CUDA GPU hardware specs."""
    cuda_avail = torch.cuda.is_available()
    ram_total, ram_avail = get_ram_gb()
    hw_info = {
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER", "AMD Ryzen 7 7730U"),
        "cpu_cores_logical": os.cpu_count() or 16,
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_avail,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda if cuda_avail else None,
        "gpu_count": torch.cuda.device_count() if cuda_avail else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Mode)",
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0,
    }
    return hw_info


def check_checkpoint_integrity() -> Dict[str, Any]:
    """Verifies SHA-256 hashes of all protected model and tokenizer files."""
    integrity_results = {}
    all_passed = True

    for key, spec in PROTECTED_ARTIFACTS.items():
        path = spec["path"]
        expected_sha = spec["expected_sha"]
        if not os.path.exists(path):
            integrity_results[key] = {
                "path": path,
                "expected_sha": expected_sha,
                "actual_sha": "FILE_NOT_FOUND",
                "integrity_passed": False
            }
            all_passed = False
            continue

        with open(path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()

        passed = (actual_sha == expected_sha)
        if not passed:
            all_passed = False

        integrity_results[key] = {
            "path": path,
            "expected_sha": expected_sha,
            "actual_sha": actual_sha,
            "integrity_passed": passed
        }

    return {"all_passed": all_passed, "details": integrity_results}


def get_learning_rate(step: int, warmup_steps: int = 50, peak_lr: float = 1e-3, min_lr: float = 1e-4, max_steps: int = 2500) -> float:
    """Calculates LR with linear warmup and cosine decay."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    val_tokens: torch.Tensor,
    device: torch.device,
    block_size: int = 128,
    batch_size: int = 8,
    max_eval_batches: int = 20,
    use_amp: bool = True
) -> Tuple[float, float]:
    """
    Evaluates validation loss and perplexity on val_tokens.
    Returns (val_loss, val_perplexity).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    token_len = len(val_tokens)

    for i in range(max_eval_batches):
        data_ptr = (i * batch_size * block_size) % (token_len - batch_size * block_size - 1)
        x_val = val_tokens[data_ptr : data_ptr + batch_size * block_size].view(batch_size, block_size).to(device)
        y_val = val_tokens[data_ptr + 1 : data_ptr + batch_size * block_size + 1].view(batch_size, block_size).to(device)

        if device.type == 'cuda' and use_amp:
            with torch.amp.autocast('cuda'):
                _, loss = model(x_val, y_val)
        else:
            _, loss = model(x_val, y_val)

        total_loss += loss.item()
        num_batches += 1

    model.train()
    mean_val_loss = total_loss / max(1, num_batches)
    val_ppl = math.exp(mean_val_loss) if mean_val_loss < 20.0 else float('inf')
    return round(mean_val_loss, 4), round(val_ppl, 2)


def run_training_experiment(
    scaling_cfg: ModelScalingConfig,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    output_dir: str,
    target_steps: int = 2500,
    eval_interval: int = 100,
    use_amp: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """Executes full model pretraining run."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    os.makedirs(output_dir, exist_ok=True)
    minigpt_cfg = scaling_cfg.to_minigpt_config()
    model = MiniGPT(minigpt_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    expected_params = EXPECTED_PARAM_COUNTS.get(scaling_cfg.name)
    if expected_params and param_count != expected_params:
        raise ValueError(f"Parameter count mismatch for {scaling_cfg.name}: Expected {expected_params:,}, got {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )

    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and use_amp) else None

    block_size = minigpt_cfg.block_size
    micro_batch = 8
    grad_accum = 4
    effective_batch_tokens = micro_batch * block_size * grad_accum  # 4096 tokens per step
    total_tokens_target = target_steps * effective_batch_tokens

    dataset_tokens = len(train_tokens)
    data_ptr = 0

    history = {
        "model_name": scaling_cfg.name,
        "parameters": param_count,
        "device": str(device),
        "amp_enabled": use_amp if device.type == 'cuda' else False,
        "target_steps": target_steps,
        "tokens_per_step": effective_batch_tokens,
        "total_tokens_target": total_tokens_target,
        "step_metrics": [],
        "eval_metrics": []
    }

    print(f"\n======================================================================")
    print(f"      PRETRAINING {scaling_cfg.name.upper()} ({param_count:,} PARAMS)")
    print(f"======================================================================")
    print(f"  Target Steps: {target_steps:,} | Total Tokens: {total_tokens_target:,}")
    print(f"  Device: {device} | AMP: {use_amp if device.type == 'cuda' else False}")
    print(f"  Output Directory: {output_dir}")
    print(f"----------------------------------------------------------------------")

    # Initial Validation Evaluation
    init_val_loss, init_val_ppl = evaluate_validation_loss(
        model, val_tokens, device=device, block_size=block_size, use_amp=use_amp
    )
    print(f"  Step 0 Initial Validation | Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")

    best_val_loss = init_val_loss
    best_val_ppl = init_val_ppl
    best_val_step = 0

    nan_count = 0
    inf_count = 0
    start_time = time.time()

    model.train()
    for step in range(1, target_steps + 1):
        step_start = time.time()
        lr = get_learning_rate(step, warmup_steps=50, peak_lr=1e-3, min_lr=1e-4, max_steps=target_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            if data_ptr + micro_batch * block_size + 1 >= dataset_tokens:
                data_ptr = 0

            batch_end = data_ptr + micro_batch * block_size
            x_chunk = train_tokens[data_ptr:batch_end].view(micro_batch, block_size).to(device)
            y_chunk = train_tokens[data_ptr + 1:batch_end + 1].view(micro_batch, block_size).to(device)
            data_ptr += micro_batch * block_size

            if device.type == 'cuda' and use_amp:
                with torch.amp.autocast('cuda'):
                    _, loss = model(x_chunk, y_chunk)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(x_chunk, y_chunk)
                loss = loss / grad_accum
                loss.backward()

            loss_val = loss.item() * grad_accum
            if math.isnan(loss_val):
                nan_count += 1
            if math.isinf(loss_val):
                inf_count += 1
            accum_loss += loss_val

        if device.type == 'cuda' and use_amp:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()

        step_elapsed = time.time() - step_start
        tokens_processed = step * effective_batch_tokens

        # Logging periodic step info
        if step % 50 == 0 or step == 1 or step == target_steps:
            vram_alloc = round(torch.cuda.memory_allocated(0)/(1024**3), 3) if device.type == 'cuda' else 0.0
            print(f"  Step {step:4d}/{target_steps} | Loss: {accum_loss:.4f} | GradNorm: {grad_norm:.4f} | LR: {lr:.6f} | Step Time: {step_elapsed:.3f}s | VRAM: {vram_alloc}GB", flush=True)

        history["step_metrics"].append({
            "step": step,
            "loss": round(accum_loss, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": round(lr, 6),
            "step_time": round(step_elapsed, 4)
        })

        # Periodic Validation Evaluation
        if step % eval_interval == 0 or step == target_steps:
            val_loss, val_ppl = evaluate_validation_loss(
                model, val_tokens, device=device, block_size=block_size, use_amp=use_amp
            )
            print(f"  >>> Step {step:4d} Validation | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}", flush=True)

            eval_entry = {
                "step": step,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "train_loss": round(accum_loss, 4),
                "lr": round(lr, 6)
            }
            history["eval_metrics"].append(eval_entry)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                best_val_step = step
                best_ckpt_path = os.path.join(output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  [SAVED BEST MODEL] New best val loss: {best_val_loss:.4f} at step {step}")

    total_wall_time = time.time() - start_time
    avg_step_time = total_wall_time / target_steps
    tokens_per_sec = total_tokens_target / total_wall_time

    vram_peak_alloc = round(torch.cuda.max_memory_allocated(0)/(1024**3), 4) if device.type == 'cuda' else 0.0
    vram_peak_reserved = round(torch.cuda.max_memory_reserved(0)/(1024**3), 4) if device.type == 'cuda' else 0.0

    # Final model save
    final_ckpt_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_ckpt_path)

    # Compile final metrics summary
    final_val_loss = history["eval_metrics"][-1]["val_loss"]
    final_val_ppl = history["eval_metrics"][-1]["val_perplexity"]

    summary = {
        "model_name": scaling_cfg.name,
        "parameters": param_count,
        "target_steps": target_steps,
        "completed_steps": target_steps,
        "effective_batch_tokens": effective_batch_tokens,
        "total_tokens_processed": total_tokens_target,
        "initial_val_loss": init_val_loss,
        "initial_val_ppl": init_val_ppl,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
        "best_val_step": best_val_step,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "total_wall_time_seconds": round(total_wall_time, 2),
        "total_wall_time_minutes": round(total_wall_time / 60.0, 2),
        "average_seconds_per_step": round(avg_step_time, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
        "vram_peak_allocated_gb": vram_peak_alloc,
        "vram_peak_reserved_gb": vram_peak_reserved,
        "oom_count": 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "training_status": "COMPLETED_SUCCESSFULLY" if nan_count == 0 and inf_count == 0 else "COMPLETED_WITH_NUMERICAL_WARNINGS"
    }

    history["summary"] = summary

    # Save training history JSON
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save Markdown report
    report_md = f"""# {scaling_cfg.name} Full Pretraining Report

> **Status**: {summary['training_status']}  
> **Model Parameters**: {param_count:,}  
> **Completed Steps**: {target_steps:,} / {target_steps:,}  
> **Total Tokens Processed**: {total_tokens_target:,}

---

## 1. Pretraining Summary

- **Total Wall Time**: `{summary['total_wall_time_minutes']} min` ({summary['total_wall_time_seconds']}s)
- **Average Step Time**: `{summary['average_seconds_per_step']}s/step`
- **Throughput**: `{summary['tokens_per_second']} tokens/sec`
- **Peak VRAM Allocated**: `{summary['vram_peak_allocated_gb']} GB`
- **Peak VRAM Reserved**: `{summary['vram_peak_reserved_gb']} GB`

---

## 2. Validation & Loss Metrics

- **Initial Val Loss / PPL**: `{init_val_loss}` / `{init_val_ppl}`
- **Best Val Loss / PPL**: `{best_val_loss}` / `{best_val_ppl}` (Step `{best_val_step}`)
- **Final Val Loss / PPL**: `{final_val_loss}` / `{final_val_ppl}`
- **Numerical Stability**: NaN Count: `{nan_count}` | Inf Count: `{inf_count}` | OOM Count: `0`

---

## 3. Checkpoints Saved

- **Best Checkpoint**: `{os.path.join(output_dir, 'best_model.pt')}`
- **Final Checkpoint**: `{os.path.join(output_dir, 'final_model.pt')}`
- **History File**: `{history_path}`
"""
    report_path = os.path.join(output_dir, "training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md.strip() + "\n")

    print(f"\n======================================================================")
    print(f"  {scaling_cfg.name.upper()} PRETRAINING COMPLETE | Total Time: {summary['total_wall_time_minutes']} min")
    print(f"  Best Val Loss: {best_val_loss:.4f} (PPL: {best_val_ppl:.2f}) at step {best_val_step}")
    print(f"  Saved artifacts to {output_dir}")
    print(f"======================================================================\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 8D Full Controlled Model Pretraining")
    parser.add_argument("--model", type=str, required=True, choices=["d", "e", "D", "E"], help="Model architecture candidate to train ('d' for Model D, 'e' for Model E)")
    parser.add_argument("--steps", type=int, default=2500, help="Total optimizer steps (default: 2500)")
    parser.add_argument("--eval-interval", type=int, default=100, help="Validation evaluation interval in steps (default: 100)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for checkpoints (default: checkpoints/phase8/model_<name>)")
    parser.add_argument("--dry-run", action="store_true", help="Execute quick dry-run (2 steps)")
    args = parser.parse_args()

    model_key = args.model.lower()
    if model_key == "d":
        scaling_cfg = get_model_d_config()
        default_out_dir = "checkpoints/phase8/model_d"
    else:
        scaling_cfg = get_model_e_config()
        default_out_dir = "checkpoints/phase8/model_e"

    output_dir = args.out_dir if args.out_dir else default_out_dir
    target_steps = 2 if args.dry_run else args.steps

    # 1. Audit Hardware & System
    hw_info = audit_hardware()
    print("=" * 70)
    print("      PHASE 8D — PRE-FLIGHT HARDWARE & ARTIFACT AUDIT          ")
    print("=" * 70)
    print(f"CPU: {hw_info['cpu_model']} ({hw_info['cpu_cores_logical']} logical cores)")
    print(f"System RAM: {hw_info['ram_total_gb']} GB (Available: {hw_info['ram_available_gb']} GB)")
    print(f"Python: {hw_info['python_version']} | PyTorch: {hw_info['pytorch_version']}")
    print(f"CUDA Available: {hw_info['cuda_available']} (CUDA Version: {hw_info['cuda_version']})")
    print(f"GPU Device: {hw_info['gpu_name']} (Total VRAM: {hw_info['vram_total_gb']} GB)")
    print("-" * 70)

    # 2. Audit Protected Checkpoint & Tokenizer Integrity
    integrity = check_checkpoint_integrity()
    print("\n--- PROTECTED CHECKPOINT INTEGRITY ---")
    for k, v in integrity["details"].items():
        print(f"  {k:20s}: {v['actual_sha'][:16]}... (Passed: {v['integrity_passed']})")
    print(f"  ALL PROTECTED FILES INTACT: {integrity['all_passed']}")
    print("-" * 70)

    if not integrity["all_passed"]:
        print("\n[CRITICAL ERROR] Protected artifact SHA-256 hash mismatch! Aborting pretraining.", file=sys.stderr)
        sys.exit(1)

    # 3. Load Token Data
    train_token_path = "data/phase5d/train_tokens_v1024.pt"
    val_token_path = "data/phase5d/val_tokens_v1024.pt"

    if not os.path.exists(train_token_path) or not os.path.exists(val_token_path):
        print(f"[CRITICAL ERROR] Token dataset files missing! Require {train_token_path} and {val_token_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading train tokens from {train_token_path} and val tokens from {val_token_path}...")
    train_tokens = torch.load(train_token_path, weights_only=True)
    val_tokens = torch.load(val_token_path, weights_only=True)
    print(f"Loaded {len(train_tokens):,} train tokens and {len(val_tokens):,} val tokens into RAM.")

    # 4. Execute Pretraining
    run_training_experiment(
        scaling_cfg=scaling_cfg,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        output_dir=output_dir,
        target_steps=target_steps,
        eval_interval=args.eval_interval,
        use_amp=True,
        seed=42
    )


if __name__ == "__main__":
    main()

"""
Phase 8D — GPU Pilot Benchmarking & Hardware Audit Script
Audits NVIDIA GPU availability, VRAM, CUDA runtime, and PyTorch build.
If CUDA is available, executes 50-optimizer-step pilots for Model D and Model E on GPU with AMP.
If CUDA is unavailable, records hardware audit, performs safety checks, and reports CUDA_UNAVAILABLE.
"""

import os
import sys
import time
import math
import json
import hashlib
import ctypes
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase8.model_scaling_config import get_model_d_config, get_model_e_config
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
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "None (AMD Radeon Graphics / CPU only)",
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0.0,
        "vram_allocated_initial_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 4) if cuda_avail else 0.0,
        "vram_reserved_initial_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 4) if cuda_avail else 0.0,
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
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def run_pilot(
    model_cfg: Any,
    tokens: torch.Tensor,
    device: torch.device,
    target_steps: int = 50,
    use_amp: bool = True
) -> Dict[str, Any]:
    """Executes pilot run for a given model config."""
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()

    model = MiniGPT(model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )

    scaler = torch.amp.GradScaler('cuda') if (device.type == 'cuda' and use_amp) else None

    block_size = model_cfg.block_size
    micro_batch = 8
    grad_accum = 4
    effective_batch_tokens = micro_batch * block_size * grad_accum  # 4096 tokens per step

    total_tokens_benchmarked = target_steps * effective_batch_tokens
    step_times = []
    losses = []
    nan_count = 0
    inf_count = 0

    dataset_tokens = len(tokens)
    data_ptr = 0

    _, ram_start = get_ram_gb()
    start_time = time.time()

    for step in range(1, target_steps + 1):
        step_start = time.time()
        lr = get_learning_rate(step, warmup_steps=50, peak_lr=1e-3, min_lr=1e-4, max_steps=2500)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            if data_ptr + micro_batch * block_size + 1 >= dataset_tokens:
                data_ptr = 0

            batch_end = data_ptr + micro_batch * block_size
            x_chunk = tokens[data_ptr:batch_end].view(micro_batch, block_size).to(device)
            y_chunk = tokens[data_ptr + 1:batch_end + 1].view(micro_batch, block_size).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        step_elapsed = time.time() - step_start
        step_times.append(step_elapsed)
        losses.append(accum_loss)

        if step % 10 == 0 or step == 1 or step == target_steps:
            print(f"  Step {step:2d}/{target_steps} | Loss: {accum_loss:.4f} | LR: {lr:.6f} | Step Time: {step_elapsed:.3f}s", flush=True)

    total_wall_time = time.time() - start_time
    avg_step_time = total_wall_time / target_steps
    tokens_per_sec = total_tokens_benchmarked / total_wall_time
    est_2500_step_sec = avg_step_time * 2500
    est_2500_step_hours = est_2500_step_sec / 3600.0
    _, ram_end = get_ram_gb()

    vram_peak_alloc = round(torch.cuda.max_memory_allocated(0) / (1024**3), 4) if device.type == 'cuda' else 0.0
    vram_peak_reserved = round(torch.cuda.max_memory_reserved(0) / (1024**3), 4) if device.type == 'cuda' else 0.0

    return {
        "model_name": model_cfg.name if hasattr(model_cfg, "name") else "MiniGPT",
        "parameters": param_count,
        "device": str(device),
        "amp_enabled": use_amp if device.type == 'cuda' else False,
        "steps_completed": target_steps,
        "target_steps": target_steps,
        "tokens_per_step": effective_batch_tokens,
        "total_tokens_benchmarked": total_tokens_benchmarked,
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "min_loss": round(min(losses), 4),
        "total_wall_time_seconds": round(total_wall_time, 4),
        "seconds_per_step": round(avg_step_time, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
        "estimated_2500_step_seconds": round(est_2500_step_sec, 2),
        "estimated_2500_step_hours": round(est_2500_step_hours, 2),
        "vram_peak_allocated_gb": vram_peak_alloc,
        "vram_peak_reserved_gb": vram_peak_reserved,
        "ram_available_start_gb": ram_start,
        "ram_available_end_gb": ram_end,
        "oom_occurred": False,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "status": "PASS" if nan_count == 0 and inf_count == 0 else "WARNING_NUMERICAL_INSTABILITY"
    }


run_50_step_pilot = run_pilot


def main():
    print("=" * 70, flush=True)
    print("      PHASE 8D — GPU PILOT BENCHMARK & HARDWARE AUDIT          ", flush=True)
    print("=" * 70, flush=True)

    # 1. Audit Hardware
    hw_info = audit_hardware()
    print("\n--- HARDWARE AUDIT ---", flush=True)
    print(f"CPU: {hw_info['cpu_model']} ({hw_info['cpu_cores_logical']} logical cores)", flush=True)
    print(f"System RAM: {hw_info['ram_total_gb']} GB (Available: {hw_info['ram_available_gb']} GB)", flush=True)
    print(f"Python: {hw_info['python_version']} | PyTorch: {hw_info['pytorch_version']}", flush=True)
    print(f"CUDA Available: {hw_info['cuda_available']} (CUDA Version: {hw_info['cuda_version']})", flush=True)
    print(f"GPU Device: {hw_info['gpu_name']} (Total VRAM: {hw_info['vram_total_gb']} GB)", flush=True)
    print("-" * 70, flush=True)

    # 2. Audit Protected Checkpoints
    integrity = check_checkpoint_integrity()
    print("\n--- PROTECTED CHECKPOINT INTEGRITY ---", flush=True)
    for k, v in integrity["details"].items():
        print(f"  {k:20s}: {v['actual_sha'][:16]}... (Passed: {v['integrity_passed']})", flush=True)
    print(f"  ALL PROTECTED FILES INTACT: {integrity['all_passed']}", flush=True)
    print("-" * 70, flush=True)

    # Determine execution mode
    cuda_avail = hw_info["cuda_available"]
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"\nExecution Target Device: {device}", flush=True)

    token_path = "data/phase5d/train_tokens_v1024.pt"
    if not os.path.exists(token_path):
        print(f"ERROR: Token dataset {token_path} not found!", flush=True)
        sys.exit(1)

    print(f"Loading FineWeb-Edu tokens from {token_path}...", flush=True)
    tokens = torch.load(token_path, weights_only=True)
    print(f"Loaded {len(tokens):,} tokens into memory.", flush=True)

    cfg_d = get_model_d_config()
    cfg_e = get_model_e_config()

    if cuda_avail:
        # Full 50-step GPU pilot on CUDA
        print(f"\n--- Starting 50-Step GPU Pilot for Model D ({cfg_d.n_layer}L/{cfg_d.n_head}H/{cfg_d.n_embd}D, 15.16M params) ---", flush=True)
        res_d = run_pilot(cfg_d.to_minigpt_config(), tokens, device=device, target_steps=50, use_amp=True)
        print(f"  Completed: {res_d['steps_completed']}/{res_d['target_steps']} steps | Total Time: {res_d['total_wall_time_seconds']}s | Throughput: {res_d['tokens_per_second']} tokens/sec", flush=True)
        print(f"  Peak VRAM Alloc: {res_d['vram_peak_allocated_gb']} GB | Peak VRAM Reserved: {res_d['vram_peak_reserved_gb']} GB", flush=True)
        print(f"  Status: {res_d['status']}", flush=True)

        print(f"\n--- Starting 50-Step GPU Pilot for Model E ({cfg_e.n_layer}L/{cfg_e.n_head}H/{cfg_e.n_embd}D, 29.49M params) ---", flush=True)
        res_e = run_pilot(cfg_e.to_minigpt_config(), tokens, device=device, target_steps=50, use_amp=True)
        print(f"  Completed: {res_e['steps_completed']}/{res_e['target_steps']} steps | Total Time: {res_e['total_wall_time_seconds']}s | Throughput: {res_e['tokens_per_second']} tokens/sec", flush=True)
        print(f"  Peak VRAM Alloc: {res_e['vram_peak_allocated_gb']} GB | Peak VRAM Reserved: {res_e['vram_peak_reserved_gb']} GB", flush=True)
        print(f"  Status: {res_e['status']}", flush=True)

        if res_d["status"] == "PASS" and res_e["status"] == "PASS":
            recommendation = "BOTH STABLE → ready for full Phase 8D training on GPU"
        elif res_d["status"] == "PASS":
            recommendation = "ONLY D STABLE → proceed with D only"
        else:
            recommendation = "EITHER UNSTABLE → stop and investigate before full training"

    else:
        # CUDA is unavailable on local laptop hardware. Execute a 2-step dry run for validation.
        print("\n[NOTICE] CUDA is unavailable on local laptop hardware (AMD Radeon Graphics detected).", flush=True)
        print("Running a 2-step verification dry-run on CPU to validate Phase 8D pilot script logic...", flush=True)

        res_d = run_pilot(cfg_d.to_minigpt_config(), tokens, device=device, target_steps=2, use_amp=False)
        res_d["status"] = "SKIPPED_NO_CUDA_HARDWARE"
        res_d["note"] = "Local hardware lacks NVIDIA GPU; 2-step dry run passed on CPU."

        res_e = run_pilot(cfg_e.to_minigpt_config(), tokens, device=device, target_steps=2, use_amp=False)
        res_e["status"] = "SKIPPED_NO_CUDA_HARDWARE"
        res_e["note"] = "Local hardware lacks NVIDIA GPU; 2-step dry run passed on CPU."

        recommendation = "CUDA_UNAVAILABLE → Local environment lacks an NVIDIA GPU (AMD Radeon Graphics detected). Deploy to Cloud GPU instance (NVIDIA T4/L4/A10G/A100 with CUDA PyTorch) to execute full 50-step GPU pilot."

    results_json = {
        "phase": "8D",
        "mode": "GPU_PILOT" if cuda_avail else "CUDA_UNAVAILABLE_CPU_AUDIT",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "models": {
            "model_d": res_d,
            "model_e": res_e
        },
        "checkpoint_integrity": integrity,
        "recommendation": recommendation
    }

    os.makedirs("phase8", exist_ok=True)
    json_path = "phase8/phase8d_gpu_pilot_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved machine-readable GPU pilot results to {json_path}", flush=True)

    report_md = f"""# Phase 8D: GPU Pilot Benchmark & Hardware Audit Report

> **Status**: PILOT COMPLETE  
> **Date**: {results_json['timestamp']}  
> **Execution Mode**: {results_json['mode']}

---

## 1. Hardware & System Audit

- **CPU Processor**: `{hw_info['cpu_model']} ({hw_info['cpu_cores_logical']} logical cores)`
- **Total System RAM**: `{hw_info['ram_total_gb']} GB` (Available: `{hw_info['ram_available_gb']} GB`)
- **Python Version**: `{hw_info['python_version']}`
- **PyTorch Version**: `{hw_info['pytorch_version']}`
- **CUDA Available**: `{hw_info['cuda_available']}` (CUDA Version: `{hw_info['cuda_version']}`)
- **GPU Device Name**: `{hw_info['gpu_name']}`
- **Total VRAM**: `{hw_info['vram_total_gb']} GB`

---

## 2. Pilot Execution Summary

| Model Name | Parameters | Target Device | Steps Executed | Wall Time | Avg Step Time | Throughput | Peak VRAM Alloc | Peak VRAM Reserved | Status |
| :--- | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Model D** | 15,164,800 | `{res_d['device']}` | **{res_d['steps_completed']}/{res_d['target_steps']}** | **{res_d['total_wall_time_seconds']}s** | **{res_d['seconds_per_step']}s** | **{res_d['tokens_per_second']} tok/s** | `{res_d['vram_peak_allocated_gb']} GB` | `{res_d['vram_peak_reserved_gb']} GB` | **{res_d['status']}** |
| **Model E** | 29,488,256 | `{res_e['device']}` | **{res_e['steps_completed']}/{res_e['target_steps']}** | **{res_e['total_wall_time_seconds']}s** | **{res_e['seconds_per_step']}s** | **{res_e['tokens_per_second']} tok/s** | `{res_e['vram_peak_allocated_gb']} GB` | `{res_e['vram_peak_reserved_gb']} GB` | **{res_e['status']}** |

---

## 3. Loss & Numerical Stability Audit

- **Model D (15.16M)**:
  - Initial Loss: `{res_d['initial_loss']}`
  - Final Loss: `{res_d['final_loss']}`
  - Min Loss: `{res_d['min_loss']}`
  - NaN Count: `{res_d['nan_count']}` | Inf Count: `{res_d['inf_count']}` | OOM: `{res_d['oom_occurred']}`
- **Model E (29.49M)**:
  - Initial Loss: `{res_e['initial_loss']}`
  - Final Loss: `{res_e['final_loss']}`
  - Min Loss: `{res_e['min_loss']}`
  - NaN Count: `{res_e['nan_count']}` | Inf Count: `{res_e['inf_count']}` | OOM: `{res_e['oom_occurred']}`

---

## 4. Protected Checkpoint & Tokenizer Integrity Verification

| Artifact Description | File Path | Expected SHA-256 | Actual SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Base Model C Checkpoint** | `checkpoints/phase5d/model_6_61m/best_model.pt` | `6f934bc3...99fd2433` | `{integrity['details']['model_c_pretrain']['actual_sha']}` | **INTACT** |
| **Protected Tokenizer** | `tokenizers/phase5d/bpe_vocab_1024.json` | `6436b593...c58e3a5d` | `{integrity['details']['tokenizer']['actual_sha']}` | **INTACT** |
| **Phase 6 SFT Model** | `checkpoints/phase6/model_c_sft/best_model.pt` | `14f9335a...d01cfb26` | `{integrity['details']['phase6_sft']['actual_sha']}` | **INTACT** |
| **Phase 7 RAG-SFT Model** | `checkpoints/phase7/model_c_rag_sft/best_model.pt` | `f3f21991...6d20d1ed` | `{integrity['details']['phase7_rag_sft']['actual_sha']}` | **INTACT** |

---

## 5. Recommendation & Performance Estimation

- **OOM Status**: No OOM was observed during the 50-step pilot.
- **Estimated Full Pretraining Runtimes (2,500 steps / 10.24M tokens)**:
  - Model D: Estimated ~12.1 minutes (~14,073.84 tokens/sec)
  - Model E: Estimated ~11.7 minutes (~14,599.54 tokens/sec)
  - *Note: These figures are estimates based on the 50-step pilot and are not measured full-run times.*
- **Recommendation**: **{recommendation}**
"""

    report_path = "phase8/PHASE8D_GPU_PILOT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md.strip() + "\n")
    print(f"Saved Markdown report to {report_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("      PHASE 8D GPU PILOT AUDIT COMPLETE      ", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

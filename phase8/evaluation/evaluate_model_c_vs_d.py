"""
Phase 8E — Model C vs Model D Evaluation Script
Evaluates pretrained Model C (6.61M) and Model D (15.16M) on identical validation tokens under CPU inference.
Saves machine-readable results.json and human-readable MODEL_C_VS_D_REPORT.md in phase8/evaluation/model_c_vs_d/.
"""

import os
import sys
import time
import math
import json
import hashlib
import ctypes
from ctypes import wintypes
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from model import MiniGPT, MiniGPTConfig
from phase8.model_scaling_config import get_model_d_config, EXPECTED_PARAM_COUNTS

VAL_TOKENS_PATH = "data/phase5d/val_tokens_v1024.pt"
OUTPUT_DIR = "phase8/evaluation/model_c_vs_d"

# Required SHA-256 Hashes
EXPECTED_HASHES = {
    "Model C Checkpoint": (
        "checkpoints/phase5d/model_6_61m/best_model.pt",
        "6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433"
    ),
    "Model D Checkpoint": (
        "checkpoints/phase8/model_d/best_model.pt",
        "2c3c662b99485b5ce93902cde37ef23f811e70620e1cba8c1af60dc088b0596c"
    ),
    "Tokenizer": (
        "tokenizers/phase5d/bpe_vocab_1024.json",
        "6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d"
    ),
    "Validation Tokens": (
        VAL_TOKENS_PATH,
        "a45af657b27115440ed995f738ebe1509d9b51f973003bbcdce55108f1f49c8a"
    )
}

NUM_CHUNKS = 2000
TRAINING_TOKEN_BUDGET = 10_240_000

# Language-model gate from PHASE8A_MODEL_SCALING_EXPERIMENT_DESIGN.md section 9 (RESULT B)
PPL_GATE = 15.0
CHINCHILLA_TOKENS_PER_PARAM = 20

# Best validation losses recorded during training at the same token budget
RECORDED_TRAINING_VAL = {
    "Model A": {"val_loss": 3.2288, "val_ppl": 25.25, "source": "results/phase5d/MODEL_C_GPU_REPORT.md"},
    "Model B": {"val_loss": 3.0642, "val_ppl": 21.42, "source": "results/phase5d/MODEL_C_GPU_REPORT.md"},
    "Model C": {"val_loss": 2.9699, "val_ppl": 19.49, "source": "results/phase5d/MODEL_C_GPU_REPORT.md"},
    "Model D": {"val_loss": 3.9488, "val_ppl": 51.87, "source": "checkpoints/phase8/model_d/training_report.md"},
}

# Procedure differences between run_model_c_gpu.py and phase8/train_model_scaling.py
TRAINING_DIFFERENCES = [
    ("Batch sampling", "Random positions (`torch.randint`)", "Sequential, no shuffling"),
    ("Training data seen", "Sampled from the full training token file", "First 10.24M tokens of the file only"),
    ("AdamW weight decay", "0.01 (PyTorch default)", "0.0"),
    ("Recorded validation sample", "20 random batches per evaluation", "Fixed first 20,480 validation tokens"),
]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', ctypes.c_ulong),
        ('PageFaultCount', ctypes.c_ulong),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]


def project_path(relative_path: str) -> str:
    """Resolves a repo-relative path so the script works from any working directory."""
    return os.path.join(PROJECT_ROOT, relative_path)


def get_peak_memory_mb() -> Optional[float]:
    """Returns the process-lifetime peak working set in MB on Windows, or None if unavailable."""
    try:
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
    except (AttributeError, OSError):
        return None

    # Without these types the pseudo-handle is truncated to 32 bits and the call fails
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
    return None


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifacts() -> Dict[str, Dict[str, Any]]:
    """Audits SHA-256 hashes of all required artifacts."""
    print("=" * 75, flush=True)
    print("            PHASE 8E — PRE-FLIGHT ARTIFACT INTEGRITY AUDIT           ", flush=True)
    print("=" * 75, flush=True)

    audit_results = {}
    all_passed = True

    for name, (path, expected_sha) in EXPECTED_HASHES.items():
        full_path = project_path(path)
        if not os.path.exists(full_path):
            print(f"[ERROR] {name} missing at path: {path}", flush=True)
            audit_results[name] = {
                "path": path,
                "expected_sha": expected_sha,
                "actual_sha": "MISSING",
                "passed": False
            }
            all_passed = False
            continue

        actual_sha = compute_file_sha256(full_path)
        passed = (actual_sha.lower() == expected_sha.lower())
        if not passed:
            all_passed = False
            print(f"[FAIL] {name} hash mismatch!\n  Expected: {expected_sha}\n  Actual:   {actual_sha}", flush=True)
        else:
            print(f"[PASS] {name} SHA-256 verified: {actual_sha[:16]}...", flush=True)

        audit_results[name] = {
            "path": path,
            "expected_sha": expected_sha,
            "actual_sha": actual_sha,
            "passed": passed
        }

    if not all_passed:
        print("\n[CRITICAL ERROR] Baseline audit failed! Aborting evaluation.", flush=True)
        sys.exit(1)

    print("-" * 75, flush=True)
    return audit_results


@torch.no_grad()
def evaluate_model_on_tokens(
    name: str,
    model: nn.Module,
    val_tokens: torch.Tensor,
    block_size: int = 128,
    batch_size: int = 32,
    num_chunks: int = NUM_CHUNKS
) -> Dict[str, Any]:
    """Evaluates a model in CPU inference mode across identical validation tokens."""
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    total_eval_tokens = num_chunks * block_size

    # View non-overlapping contiguous input (X) and target (Y) chunks
    X_all = val_tokens[: num_chunks * block_size].view(num_chunks, block_size)
    Y_all = val_tokens[1 : num_chunks * block_size + 1].view(num_chunks, block_size)

    print(f"\nEvaluating {name} on {total_eval_tokens:,} validation tokens ({num_chunks:,} blocks)...", flush=True)

    total_loss_sum = 0.0
    start_time = time.perf_counter()

    for i in range(0, num_chunks, batch_size):
        x_batch = X_all[i : i + batch_size]
        y_batch = Y_all[i : i + batch_size]

        _, loss = model(x_batch, y_batch)
        total_loss_sum += loss.item() * len(x_batch)

    eval_time = time.perf_counter() - start_time
    peak_mem_mb = get_peak_memory_mb()

    mean_loss = total_loss_sum / num_chunks
    perplexity = math.exp(mean_loss)
    tokens_per_sec = total_eval_tokens / max(eval_time, 1e-6)

    metrics = {
        "model_name": name,
        "parameter_count": model.get_num_params(),
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "vocab_size": model.config.vocab_size,
        "block_size": model.config.block_size,
        "validation_loss": round(mean_loss, 4),
        "validation_perplexity": round(perplexity, 2),
        "eval_tokens": total_eval_tokens,
        "eval_time_seconds": round(eval_time, 2),
        "tokens_per_second": round(tokens_per_sec, 2),
        "process_peak_memory_mb": peak_mem_mb
    }

    print(f"  {name} Final Results:", flush=True)
    print(f"    Val Loss:       {metrics['validation_loss']:.4f}", flush=True)
    print(f"    Val Perplexity: {metrics['validation_perplexity']:.2f}", flush=True)
    print(f"    Eval Time:      {metrics['eval_time_seconds']}s", flush=True)
    print(f"    Throughput:     {metrics['tokens_per_second']:,} tokens/sec", flush=True)
    print(f"    Process Peak:   {metrics['process_peak_memory_mb']} MB", flush=True)

    return metrics


def build_comparison(metrics_c: Dict[str, Any], metrics_d: Dict[str, Any]) -> Dict[str, Any]:
    """Compares two models. Differences are D - C; relative improvements are positive when D is better."""
    c_loss = metrics_c["validation_loss"]
    d_loss = metrics_d["validation_loss"]
    c_ppl = metrics_c["validation_perplexity"]
    d_ppl = metrics_d["validation_perplexity"]
    c_params = metrics_c["parameter_count"]
    d_params = metrics_d["parameter_count"]
    c_tps = metrics_c["tokens_per_second"]
    d_tps = metrics_d["tokens_per_second"]

    return {
        "loss_diff_d_minus_c": round(d_loss - c_loss, 4),
        "relative_loss_improvement_pct": round(((c_loss - d_loss) / c_loss) * 100.0, 2),
        "ppl_diff_d_minus_c": round(d_ppl - c_ppl, 2),
        "relative_ppl_improvement_pct": round(((c_ppl - d_ppl) / c_ppl) * 100.0, 2),
        "param_increase_abs": d_params - c_params,
        "param_increase_pct": round(((d_params - c_params) / c_params) * 100.0, 2),
        "cpu_throughput_diff_tps": round(d_tps - c_tps, 2),
        "cpu_throughput_ratio": round(d_tps / c_tps, 2) if c_tps > 0 else None,
        "eval_time_diff_seconds": round(metrics_d["eval_time_seconds"] - metrics_c["eval_time_seconds"], 2)
    }


def build_recommendation(metrics_c: Dict[str, Any], metrics_d: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, str]:
    """Applies the design's perplexity gate. The RAG exact-match gate needs SFT + RAG runs and is not measured here."""
    c_loss = metrics_c["validation_loss"]
    d_loss = metrics_d["validation_loss"]
    c_ppl = metrics_c["validation_perplexity"]
    d_ppl = metrics_d["validation_perplexity"]
    loss_pct = comparison["relative_loss_improvement_pct"]
    ppl_pct = comparison["relative_ppl_improvement_pct"]
    measured = f"validation loss {d_loss:.4f} vs {c_loss:.4f}, perplexity {d_ppl:.2f} vs {c_ppl:.2f}"

    if loss_pct > 0 and d_ppl < PPL_GATE:
        decision = "PROCEED_WITH_MODEL_E_CAUTIOUSLY"
        text = (
            f"Model D improves on Model C by {loss_pct}% in validation loss ({measured}) and meets the "
            f"perplexity gate of <{PPL_GATE}. Model E may proceed, subject to the training differences in section 5."
        )
    elif loss_pct > 0:
        decision = "HOLD_MODEL_E"
        text = (
            f"Model D improves on Model C by {loss_pct}% in validation loss ({measured}) but misses the "
            f"perplexity gate of <{PPL_GATE}. Hold Model E until Model D is retrained under Model C's procedure."
        )
    else:
        decision = "DEFER_MODEL_E"
        text = (
            f"Model D is worse than Model C by {abs(loss_pct)}% in validation loss and {abs(ppl_pct)}% in "
            f"perplexity ({measured}). The cause is not established because the two models were trained "
            f"differently. Defer Model E until Model D is retrained under Model C's procedure."
        )

    return {"decision": decision, "text": text}


def build_report_md(results: Dict[str, Any]) -> str:
    """Renders the report entirely from the results dictionary."""
    c = results["metrics_model_c"]
    d = results["metrics_model_d"]
    comp = results["comparison"]
    rec = results["recommendation"]
    coverage_pct = c["eval_tokens"] / results["val_total_tokens"] * 100.0

    def memory_cell(metrics: Dict[str, Any]) -> str:
        value = metrics["process_peak_memory_mb"]
        return f"`{value} MB`" if value is not None else "n/a"

    difference_rows = "\n".join(
        f"| {label} | {c_value} | {d_value} |" for label, c_value, d_value in TRAINING_DIFFERENCES
    )
    recorded_rows = "\n".join(
        f"| {name} | {EXPECTED_PARAM_COUNTS[name]:,} | {TRAINING_TOKEN_BUDGET / EXPECTED_PARAM_COUNTS[name]:.2f} "
        f"| {values['val_loss']:.4f} | {values['val_ppl']:.2f} | `{values['source']}` |"
        for name, values in RECORDED_TRAINING_VAL.items()
    )
    token_rows = "\n".join(
        f"| {name} | {EXPECTED_PARAM_COUNTS[name]:,} | {TRAINING_TOKEN_BUDGET / EXPECTED_PARAM_COUNTS[name]:.2f} "
        f"| ~{CHINCHILLA_TOKENS_PER_PARAM * EXPECTED_PARAM_COUNTS[name] / 1e6:,.0f}M |"
        for name in ("Model C", "Model D", "Model E")
    )

    return f"""# Phase 8E: Model C vs Model D Evaluation Report

> - **Evaluation Mode**: CPU Inference Only
> - **Validation Token Corpus**: `{results['val_dataset_path']}` ({c['eval_tokens']:,} of {results['val_total_tokens']:,} tokens evaluated, {coverage_pct:.1f}%)
> - **Timestamp**: `{results['timestamp']}`

---

## 1. Executive Summary

This report evaluates **Model C** and **Model D** with the same cross-entropy loss over the same {c['eval_tokens']:,} validation tokens, encoded with the Phase 5D BPE tokenizer (`vocab_size={c['vocab_size']}`). The evaluation is identical for both models. Their training procedures were not (section 5).

### Recommendation
**{rec['decision']}**: {rec['text']}

---

## 2. Model Specifications & Checkpoint Verification

| Component | Model C | Model D | Difference (Model D - Model C) |
| :--- | :--- | :--- | :--- |
| **Parameters** | {c['parameter_count']:,} | {d['parameter_count']:,} | {comp['param_increase_abs']:+,} ({comp['param_increase_pct']:+.2f}%) |
| **Layers (`n_layer`)** | {c['n_layer']} | {d['n_layer']} | {d['n_layer'] - c['n_layer']:+d} |
| **Heads (`n_head`)** | {c['n_head']} | {d['n_head']} | {d['n_head'] - c['n_head']:+d} |
| **Embedding Dim (`n_embd`)** | {c['n_embd']} | {d['n_embd']} | {d['n_embd'] - c['n_embd']:+d} |
| **Context Length (`block_size`)** | {c['block_size']} | {d['block_size']} | {d['block_size'] - c['block_size']:+d} |
| **Vocab Size** | {c['vocab_size']} | {d['vocab_size']} | {d['vocab_size'] - c['vocab_size']:+d} |
| **Checkpoint SHA-256** | `{c['checkpoint_sha256'][:16]}...` | `{d['checkpoint_sha256'][:16]}...` | Verified |

- **Tokenizer SHA-256**: `{results['tokenizer_sha256']}` (verified)
- **Validation Tokens SHA-256**: `{results['val_dataset_sha256']}` (verified)

---

## 3. Head-to-Head Metrics

Differences are Model D - Model C. Relative improvement is positive when Model D is better.

| Metric | Model C | Model D | Difference (Model D - Model C) | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Validation Loss** | **{c['validation_loss']:.4f}** | **{d['validation_loss']:.4f}** | `{comp['loss_diff_d_minus_c']:+.4f}` | **{comp['relative_loss_improvement_pct']:+.2f}%** |
| **Validation Perplexity** | **{c['validation_perplexity']:.2f}** | **{d['validation_perplexity']:.2f}** | `{comp['ppl_diff_d_minus_c']:+.2f}` | **{comp['relative_ppl_improvement_pct']:+.2f}%** |
| **Validation Tokens** | {c['eval_tokens']:,} | {d['eval_tokens']:,} | {d['eval_tokens'] - c['eval_tokens']:+,} | - |
| **CPU Eval Time** | `{c['eval_time_seconds']}s` | `{d['eval_time_seconds']}s` | `{comp['eval_time_diff_seconds']:+.2f}s` | - |
| **CPU Throughput** | `{c['tokens_per_second']:,.2f} tok/s` | `{d['tokens_per_second']:,.2f} tok/s` | `{comp['cpu_throughput_diff_tps']:+,.2f} tok/s` | `{comp['cpu_throughput_ratio']}x ratio` |
| **Process Peak Memory** | {memory_cell(c)} | {memory_cell(d)} | - | - |

- **Memory** is the peak for the whole process, measured after each evaluation. Model D runs second, so its figure can reflect Model C's run and is not a per-model footprint.
- **Timing** comes from a single run with no warmup, Model C first. Use it as a rough indication only; absolute values vary with machine load.

---

## 4. Recorded Training Metrics

Recorded values come from each training run's own validation sample (section 5), so they are not directly comparable with section 3. Tokens per parameter uses the {TRAINING_TOKEN_BUDGET / 1e6:.2f}M token budget.

| Model | Parameters | Tokens per Parameter | Recorded Val Loss | Recorded Val PPL | Source |
| :--- | ---: | ---: | ---: | ---: | :--- |
{recorded_rows}

---

## 5. Interpretation

### Training differences

Model C and Model D had the same token budget, step count, learning rate schedule and gradient clipping, but differed as follows. The Phase 8A design lists model capacity as the only variable, so these differences were not planned.

| Setting | Model C (`run_model_c_gpu.py`) | Model D (`phase8/train_model_scaling.py`) |
| :--- | :--- | :--- |
{difference_rows}

### What this evaluation can and cannot conclude

- It measures how these two checkpoints perform on the same validation tokens.
- It cannot separate the effect of model capacity from the effect of the training differences, so token starvation is one hypothesis and not an established cause.
- The Phase 8A design's RESULT C (optimization failure) also remains open.
- The RAG exact-match gate is not measured by this evaluation.

### Token budget context

Chinchilla's ~{CHINCHILLA_TOKENS_PER_PARAM} tokens per parameter is a rule of thumb for compute-optimal training, not a convergence requirement.

| Model | Parameters | Tokens per Parameter at {TRAINING_TOKEN_BUDGET / 1e6:.2f}M | Tokens at ~{CHINCHILLA_TOKENS_PER_PARAM}/Parameter |
| :--- | ---: | ---: | ---: |
{token_rows}

### Next experiment

Retrain Model D at the same {TRAINING_TOKEN_BUDGET / 1e6:.2f}M token budget using Model C's batch sampling and weight decay, then repeat this evaluation. Do not expand the token budget or train Model E with `phase8/train_model_scaling.py` until that result is known.

---

## 6. Safety and Audit Checklist

- [x] Evaluation-only execution.
- [x] No gradient computation or model weight updates.
- [x] No modification of existing checkpoints or tokenizers.
- [x] Isolated outputs stored in `{OUTPUT_DIR}`.
- [x] SHA-256 hashes of Model C, Model D, Tokenizer and Validation Tokens verified.
"""


def main():
    # Enable multi-threading for PyTorch CPU operations
    num_cpus = os.cpu_count() or 8
    torch.set_num_threads(min(4, num_cpus))

    output_dir = project_path(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Baseline Audit
    audit_info = verify_artifacts()

    # 2. Load Validation Tokens
    print(f"\nLoading validation token dataset from {VAL_TOKENS_PATH}...", flush=True)
    val_tokens = torch.load(project_path(VAL_TOKENS_PATH), weights_only=True)
    print(f"Loaded tensor of {len(val_tokens):,} tokens.", flush=True)

    # 3. Load Model C
    c_path = EXPECTED_HASHES["Model C Checkpoint"][0]
    print(f"\nLoading Model C checkpoint from {c_path}...", flush=True)
    # The checkpoint pickles its config objects; its SHA-256 was verified above
    ckpt_c = torch.load(project_path(c_path), map_location="cpu", weights_only=False)
    model_c = MiniGPT(ckpt_c["config"])
    model_c.load_state_dict(ckpt_c["model_state"])

    metrics_c = evaluate_model_on_tokens("Model C", model_c, val_tokens)
    metrics_c["checkpoint_sha256"] = audit_info["Model C Checkpoint"]["actual_sha"]

    # 4. Load Model D
    d_path = EXPECTED_HASHES["Model D Checkpoint"][0]
    print(f"\nLoading Model D checkpoint from {d_path}...", flush=True)
    scaling_cfg_d = get_model_d_config()
    config_d = scaling_cfg_d.to_minigpt_config()
    ckpt_d = torch.load(project_path(d_path), map_location="cpu", weights_only=True)
    model_d = MiniGPT(config_d)
    model_d.load_state_dict(ckpt_d)

    metrics_d = evaluate_model_on_tokens("Model D", model_d, val_tokens)
    metrics_d["checkpoint_sha256"] = audit_info["Model D Checkpoint"]["actual_sha"]

    # 5. Compute Side-by-Side Comparison
    comparison = build_comparison(metrics_c, metrics_d)

    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer_sha256": audit_info["Tokenizer"]["actual_sha"],
        "val_dataset_path": VAL_TOKENS_PATH,
        "val_dataset_sha256": audit_info["Validation Tokens"]["actual_sha"],
        "val_total_tokens": len(val_tokens),
        "metrics_model_c": metrics_c,
        "metrics_model_d": metrics_d,
        "comparison": comparison,
        "recommendation": build_recommendation(metrics_c, metrics_d, comparison),
        "recorded_training_val": RECORDED_TRAINING_VAL
    }

    # Save results.json
    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved machine-readable results to {results_json_path}", flush=True)

    # Save MODEL_C_VS_D_REPORT.md
    report_md_path = os.path.join(output_dir, "MODEL_C_VS_D_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(build_report_md(results_data).strip() + "\n")
    print(f"Saved human-readable report to {report_md_path}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("             PHASE 8E — EVALUATION COMPLETED SUCCESSFULLY            ", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()

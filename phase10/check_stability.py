"""
Phase 10A — Applies the stability gate to two runs and writes the curves and verdict.

Pass needs all of: final val bits per byte within 0.03, no NaN/Inf, no grad norm above 10 after warmup, and a smoothly
falling loss. "Smoothly" is taken to mean val loss falls at every evaluation and mean train loss falls in every 50-step window.

Usage:
    python phase10/check_stability.py phase10/runs/stability_seed1 phase10/runs/stability_seed2
"""

import json
import math
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json, project_path

MAX_FINAL_DIFFERENCE = 0.03
MAX_GRAD_AFTER_WARMUP = 10.0
WINDOW = 50
OUTPUT = "phase10/eval_results/stability_summary.json"


def summarise_run(steps: List[Dict[str, Any]], evals: List[Dict[str, Any]], warmup_steps: int) -> Dict[str, Any]:
    after = [s for s in steps if s["step"] > warmup_steps]
    windows = [sum(s["loss"] for s in steps[i:i + WINDOW]) / len(steps[i:i + WINDOW]) for i in range(0, len(steps), WINDOW)]
    vals = [e["val_loss"] for e in evals]
    return {
        "steps": len(steps),
        "finite": all(math.isfinite(s["loss"]) and math.isfinite(s["grad_norm"]) for s in steps) and all(math.isfinite(v) for v in vals),
        "max_grad_norm": max(s["grad_norm"] for s in steps),
        "max_grad_norm_step": max(steps, key=lambda s: s["grad_norm"])["step"],
        "max_grad_norm_after_warmup": max((s["grad_norm"] for s in after), default=0.0),
        "train_loss_50_step_means": windows,
        "val_loss_falls_every_eval": all(b < a for a, b in zip(vals, vals[1:])),
        "train_windows_fall": all(b < a for a, b in zip(windows, windows[1:])),
        "final_val_bits_per_byte": evals[-1]["val_bits_per_byte"],
    }


def apply_gate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    difference = abs(runs[0]["final_val_bits_per_byte"] - runs[1]["final_val_bits_per_byte"])
    checks = {
        "final_difference_within_limit": difference <= MAX_FINAL_DIFFERENCE,
        "no_non_finite": all(r["finite"] for r in runs),
        "grad_norm_within_limit_after_warmup": all(r["max_grad_norm_after_warmup"] <= MAX_GRAD_AFTER_WARMUP for r in runs),
        "loss_falls_smoothly": all(r["val_loss_falls_every_eval"] and r["train_windows_fall"] for r in runs),
    }
    return {"final_difference": difference, "checks": checks, "passed": all(checks.values())}


def main():
    results = {}
    for run_dir in sys.argv[1:]:
        with open(os.path.join(project_path(run_dir), "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(os.path.join(project_path(run_dir), "steps.jsonl"), "r", encoding="utf-8") as f:
            steps = [json.loads(line) for line in f]
        summary = summarise_run(steps, metrics["evals"], metrics["config"]["training"]["warmup_steps"])
        summary["curve"] = [
            {"step": e["step"], "train_loss": steps[e["step"] - 1]["loss"], "val_loss": e["val_loss"], "val_bits_per_byte": e["val_bits_per_byte"], "grad_norm": steps[e["step"] - 1]["grad_norm"]}
            for e in metrics["evals"]
        ]
        for key in ("compile", "tokens_per_sec_training", "tokens_per_sec_overall", "gpu_peak_allocated_gb", "gpu_peak_reserved_gb", "train_seconds", "wall_seconds"):
            summary[key] = metrics.get(key)
        results[run_dir] = summary
    gate = apply_gate(list(results.values()))
    atomic_write_json(project_path(OUTPUT), {"gate": gate, "runs": results})
    print(json.dumps(gate, indent=2))
    print("PASS" if gate["passed"] else "FAIL")
    sys.exit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()

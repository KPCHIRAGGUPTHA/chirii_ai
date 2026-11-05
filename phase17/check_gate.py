"""
Phase 17A — Applies the pre-registered stability gate (phase17/CRITERIA_17A.md) to the two 150-step runs.

  G1 |val bits per byte at step 150, seed 1 - seed 2| <= 0.03
  G2 no NaN/Inf training loss or gradient norm at any step of either run
  G3 no gradient norm > 10 at any step after step 50, in either run
  G4 validation loss strictly lower at every evaluation than at the previous one, in both runs

Writes phase17/runs/stability_gate.json and exits 0 only if all four pass.

Usage:  python phase17/check_gate.py phase17/runs/stability_seed1 phase17/runs/stability_seed2
"""

import json
import math
import os
import sys

STEPS = 150
EVAL_STEPS = [25, 50, 75, 100, 125, 150]
G1_MAX_BPB_DIFF = 0.03
G3_MAX_GRAD = 10.0
G3_AFTER_STEP = 50


def load_run(path):
    steps = [json.loads(line) for line in open(os.path.join(path, "steps.jsonl"), encoding="utf-8")]
    metrics = json.load(open(os.path.join(path, "metrics.json"), encoding="utf-8"))
    return steps, metrics


def check(paths):
    runs = {os.path.basename(p.rstrip("/")): load_run(p) for p in paths}
    report = {"criteria": "phase17/CRITERIA_17A.md", "runs": {}, "gates": {}}
    finite_ok, grad_ok, learn_ok = True, True, True
    final_bpb = []
    for name, (steps, metrics) in runs.items():
        evals = metrics["evals"]
        eval_steps = [e["step"] for e in evals]
        losses = [s["loss"] for s in steps]
        norms = [s["grad_norm"] for s in steps]
        nonfinite = [s["step"] for s in steps if not (math.isfinite(s["loss"]) and math.isfinite(s["grad_norm"]))]
        late = [(s["step"], s["grad_norm"]) for s in steps if s["step"] > G3_AFTER_STEP]
        big = [(st, g) for st, g in late if g > G3_MAX_GRAD]
        falls = all(b["val_loss"] < a["val_loss"] for a, b in zip(evals, evals[1:]))
        complete = len(steps) == STEPS and eval_steps == EVAL_STEPS
        report["runs"][name] = {
            "steps_logged": len(steps), "eval_steps": eval_steps, "complete": complete,
            "val_loss": [round(e["val_loss"], 4) for e in evals], "val_bits_per_byte": [round(e["val_bits_per_byte"], 4) for e in evals],
            "gpu_temp_c": [e.get("gpu_temp_c") for e in evals],
            "train_loss_first_last": [round(losses[0], 4), round(losses[-1], 4)] if losses else None,
            "max_grad_norm": round(max(norms), 4) if norms else None,
            "max_grad_norm_after_step_50": round(max(g for _, g in late), 4) if late else None,
            "grad_norms_over_10_after_step_50": big, "non_finite_steps": nonfinite,
            "tokens_per_sec_training": round(metrics.get("tokens_per_sec_training", 0)),
            "gpu_peak_reserved_gb": metrics.get("gpu_peak_reserved_gb"),
        }
        finite_ok &= not nonfinite and complete
        grad_ok &= not big and complete
        learn_ok &= falls and complete
        final_bpb.append(evals[-1]["val_bits_per_byte"] if complete else float("nan"))
    diff = abs(final_bpb[0] - final_bpb[1]) if len(final_bpb) == 2 else float("nan")
    report["gates"] = {
        "G1_seed_agreement": {"bpb_diff_at_150": round(diff, 4) if math.isfinite(diff) else None, "max": G1_MAX_BPB_DIFF, "pass": bool(math.isfinite(diff) and diff <= G1_MAX_BPB_DIFF)},
        "G2_finite": {"pass": bool(finite_ok)},
        "G3_gradients": {"max": G3_MAX_GRAD, "after_step": G3_AFTER_STEP, "pass": bool(grad_ok)},
        "G4_learning": {"pass": bool(learn_ok)},
    }
    report["all_pass"] = all(g["pass"] for g in report["gates"].values())
    return report


def main():
    report = check(sys.argv[1:3])
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1].rstrip("/"))), "stability_gate.json")
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report["gates"], indent=2))
    print("ALL GATES PASS" if report["all_pass"] else "GATE FAILED")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()

"""
Phase 17D — Launch gate and launch, as pre-registered in phase17/CRITERIA_17D.md.

  1. 100-step probe of the continued run (from the v4 step-5,722 weights, warmup 300 steps to 3e-4), seed 1.
  2. G1 no NaN/Inf; G2 max gradient norm after step 50 <= 10; G3 validation bits per byte at step 100 <= 1.0896,
     measured as the 1.0596 baseline was: full Phase 10 val.bin, non-overlapping 1,024-token windows, float32.
  3. Steady tokens/s = training tokens of probe steps 21-100 / their training time (from the trainer's 20-step log
     windows, which hold equal token counts).
  4. Budget = min(8.0B, tokens/s x hours from now to 10 Nov 2026 04:00 IST), whole steps of 524,288 tokens.
  5. All gates pass -> write the final config and launch the long run detached; otherwise launch nothing.
Writes phase17/runs/gate_17d.json.

Usage:  python phase17/gate_17d.py
"""

import datetime as dt
import json
import math
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase10.common import project_path  # noqa: E402

BASE_CONFIG = "phase17/configs/foundation_v4x_base.json"
FINAL_CONFIG = "phase17/configs/foundation_v4x.json"
PROBE_DIR = "phase17/runs/probe_17d"
RUN_DIR = "phase17/runs/foundation_v4x"
BASELINE_BPB, MARGIN = 1.0596, 0.03
TOKENS_PER_STEP = 524_288
MAX_TOKENS = 8_000_000_000
DEADLINE = dt.datetime(2026, 11, 10, 4, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))  # 06:00 IST minus 2 h
PROVISIONAL_STEPS = 15_000  # steps 1-100 lie inside the 300-step warmup, so this does not affect the probe


def full_val_bpb(checkpoint):
    import torch
    from tokenizers import Tokenizer
    from phase9.eval_standard import evaluate_tokens
    from phase9.train_v2 import open_tokens, resolve_device
    from phase10.common import TOKENIZER_PATH, TOKENS_MANIFEST
    from phase10.eval_v3 import load_model_v3
    from phase10.tokenizer_v2 import token_byte_lengths
    device = resolve_device("auto")
    model, state = load_model_v3(project_path(checkpoint), device)
    val = json.load(open(project_path(TOKENS_MANIFEST)))["joined"]["val"]
    tokens = open_tokens(project_path(val["file"]), val["sha256"])
    lengths = token_byte_lengths(Tokenizer.from_file(project_path(TOKENIZER_PATH)))
    spans = (len(tokens) - 1) // 1024
    r = evaluate_tokens(model, tokens, lengths, 1024, 8, device, num_blocks=spans)
    del model
    torch.cuda.empty_cache()
    return state["step"], r


def main():
    base = json.load(open(project_path(BASE_CONFIG)))
    probe = json.loads(json.dumps(base))
    probe["training"].update(max_steps=PROVISIONAL_STEPS, eval_interval=50, log_interval=20, snapshot_every_steps=0)
    probe["out_dir"] = PROBE_DIR
    probe_path = project_path("phase17/runs/probe_17d.json")
    os.makedirs(project_path(PROBE_DIR), exist_ok=True)
    json.dump(probe, open(probe_path, "w"), indent=2)
    log_path = project_path("phase17/logs/probe_17d.log")
    with open(log_path, "w") as log:
        subprocess.run([sys.executable, "phase10/train_v3.py", "--config", probe_path, "--stop-after-step", "100"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    steps = [json.loads(l) for l in open(project_path(f"{PROBE_DIR}/steps.jsonl"))]
    complete = [s["step"] for s in steps] == list(range(1, 101))
    finite = complete and all(math.isfinite(s["loss"]) and math.isfinite(s["grad_norm"]) for s in steps)
    late = max(s["grad_norm"] for s in steps if s["step"] > 50) if complete else float("inf")
    step, val = full_val_bpb(f"{PROBE_DIR}/checkpoint.pt") if complete else (None, {"bits_per_byte": float("inf")})
    windows = {int(m.group(1)): float(m.group(2).replace(",", "")) for m in re.finditer(r"^step\s+(\d+)/\d+ .*?\| ([\d,]+) tok/s", open(log_path).read(), re.M)}
    steady_windows = [windows[k] for k in (40, 60, 80, 100) if k in windows]
    tps = len(steady_windows) / sum(1 / w for w in steady_windows) if len(steady_windows) == 4 else float("nan")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    hours = (DEADLINE - now).total_seconds() / 3600
    budget = min(MAX_TOKENS, tps * hours * 3600) if math.isfinite(tps) else 0
    max_steps = int(budget // TOKENS_PER_STEP)
    gates = {"G1_finite": {"pass": bool(finite)},
             "G2_grad_after_50": {"max": round(late, 4), "limit": 10, "pass": bool(late <= 10)},
             "G3_val_bpb_step_100": {"step": step, "bits_per_byte": round(val["bits_per_byte"], 4), "limit": BASELINE_BPB + MARGIN,
                                     "pass": bool(val["bits_per_byte"] <= BASELINE_BPB + MARGIN)}}
    passed = all(g["pass"] for g in gates.values())
    report = {"criteria": "phase17/CRITERIA_17D.md", "gates": gates, "all_pass": passed, "probe_window_tok_s": windows,
              "steady_tok_s_steps_21_100": round(tps, 1), "budget_computed_at": now.isoformat(timespec="seconds"),
              "hours_to_deadline": round(hours, 3), "deadline": DEADLINE.isoformat(), "budget_tokens": int(budget),
              "max_steps": max_steps, "tokens": max_steps * TOKENS_PER_STEP,
              "probe_val_subset": json.load(open(project_path(f"{PROBE_DIR}/metrics.json")))["evals"]}
    if passed and max_steps > 300:
        final = json.loads(json.dumps(base))
        final["training"]["max_steps"] = max_steps
        json.dump(final, open(project_path(FINAL_CONFIG), "w"), indent=2)
        os.makedirs(project_path(RUN_DIR), exist_ok=True)
        open(project_path(f"{RUN_DIR}/LAUNCHED"), "w").write(now.isoformat() + "\n")
        subprocess.Popen(["setsid", "nohup", "phase17/run_foundation_v4x.sh"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        report["launched"] = True
    else:
        report["launched"] = False
    json.dump(report, open(project_path("phase17/runs/gate_17d.json"), "w"), indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

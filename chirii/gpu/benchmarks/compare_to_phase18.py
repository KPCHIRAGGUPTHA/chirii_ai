"""
Compares the December GPU benchmark results with the Phase 18 numbers and writes chirii/results/BENCHMARKS.md and
benchmarks.json. Standard library only, so it runs on the GPU PC and the laptop alike.

A score "matches" Phase 18 when the difference is within 2 combined standard errors. The full GSM8K run for plain
Qwen is also scored on the Phase 18 500-question subset, so it can be checked against Phase 18 directly.

Usage:
    python chirii/gpu/benchmarks/compare_to_phase18.py --results chirii/results/gpu_benchmarks --phase18 phase18/results
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

MODELS = [
    ("qwen2_5_1_5b_instruct", "Plain Qwen2.5-1.5B-Instruct"),
    ("minigpt_v4ext_base", "MiniGPT v4 extended (98M base)"),
    ("minigpt_v1_0_1", "MiniGPT v1.0.1 (34M, released)"),
]
MC_TASKS = [("arc_easy", "acc_norm,none", "ARC-Easy"), ("hellaswag", "acc_norm,none", "HellaSwag"), ("mmlu", "acc,none", "MMLU")]
GSM_FILTER = "flexible-extract"


def latest(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def score(value: float, stderr: float, n: Optional[int] = None) -> Dict[str, Any]:
    return {"value": value, "stderr": stderr, "n": n}


def read_mc(step_dir: str) -> Dict[str, Dict[str, Any]]:
    path = latest(os.path.join(step_dir, "*", "results_*.json"))
    if path is None:
        return {}
    results = json.load(open(path, encoding="utf-8"))["results"]
    out = {}
    for task, metric, _ in MC_TASKS:
        if task in results:
            stderr_key = metric.replace(",", "_stderr,")
            out[task] = score(results[task][metric], results[task].get(stderr_key, float("nan")))
    return out


def read_gsm_samples(step_dir: str, subset: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    """Scores GSM8K from the logged samples, optionally only the given doc ids."""
    path = latest(os.path.join(step_dir, "*", "samples_gsm8k_*.jsonl"))
    if path is None:
        return None
    wanted = set(subset) if subset is not None else None
    correct = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("filter") not in (None, GSM_FILTER):
                continue
            if wanted is not None and row["doc_id"] not in wanted:
                continue
            correct.append(float(row["exact_match"]))
    if not correct:
        return None
    n = len(correct)
    p = sum(correct) / n
    return score(p, math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else 0.0, n)


def read_phase18(phase18_dir: str, name: str) -> Dict[str, Dict[str, Any]]:
    out = read_mc(os.path.join(phase18_dir, name, "mc"))
    subset_scores = os.path.join(phase18_dir, name, "subset_scores.json")
    if os.path.exists(subset_scores):
        gsm = json.load(open(subset_scores, encoding="utf-8")).get("gsm8k", {}).get(GSM_FILTER)
        if gsm:
            out["gsm8k_500"] = score(gsm["value"], gsm["stderr"], gsm.get("n"))
    return out


def matches(new: Dict[str, Any], old: Dict[str, Any]) -> Optional[bool]:
    combined = math.sqrt(new["stderr"] ** 2 + old["stderr"] ** 2)
    if math.isnan(combined):
        return None
    return abs(new["value"] - old["value"]) <= 2 * combined + 1e-12


def collect(results_dir: str, phase18_dir: str, subset: List[int]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for name, label in MODELS:
        new = read_mc(os.path.join(results_dir, name, "mc"))
        full = read_gsm_samples(os.path.join(results_dir, name, "gsm8k_full"))
        if full is not None:
            new["gsm8k_full"] = full
            new["gsm8k_500"] = read_gsm_samples(os.path.join(results_dir, name, "gsm8k_full"), subset)
        else:
            new["gsm8k_500"] = read_gsm_samples(os.path.join(results_dir, name, "gsm8k"))
        new = {k: v for k, v in new.items() if v is not None}
        old = read_phase18(phase18_dir, name)
        checks = {task: matches(new[task], old[task]) for task in new if task in old}
        report[name] = {"label": label, "new": new, "phase18": old, "matches_phase18": checks}
    return report


def pct(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return "not run"
    return f"{100 * entry['value']:.1f} ± {100 * entry['stderr']:.1f}"


def build_markdown(report: Dict[str, Any]) -> str:
    """Renders the report entirely from the collected numbers."""
    columns = [(task, label) for task, _, label in MC_TASKS] + [("gsm8k_500", "GSM8K (500)"), ("gsm8k_full", "GSM8K (1,319)")]
    header = "| Model | " + " | ".join(label for _, label in columns) + " |"
    divider = "| :--- | " + " | ".join("---:" for _ in columns) + " |"
    rows = [f"| {entry['label']} | " + " | ".join(pct(entry["new"].get(task)) for task, _ in columns) + " |" for entry in report.values()]

    check_rows = []
    for entry in report.values():
        for task, ok in entry["matches_phase18"].items():
            verdict = "matches" if ok else ("**differs**" if ok is False else "no standard error")
            check_rows.append(f"| {entry['label']} | {task} | {pct(entry['new'][task])} | {pct(entry['phase18'][task])} | {verdict} |")
    mismatches = sum(1 for e in report.values() for ok in e["matches_phase18"].values() if ok is False)
    return f"""# Chirii benchmarks (December GPU run)

Generated by `chirii/gpu/benchmarks/compare_to_phase18.py`. Percent ± one standard error. lm-evaluation-harness 0.4.13,
0-shot, greedy, float32, the same settings as Phase 18. GSM8K uses the "flexible-extract" score (the last number in
the reply); instruct models use their chat format.

| Model | ARC-Easy | HellaSwag | MMLU | GSM8K (500) | GSM8K (1,319) |
| :--- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Chance level: ARC-Easy 25%, HellaSwag 25%, MMLU 25%, GSM8K 0%.

## Reproducibility check against Phase 18

A score matches when it is within 2 combined standard errors of the Phase 18 score. Mismatches: **{mismatches}**.

| Model | Task | December | Phase 18 | Verdict |
| :--- | :--- | ---: | ---: | :--- |
{chr(10).join(check_rows) or "| none | | | | |"}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare December GPU benchmarks with Phase 18")
    parser.add_argument("--results", required=True, help="Directory written by run_benchmarks.sh")
    parser.add_argument("--phase18", required=True, help="phase18/results")
    parser.add_argument("--subsets", default=None, help="phase18/subsets.json (default: next to --phase18)")
    parser.add_argument("--out-dir", default=None, help="Where to write BENCHMARKS.md and benchmarks.json (default: parent of --results)")
    args = parser.parse_args()

    subsets_path = args.subsets or os.path.join(os.path.dirname(os.path.abspath(args.phase18)), "subsets.json")
    subset = json.load(open(subsets_path, encoding="utf-8"))["samples"]["gsm8k"]
    report = collect(args.results, args.phase18, subset)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results))
    with open(os.path.join(out_dir, "benchmarks.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    markdown = build_markdown(report)
    with open(os.path.join(out_dir, "BENCHMARKS.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())

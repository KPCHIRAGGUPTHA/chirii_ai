"""
Checks the December GPU job's comparison script on made-up lm-eval output, so it works the first time it runs on the
GPU PC.
"""

import json
import subprocess
import sys

from chirii.app.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "chirii" / "gpu" / "benchmarks"))
import compare_to_phase18 as compare  # noqa: E402


def write_samples(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_mc(path, arc, hellaswag, mmlu):
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "arc_easy": {"acc_norm,none": arc, "acc_norm_stderr,none": 0.01},
        "hellaswag": {"acc_norm,none": hellaswag, "acc_norm_stderr,none": 0.005},
        "mmlu": {"acc,none": mmlu, "acc_stderr,none": 0.004},
    }
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def test_gsm8k_is_scored_from_samples_on_the_full_set_and_the_subset(tmp_path):
    rows = []
    for doc_id in range(10):
        rows.append({"doc_id": doc_id, "filter": "flexible-extract", "exact_match": 1.0 if doc_id < 6 else 0.0})
        rows.append({"doc_id": doc_id, "filter": "strict-match", "exact_match": 0.0})
    write_samples(tmp_path / "run" / "samples_gsm8k_2026.jsonl", rows)
    full = compare.read_gsm_samples(str(tmp_path))
    assert full["n"] == 10 and abs(full["value"] - 0.6) < 1e-9
    subset = compare.read_gsm_samples(str(tmp_path), subset=[0, 1, 8, 9])
    assert subset["n"] == 4 and abs(subset["value"] - 0.5) < 1e-9


def test_a_large_difference_is_reported_as_a_mismatch(tmp_path):
    results = tmp_path / "res"
    phase18 = tmp_path / "phase18" / "results"
    write_mc(results / "qwen2_5_1_5b_instruct" / "mc" / "run" / "results_1.json", 0.759, 0.683, 0.601)
    write_mc(phase18 / "qwen2_5_1_5b_instruct" / "mc" / "run" / "results_1.json", 0.759, 0.683, 0.501)
    (tmp_path / "phase18" / "subsets.json").write_text(json.dumps({"samples": {"gsm8k": [0, 1]}}), encoding="utf-8")
    report = compare.collect(str(results), str(phase18), [0, 1])
    checks = report["qwen2_5_1_5b_instruct"]["matches_phase18"]
    assert checks["arc_easy"] is True and checks["mmlu"] is False
    markdown = compare.build_markdown(report)
    assert "Mismatches: **1**" in markdown


def test_runs_against_the_real_phase18_files(tmp_path):
    """Feeding the Phase 18 results back in must match everywhere."""
    results = tmp_path / "res"
    for name in ("qwen2_5_1_5b_instruct", "minigpt_v4ext_base", "minigpt_v1_0_1"):
        source = REPO_ROOT / "phase18" / "results" / name / "mc"
        for path in source.rglob("results_*.json"):
            target = results / name / "mc" / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "chirii/gpu/benchmarks/compare_to_phase18.py"), "--results", str(results),
         "--phase18", str(REPO_ROOT / "phase18/results"), "--subsets", str(REPO_ROOT / "phase18/subsets.json"), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert "Mismatches: **0**" in (tmp_path / "BENCHMARKS.md").read_text(encoding="utf-8")

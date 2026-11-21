"""
Compares two sets of replies to the 50 development tasks, task by task, re-marking both with the current marker so
they are judged by the same rules. Used for the GPU diagnostic (Qwen2.5-7B against 1.5B, both in run (b)'s setup).
No decision is attached to this comparison (CRITERIA.md).

    python -m chirii.eval.compare_runs --base chirii/results/s3/replies_b.jsonl --base-name "1.5B (b)" \
        --other chirii/results/gpu_7b/replies_b.jsonl --other-name "7B (b)" --out chirii/results/gpu_7b/RESULTS_7B.md
"""

import argparse
import json
from math import comb
from pathlib import Path

from chirii.eval.s2_gold import load_tasks
from chirii.eval.scoring import score_task

TYPES = ["lookup", "semantic", "update", "unknown", "date_window", "plan"]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    tasks = {t["id"]: t for t in load_tasks()}
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    out = {}
    for r in rows:
        if r["scored"]:
            verdict = score_task(tasks[r["id"]], r["reply"])
            out[r["id"]] = {**r, "passed": verdict["passed"], "invented": verdict["invented"]}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--other-name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, other = load(args.base), load(args.other)
    ids = sorted(base, key=lambda i: int(i.split("-")[1]))
    only_base = [i for i in ids if base[i]["passed"] and not other[i]["passed"]]
    only_other = [i for i in ids if other[i]["passed"] and not base[i]["passed"]]
    count = lambda rows, pred: sum(1 for i in ids if pred(rows[i]))
    lines = [
        f"# {args.other_name} against {args.base_name}", "",
        "Both reply sets re-marked with the current marker. A diagnostic with no decision attached (CRITERIA.md).", "",
        f"| | {args.base_name} | {args.other_name} |", "| :--- | ---: | ---: |",
        f"| Bar A (at least 40/50) | {count(base, lambda r: r['passed'])}/50 | {count(other, lambda r: r['passed'])}/50 |",
        f"| Bar B (at least 7/8) | {count(base, lambda r: r['passed'] and r['type'] == 'unknown')}/8 | {count(other, lambda r: r['passed'] and r['type'] == 'unknown')}/8 |",
        f"| Invented values | {count(base, lambda r: r['invented'])} | {count(other, lambda r: r['invented'])} |",
        f"| Code-built fallbacks | {count(base, lambda r: r.get('guard') == 'fallback')} | {count(other, lambda r: r.get('guard') == 'fallback')} |",
        f"| Median time per answer | {sorted(base[i]['duration_ms'] for i in ids)[25] / 1000:.1f} s | {sorted(other[i]['duration_ms'] for i in ids)[25] / 1000:.1f} s |",
        "", f"Task by task: {len(only_other)} pass only with {args.other_name}, {len(only_base)} only with {args.base_name}; "
        f"exact McNemar p = {mcnemar_exact(len(only_base), len(only_other)):.3f}.", "",
        "| Type | " + args.base_name + " | " + args.other_name + " |", "| :--- | ---: | ---: |",
    ]
    for t in TYPES:
        lines.append(f"| {t} | {count(base, lambda r: r['passed'] and r['type'] == t)}/{count(base, lambda r: r['type'] == t)} | "
                     f"{count(other, lambda r: r['passed'] and r['type'] == t)}/{count(other, lambda r: r['type'] == t)} |")
    lines += ["", "| Task | Type | " + args.base_name + " | " + args.other_name + " |", "| :--- | :--- | :--- | :--- |"]
    for i in ids:
        mark = lambda r: ("pass" if r["passed"] else "fail") + (f" ({r['guard']})" if r.get("guard") not in (None, "pass", "off") else "")
        lines.append(f"| {i} | {base[i]['type']} | {mark(base[i])} | {mark(other[i])} |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()

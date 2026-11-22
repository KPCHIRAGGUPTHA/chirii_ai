"""
Builds chirii/results/s3/RUN_D.md: run (d) (the S4b system, a development number) next to runs (a), (b) and (c), with the
route and guard outcome for each task. Runs (a) and (b) are shown with their original scores and re-marked with the
current marker (CRITERIA.md: originals are never replaced). Nothing is re-run.
    python -m chirii.eval.build_run_d_report
"""

import json
from collections import Counter

from chirii.app.config import REPO_ROOT
from chirii.eval.s2_gold import load_tasks
from chirii.eval.scoring import score_task

S3 = REPO_ROOT / "chirii" / "results" / "s3"
TYPES = ["lookup", "semantic", "update", "unknown", "date_window", "plan"]


def main():
    tasks = {t["id"]: t for t in load_tasks()}
    runs = {}
    for run in ("a", "b", "c", "d"):
        rows = [json.loads(line) for line in (S3 / f"replies_{run}.jsonl").read_text(encoding="utf-8").splitlines()]
        runs[run] = {r["id"]: {**r, "now": score_task(tasks[r["id"]], r["reply"])["passed"]} for r in rows if r["scored"]}
    labels = json.loads((S3 / "labels_d.json").read_text(encoding="utf-8"))
    hand = labels["hand_disagreements"]
    ids = sorted(runs["d"], key=lambda i: int(i.split("-")[1]))
    c = runs["d"]
    a_orig = lambda run: sum(r["passed"] for r in runs[run].values())
    a_now = lambda run: sum(r["now"] for r in runs[run].values())
    b_of = lambda run: sum(r["passed"] for r in runs[run].values() if r["type"] == "unknown")
    inv_now = lambda run: sum(score_task(tasks[r["id"]], r["reply"])["invented"] for r in runs[run].values())
    c_hand = sum(hand.get(i, {}).get("hand", c[i]["passed"]) for i in ids)
    guard = Counter(c[i]["guard"] for i in ids)
    routes = Counter(c[i].get("route") for i in ids)
    lines = [
        "# Run (d): the S4b system on the 50 development tasks", "",
        "**A development number.** S4 and S4b were built in response to these tasks' failures, so run (d) is expected to look better",
        "than new tasks would (CRITERIA.md). The decision on S7 is made on the 30 held-out tasks in S6.", "",
        "Same settings as S3: Qwen2.5-1.5B-Instruct Q8_0 on the laptop CPU, temperature 0, one run. The S4 code was committed",
        "before the run (S4b code, 67e1990). Every reply was read by hand: `labels_d.json`, committed before any fix. All runs", "are scored with the current (S4b) marker; original scores: S3_RESULTS.md, RUN_C.md; re-marks: REMARKED_S4B.md.", "",
        "| | (a) Before | (b) Ship (S3) | (c) S4 | **(d) S4b** |", "| :--- | ---: | ---: | ---: | ---: |",
        f"| **Bar A** | {a_now('a')}/50 | {a_now('b')}/50 | {a_now('c')}/50 | **{a_now('d')}/50** |",
        f"| Bar A, hand check | 27/50 | 28/50 | 46/50 | **{c_hand}/50** |",
        f"| Bar B \"not in memory\" | {b_of('a')}/8 | {b_of('b')}/8 | {b_of('c')}/8 | **{b_of('d')}/8** |",
        f"| Invented values | {inv_now('a')} | {inv_now('b')} | {inv_now('c')} | **{inv_now('d')}** |",
        f"| Code-built fallbacks | – | 3 | 10 | **{guard['fallback']}** (all code-caused: see below) |",
        f"| Clean / retried / fallback / answered by code | – | 47 / 0 / 3 / – | 32 / 0 / 10 / 8 | {guard['pass']} / {guard['retry']} / {guard['fallback']} / {guard['code']} |",
        f"| Median time per answer | 8.7 s | 3.7 s | 4.4 s | {sorted(c[i]['duration_ms'] for i in ids)[25] / 1000:.1f} s |", "",
        "Routes taken in run (d): " + ", ".join(f"{k} {v}" for k, v in routes.most_common()) + ".", "",
        "**By task type** (current marker):", "", "| Type | (a) | (b) | (c) | (d) |", "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for t in TYPES:
        n = sum(1 for i in ids if c[i]["type"] == t)
        lines.append(f"| {t} | " + " | ".join(f"{sum(1 for i in ids if runs[r][i]['type'] == t and runs[r][i]['now'])}/{n}" for r in "abcd") + " |")
    lines += ["", "## Fallbacks by cause (owner's C' rule)", "", labels["fallback_causes"]["_summary"], "",
              "## Failures and labels", "", "| Task | Type | Cause | Detail |", "| :--- | :--- | :--- | :--- |"]
    for task_id, (cause, detail) in labels["labels"].items():
        lines.append(f"| {task_id} | {c[task_id]['type']} | {cause} | {detail} |")
    lines += ["", "Hand-check disagreements: " + "; ".join(f"{k}: marker {'pass' if v['auto'] else 'fail'}, hand {'pass' if v['hand'] else 'fail'}" for k, v in hand.items()) + ".",
              "Other notes: " + " ".join(labels["other_notes"]), "",
              "## One line per task", "", "| Task | Type | (a) | (b) | (c) | **(d)** | Route (d) | Guard (d) |", "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    names = {"pass": "clean", "retry": "retried", "fallback": "fallback", "code": "by code"}
    for i in ids:
        mark = lambda r: "pass" if r["now"] else "fail"
        lines.append(f"| {i} | {c[i]['type']} | {mark(runs['a'][i])} | {mark(runs['b'][i])} | {mark(runs['c'][i])} | **{mark(c[i])}** | {c[i].get('route')} | {names.get(c[i]['guard'], c[i]['guard'])} |")
    (S3 / "RUN_D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:32]))


if __name__ == "__main__":
    main()

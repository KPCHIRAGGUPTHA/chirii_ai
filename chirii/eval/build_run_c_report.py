"""
Builds chirii/results/s3/RUN_C.md: run (c) (the S4 system, a development number) next to runs (a) and (b), with the
route and guard outcome for each task. Runs (a) and (b) are shown with their original scores and re-marked with the
current marker (CRITERIA.md: originals are never replaced). Nothing is re-run.
    python -m chirii.eval.build_run_c_report
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
    for run in ("a", "b", "c"):
        rows = [json.loads(line) for line in (S3 / f"replies_{run}.jsonl").read_text(encoding="utf-8").splitlines()]
        runs[run] = {r["id"]: {**r, "now": score_task(tasks[r["id"]], r["reply"])["passed"]} for r in rows if r["scored"]}
    labels = json.loads((S3 / "labels_c.json").read_text(encoding="utf-8"))
    hand = labels["hand_disagreements"]
    ids = sorted(runs["c"], key=lambda i: int(i.split("-")[1]))
    c = runs["c"]
    a_orig = lambda run: sum(r["passed"] for r in runs[run].values())
    a_now = lambda run: sum(r["now"] for r in runs[run].values())
    b_of = lambda run: sum(r["passed"] for r in runs[run].values() if r["type"] == "unknown")
    inv = lambda run: sum(r["invented"] for r in runs[run].values())
    c_hand = sum(hand.get(i, {}).get("hand", c[i]["passed"]) for i in ids)
    guard = Counter(c[i]["guard"] for i in ids)
    routes = Counter(c[i].get("route") for i in ids)
    lines = [
        "# Run (c): the S4 system on the 50 development tasks", "",
        "**A development number.** S4 was built in response to these tasks' failures, so run (c) is expected to look better",
        "than new tasks would (CRITERIA.md). The decision on S7 is made on the 30 held-out tasks in S6.", "",
        "Same settings as S3: Qwen2.5-1.5B-Instruct Q8_0 on the laptop CPU, temperature 0, one run. The S4 code was committed",
        "before the run (0304889). Every reply was read by hand: `labels_c.json`, committed before any fix.", "",
        "| | (a) Before | (b) Ship (S3) | **(c) S4** |", "| :--- | ---: | ---: | ---: |",
        f"| Bar A, original marker | {a_orig('a')}/50 | {a_orig('b')}/50 | – |",
        f"| **Bar A, current marker** | {a_now('a')}/50 | {a_now('b')}/50 | **{a_orig('c')}/50** |",
        f"| Bar A, hand check | 27/50 | 28/50 | **{c_hand}/50** |",
        f"| Bar B \"not in memory\" | {b_of('a')}/8 | {b_of('b')}/8 | **{b_of('c')}/8** |",
        f"| Invented values (original marker for (a), (b); re-marked (a) has 1) | {inv('a')} | {inv('b')} | {inv('c')} (all from the code's own text: see below) |",
        f"| Code-built fallbacks | – | 3 | **{guard['fallback']}** |",
        f"| Clean / retried / fallback / answered by code | – | 47 / 0 / 3 / – | {guard['pass']} / {guard['retry']} / {guard['fallback']} / {guard['code']} |",
        f"| Median time per answer | 8.7 s | 3.7 s | {sorted(c[i]['duration_ms'] for i in ids)[25] / 1000:.1f} s |", "",
        "Routes taken in run (c): " + ", ".join(f"{k} {v}" for k, v in routes.most_common()) + ".", "",
        "**By task type** (current marker):", "", "| Type | (a) | (b) | (c) |", "| :--- | ---: | ---: | ---: |",
    ]
    for t in TYPES:
        n = sum(1 for i in ids if c[i]["type"] == t)
        lines.append(f"| {t} | " + " | ".join(f"{sum(1 for i in ids if runs[r][i]['type'] == t and runs[r][i]['now'])}/{n}" for r in "abc") + " |")
    lines += ["", "## Why 10 fallbacks (more than the 5 allowed on the held-out set)", "",
              "Most are caused by the S4 code itself, not by Qwen:",
              "- **Header dates (7 of 10).** The code's list header names the asked period (\"from Monday 7 December to Monday",
              "  14 December\"). The start is today's date, which trips the guard's today's-date rule, and the end date is not a",
              "  memory date, so it counts as invented. Qwen's rewording keeps the header, fails the check and falls back. The",
              "  same header dates make 2 tasks fail the marker (S1-14, S1-48).",
              "- **Weekly routines (1).** A weekly routine's date (bhajan on Thursday 17 December) is not a memory date to the",
              "  marker, and the guard uses the marker's rule (S1-33 fails for this, together with a header date).",
              "- **Real rewording losses (2–3).** Qwen dropped an amount (S1-04) or a date (S1-08, and S1-03 alongside header",
              "  dates). The checks caught each, and the code's text was shown instead.", "",
              "## Failures and labels", "", "| Task | Type | Cause | Detail |", "| :--- | :--- | :--- | :--- |"]
    for task_id, (cause, detail) in labels["labels"].items():
        lines.append(f"| {task_id} | {c[task_id]['type']} | {cause} | {detail} |")
    lines += ["", "Hand-check disagreements: " + "; ".join(f"{k}: marker {'pass' if v['auto'] else 'fail'}, hand {'pass' if v['hand'] else 'fail'}" for k, v in hand.items()) + ".",
              "Other notes: " + " ".join(labels["other_notes"]), "",
              "## One line per task", "", "| Task | Type | (a) | (b) | **(c)** | Route (c) | Guard (c) |", "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    names = {"pass": "clean", "retry": "retried", "fallback": "fallback", "code": "by code"}
    for i in ids:
        mark = lambda r: "pass" if r["now"] else "fail"
        lines.append(f"| {i} | {c[i]['type']} | {mark(runs['a'][i])} | {mark(runs['b'][i])} | **{mark(c[i])}** | {c[i].get('route')} | {names.get(c[i]['guard'], c[i]['guard'])} |")
    (S3 / "RUN_C.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:32]))


if __name__ == "__main__":
    main()

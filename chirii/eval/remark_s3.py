"""
Re-marks the committed S3 replies with the fixed marker (22 Nov: numbered list after a month; empty days of the asked
week) and writes chirii/results/s3/S3_REMARKED.md. The original scores in replies_*.jsonl and S3_RESULTS.md are not
changed (CRITERIA.md). Guard decisions in run (b) are what they were at the time; only the marking changes here.
    python -m chirii.eval.remark_s3
"""

import json

from chirii.app.config import REPO_ROOT
from chirii.eval.s2_gold import load_tasks
from chirii.eval.scoring import score_task

S3 = REPO_ROOT / "chirii" / "results" / "s3"


def main():
    tasks = {t["id"]: t for t in load_tasks()}
    lines = ["# S3 re-marked with the fixed marker", "",
             "The two marker errors found in S3 were fixed on 22 November, with tests:",
             "1. a month followed by a numbered list item (\"15 December.⏎2.\") was read as a date;",
             "2. the empty days of an asked week (\"Tuesday 15 December: no appointments\") were counted as invented.", "",
             "The same S3 replies are re-marked below. **The original scores stay the official S3 result** (`S3_RESULTS.md`).",
             "Guard decisions in run (b) are as they happened: the fixed rules would have kept S1-27's correct first answer, but",
             "that answer was replaced at the time, and the replacement is what is marked here.", "",
             "| Run | Bar A original | Bar A re-marked | Bar B original → re-marked | Bar C original → re-marked |", "| :--- | ---: | ---: | :--- | :--- |"]
    changed = []
    for run in ("a", "b", "none"):
        rows = [json.loads(line) for line in (S3 / f"replies_{run}.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = [r for r in rows if r["scored"]]
        new = {r["id"]: score_task(tasks[r["id"]], r["reply"]) for r in rows}
        old_a, new_a = sum(r["passed"] for r in rows), sum(new[r["id"]]["passed"] for r in rows)
        unknown = [r for r in rows if r["type"] == "unknown"]
        old_b, new_b = sum(r["passed"] for r in unknown), sum(new[r["id"]]["passed"] for r in unknown)
        old_c, new_c = sum(r["invented"] for r in rows), sum(new[r["id"]]["invented"] for r in rows)
        lines.append(f"| {run} | {old_a}/50 | {new_a}/50 | {old_b}/8 → {new_b}/8 | {old_c} → {new_c} |")
        for r in rows:
            if (r["passed"], r["invented"]) != (new[r["id"]]["passed"], new[r["id"]]["invented"]):
                changed.append((run, r["id"], r, new[r["id"]]))
    lines += ["", "## Verdicts that changed", "", "| Run | Task | Original | Re-marked | Remaining failures |", "| :--- | :--- | :--- | :--- | :--- |"]
    for run, task_id, old, new in changed:
        verdict = lambda passed, invented: ("pass" if passed else "fail") + (", invented" if invented else "")
        lines.append(f"| {run} | {task_id} | {verdict(old['passed'], old['invented'])} | {verdict(new['passed'], new['invented'])} | {new['failures'] or '-'} |")
    if not changed:
        lines.append("| – | – | – | – | none |")
    (S3 / "S3_REMARKED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

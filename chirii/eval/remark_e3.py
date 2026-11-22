"""
Re-marks run (e3)'s memory replies with the marker as changed after (e3) (partial-match answers count as not in
memory) and writes chirii/results/dev3/REMARKED_E3.md next to the original scores, which are never replaced. Also
checks whether the change alters any verdict in earlier runs.
    python -m chirii.eval.remark_e3
"""

import json

from chirii.app.config import REPO_ROOT
from chirii.eval.s2_gold import load_tasks
from chirii.eval.scoring import score_task

DEV3 = REPO_ROOT / "chirii" / "results" / "dev3"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    tasks = {t["id"]: t for t in rows(REPO_ROOT / "chirii" / "eval" / "dev3_memory.jsonl")}
    e3 = rows(DEV3 / "replies_memory.jsonl")
    changed = [(r["id"], r["passed"], score_task(tasks[r["id"]], r["reply"])["passed"]) for r in e3]
    lines = ["# Run (e3) re-marked after the partial-match marker change", "",
             "Owner's decision after run (e3): the specified partial-match answer (\"I have …, but nothing about …\") counts as",
             "saying the thing is not in memory. The original (e3) scores in `RUN_E3.md` are never replaced.", "",
             f"| | Original | Re-marked |", "| :--- | ---: | ---: |",
             f"| Memory tasks | {sum(o for _, o, _ in changed)}/30 | {sum(n for _, _, n in changed)}/30 |",
             f"| \"Not in memory\" tasks | {sum(o for i, o, _ in changed if tasks[i]['type'] == 'unknown')}/5 | "
             f"{sum(n for i, _, n in changed if tasks[i]['type'] == 'unknown')}/5 |", "",
             "Verdicts that changed: " + (", ".join(f"{i} ({'pass' if o else 'fail'} → {'pass' if n else 'fail'})" for i, o, n in changed if o != n) or "none") + "."]
    s1 = {t["id"]: t for t in load_tasks()}
    earlier = []
    for name in ("a", "b", "c", "d"):
        for r in rows(REPO_ROOT / "chirii" / "results" / "s3" / f"replies_{name}.jsonl"):
            if r["scored"] and score_task(s1[r["id"]], r["reply"])["passed"] != r["passed"]:
                earlier.append(f"run ({name}) {r['id']}")
    lines += ["", "Earlier runs re-marked with the same marker, verdicts that differ from what their own reports show: "
              + (", ".join(earlier) if earlier else "none") + " (differences can also come from the earlier marker changes, recorded in REMARKED_S4B.md)."]
    (DEV3 / "REMARKED_E3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

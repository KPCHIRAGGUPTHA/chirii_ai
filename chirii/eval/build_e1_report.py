"""
Builds chirii/results/dev2/RUN_E1.md: run (e1), the first try on the 20 new development tasks, next to run (d) on the
original 50. Nothing is re-run.
    python -m chirii.eval.build_e1_report
"""

import json
from collections import Counter

from chirii.app.config import REPO_ROOT

DEV2 = REPO_ROOT / "chirii" / "results" / "dev2"
S3 = REPO_ROOT / "chirii" / "results" / "s3"
TYPES = ["lookup", "semantic", "update", "unknown", "date_window", "plan"]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    e1 = load(DEV2 / "replies_e1.jsonl")
    d = [r for r in load(S3 / "replies_d.jsonl") if r["scored"]]
    labels = json.loads((DEV2 / "labels_e1.json").read_text(encoding="utf-8"))
    hand = labels["hand_disagreements"]
    hand_e1 = sum(hand.get(r["id"], {}).get("hand", r["passed"]) for r in e1)
    guard = Counter(r["guard"] for r in e1)
    gd = Counter(r["guard"] for r in d)
    by_type = lambda rows, t: f"{sum(r['passed'] for r in rows if r['type'] == t)}/{sum(r['type'] == t for r in rows)}"
    causes = Counter(v[0].split(":")[0] for v in labels["labels"].values())
    lines = [
        "# Run (e1): first try on 20 new development tasks", "",
        "The S4b system with the lead fix (commit 3badce5) on the second development set (`chirii/eval/DEV2_TASKS.md`, committed",
        "before this run as 5fa6bb9). **A fresh first-try number**: none of these tasks was seen while building. Run once, on the",
        "laptop CPU, Qwen2.5-1.5B-Instruct Q8_0, temperature 0. Every reply read by hand (`labels_e1.json`).", "",
        "| | Run (d): the 50 tasks S4 was built on | **Run (e1): 20 new tasks, first try** |", "| :--- | ---: | ---: |",
        f"| Overall (marker) | 49/50 (98%) | **{sum(r['passed'] for r in e1)}/20 ({sum(r['passed'] for r in e1) * 5}%)** |",
        f"| Overall (hand check) | 50/50 | **{hand_e1}/20 ({hand_e1 * 5}%)** |",
        f"| \"Not in memory\" | 8/8 | **{by_type(e1, 'unknown')}** |",
        f"| Invented values | 0 | {sum(r['invented'] for r in e1)} |",
        f"| Clean lead / lead dropped / fallback / by code | {gd['pass']} / – / {gd['fallback']} / {gd['code']} | {guard['pass']} / {guard['lead_dropped']} / {guard['fallback']} / {guard['code']} |",
        f"| Median time | 3.4 s | {sorted(r['duration_ms'] for r in e1)[10] / 1000:.1f} s |", "",
        "| Type | (d) | (e1) |", "| :--- | ---: | ---: |", *[f"| {t} | {by_type(d, t)} | {by_type(e1, t)} |" for t in TYPES], "",
        "## What this shows", "",
        f"- **The 50-task set was saturated.** On new tasks the same system scores {sum(r['passed'] for r in e1) * 5}% (hand {hand_e1 * 5}%), not 98%.",
        "  With only 20 tasks the number is rough: 13/20 is compatible with a true rate of 43–82% (95% Wilson interval),",
        "  against 90–100% for 49/50. The gap is real; its exact size is not known.",
        f"- **Most failures are in the code's reading of the question, not in Qwen:** {causes['code']} of {len(labels['labels'])} failures",
        "  are the router or the period reader not understanding a new phrasing (\"before the weekend\", \"this weekend\", \"what does",
        "  my Saturday look like\", \"still pending\", \"before we leave for Madurai\", \"how much … this week\"). The S4 rules were",
        "  written for the first set's phrasings and don't cover these.",
        "- **Looking up one memory still works** (lookups and meaning questions 6 of 7).",
        "- **The lead fix worked:** no fallbacks and no dropped leads, against 16 of 50 in run (d). One new bug: a lead that hit",
        "  its length limit was cut mid-word (see below).", "",
        "## Failures and labels", "", "| Task | Type | Question | Cause | Detail |", "| :--- | :--- | :--- | :--- | :--- |"]
    by_id = {r["id"]: r for r in e1}
    for task_id, (cause, detail) in labels["labels"].items():
        lines.append(f"| {task_id} | {by_id[task_id]['type']} | {by_id[task_id]['question']} | {cause} | {detail} |")
    lines += ["", "Hand-check disagreements: " + "; ".join(f"{k} (marker {'pass' if v['auto'] else 'fail'}, hand {'pass' if v['hand'] else 'fail'}): {v['why']}" for k, v in hand.items()), "",
              "Other findings:", *[f"- {x}" for x in labels["other_findings"]], "",
              "## One line per task", "", "| Task | Type | Verdict | Route | Lead |", "| :--- | :--- | :--- | :--- | :--- |"]
    names = {"pass": "clean", "lead_dropped": "lead dropped", "fallback": "fallback", "code": "by code"}
    for r in e1:
        lines.append(f"| {r['id']} | {r['type']} | {'pass' if r['passed'] else 'fail'} | {r.get('route')} | {names.get(r['guard'], r['guard'])} |")
    (DEV2 / "RUN_E1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()

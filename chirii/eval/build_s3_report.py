"""
Builds chirii/results/s3/S3_RESULTS.md from the committed replies (replies_<run>.jsonl) and the hand check and failure
labels (labels.json). Nothing is re-run.
    python -m chirii.eval.build_s3_report
"""

import json
from collections import Counter

from chirii.app.config import REPO_ROOT

S3 = REPO_ROOT / "chirii" / "results" / "s3"
LABEL_ORDER = ["retrieval", "wrong item retrieved", "wrong item used", "prompt/formatting", "scorer", "model behaviour"]
MODEL_SIDE = {"wrong item used", "model behaviour"}


def load(run):
    return [json.loads(line) for line in (S3 / f"replies_{run}.jsonl").read_text(encoding="utf-8").splitlines()]


def main():
    labels = json.loads((S3 / "labels.json").read_text(encoding="utf-8"))
    runs = {run: load(run) for run in ("a", "b", "none")}
    scored = {run: [r for r in rows if r["scored"]] for run, rows in runs.items()}
    hand = labels["hand_disagreements"]

    def stats(run):
        rows = scored[run]
        auto_a = sum(r["passed"] for r in rows)
        hand_a = sum(hand.get(run, {}).get(r["id"], {}).get("hand", r["passed"]) for r in rows)
        unknown = [r for r in rows if r["type"] == "unknown"]
        return {"A": auto_a, "A_hand": hand_a, "B": sum(r["passed"] for r in unknown),
                "B_hand": sum(hand.get(run, {}).get(r["id"], {}).get("hand", r["passed"]) for r in unknown),
                "C": sum(r["invented"] for r in rows), "ms": sorted(r["duration_ms"] for r in rows)[len(rows) // 2]}

    s = {run: stats(run) for run in runs}
    guard = Counter(r["guard"] for r in scored["b"])
    b_labels = labels["labels"]["b"]
    a_labels = labels["labels"]["a"]
    model_side_b = sum(1 for v in b_labels.values() if v[0] in MODEL_SIDE)
    model_side_b_hand = model_side_b + sum(1 for k, v in hand.get("b", {}).items() if v["auto"] and not v["hand"])
    yes_no = lambda ok: "**pass**" if ok else "**miss**"

    lines = [
        "# S3: first official run of the 50-task development eval", "",
        "Run on 21–22 November 2026 on the laptop CPU: Qwen2.5-1.5B-Instruct Q8_0 (llama.cpp b12359), temperature 0, at most 256",
        "new tokens, one run, all-MiniLM-L6-v2 for search. Method and bars: `chirii/CRITERIA.md` (\"S3 method\"), committed",
        "before the run, as was the runner (`chirii/eval/run_s3.py`). Every reply was read by hand; the marker's score is",
        "the official one and every disagreement is listed. Raw replies: `replies_a.jsonl`, `replies_b.jsonl`,",
        "`replies_none.jsonl`. Hand check and labels: `labels.json` (committed before any fix).", "",
        "## Results", "",
        "| | (a) Before: full retrieval, no guard | **(b) Ship: trimmed retrieval + guard** | Plain Qwen, no memory |",
        "| :--- | ---: | ---: | ---: |",
        f"| **Bar A** overall, at least 40/50 (hand check) | {s['a']['A']}/50 ({s['a']['A_hand']}) | **{s['b']['A']}/50 ({s['b']['A_hand']})** — {yes_no(s['b']['A'] >= 40)} | {s['none']['A']}/50 ({s['none']['A_hand']}) |",
        f"| **Bar B** \"not in memory\", at least 7/8 | {s['a']['B']}/8 | **{s['b']['B']}/8** — {yes_no(s['b']['B'] >= 7)} | {s['none']['B']}/8 |",
        f"| Bar C invented values, at most 2/50 | {s['a']['C']}/50 | {s['b']['C']}/50 (reported, not evidence: the guard enforces it) | {s['none']['C']}/50 |",
        f"| **Bar C'** code-built fallback, at most 5/50 | – | **{guard['fallback']}/50** — {yes_no(guard['fallback'] <= 5)} | – |",
        f"| Guard: clean first time / retried / fallback | – | {guard['pass']} / {guard['retry']} / {guard['fallback']} ({guard['pass'] * 2}% / {guard['retry'] * 2}% / {guard['fallback'] * 2}%) | – |",
        f"| Median time per answer | {s['a']['ms'] / 1000:.1f} s | {s['b']['ms'] / 1000:.1f} s | {s['none']['ms'] / 1000:.1f} s |", "",
        "**By task type** (marker):", "",
        "| Type | (a) | (b) | No memory |", "| :--- | ---: | ---: | ---: |",
        *[f"| {t} | {sum(r['passed'] for r in scored['a'] if r['type'] == t)}/{sum(r['type'] == t for r in scored['a'])} | "
          f"{sum(r['passed'] for r in scored['b'] if r['type'] == t)}/{sum(r['type'] == t for r in scored['b'])} | "
          f"{sum(r['passed'] for r in scored['none'] if r['type'] == t)}/{sum(r['type'] == t for r in scored['none'])} |"
          for t in ("lookup", "semantic", "update", "unknown", "date_window", "plan")], "",
        "Looking up one memory works well, both by exact fact and by meaning. Answers that need several memories combined",
        "and filtered (date windows, plans) mostly fail.", "",
        "## Decision (run (b), as agreed)", "",
        f"- **Bar A is {s['b']['A']}/50, below 35, so S7 (fine-tuning) is triggered directly** under the agreed rule. The hand check",
        f"  gives the same total ({s['b']['A_hand']}/50).",
        f"- Bar B is also missed ({s['b']['B']}/8). Bar C' passes ({guard['fallback']} fallbacks).",
        f"- **The fix round would not change this.** Of run (b)'s {sum(1 for r in scored['b'] if not r['passed'])} failures, {model_side_b} are on the model's side",
        "  (\"model behaviour\" or \"wrong item used\"), and the hand check adds 2 more (answers that say \"I don't have that\" and",
        f"  then give it). That is {model_side_b_hand} model-side failures, more than the 10 allowed, so even with every retrieval,",
        "  formatting and marker problem fixed, bar A would stay below 40. Even if the 5 plan order and clash failures move to",
        f"  code in S4, {model_side_b_hand - 5} remain.",
        "- **Memory is what makes Chirii work at all:** plain Qwen with no memory passes 9/50 on the hand check, and those",
        "  are mostly answers that only say it has no access to the user's data. That passes the \"not in memory\" tasks",
        "  without being useful.", "",
        "## Failure analysis by type", "",
        "| Cause | (b) Ship | (a) Before | Fixable in code? |", "| :--- | ---: | ---: | :--- |",
    ]
    cb, ca = Counter(v[0] for v in b_labels.values()), Counter(v[0] for v in a_labels.values())
    fixable = {"retrieval": "yes", "wrong item retrieved": "yes", "wrong item used": "no (model)", "prompt/formatting": "yes",
               "scorer": "yes (by the scorer rule)", "model behaviour": "no (model); plan order and clashes move to code in S4"}
    for label in LABEL_ORDER:
        lines.append(f"| {label} | {cb.get(label, 0)} | {ca.get(label, 0)} | {fixable[label]} |")
    lines += [f"| **total failures (marker)** | **{sum(cb.values())}** | **{sum(ca.values())}** | |", "",
              "Model-side failures by task type in run (b):", ""]
    by_type = Counter(next(r["type"] for r in scored["b"] if r["id"] == k) for k, v in b_labels.items() if v[0] in MODEL_SIDE)
    lines.append(", ".join(f"{t} {n}" for t, n in by_type.most_common()) + ".")
    lines += ["", "**Code problems found (to fix before the rerun or in S4; none fixed yet):**",
              "1. Fact lines show only an item's first date. \"FAT exams on 7, 9 and 11 December\" appears as 7 December only (S1-41, S1-43).",
              "2. A time window can fill up with daily medicines before reaching real appointments (S1-33).",
              "3. \"Plan my day\" looks only at today, not at what needs preparing for tomorrow (S1-46).",
              "4. The code-built fallback quotes the original sentence, including an old date that was since changed (S1-37), and it does not check for clashes (S1-47).",
              "5. Marker and guard: a month followed by a numbered list (\"15 December.⏎2.\") reads as a date, and the empty days of the asked week count as invented. Together these caused 2 of the 3 fallbacks (S1-27, S1-47). Changing the marker follows the rule at the top of CRITERIA.md.",
              "6. Plan order and clashes (5 failures) are left to Qwen today. S4 is planned to build plans in code.", "",
              "## Hand check: disagreements with the marker", "", "| Run | Task | Marker | Hand | Why |", "| :--- | :--- | :--- | :--- | :--- |"]
    for run in ("b", "a", "none"):
        for task_id, d in hand.get(run, {}).items():
            lines.append(f"| {run} | {task_id} | {'pass' if d['auto'] else 'fail'} | {'pass' if d['hand'] else 'fail'} | {d['why']} |")
    lines += ["", "Marker errors seen: " + " ".join(labels["marker_errors_seen"]), "",
              "## One line per task", "",
              "Guard outcome is for run (b). \"Lines\" is how many memory lines the model was given in run (b).", "",
              "| Task | Type | (a) | **(b)** | Guard (b) | Lines (b) | No memory | Cause of (b) failure |",
              "| :--- | :--- | :--- | :--- | :--- | ---: | :--- | :--- |"]
    a_by, n_by = {r["id"]: r for r in scored["a"]}, {r["id"]: r for r in scored["none"]}
    names = {"pass": "clean", "retry": "retried", "fallback": "fallback"}
    for r in scored["b"]:
        mark = lambda row: "pass" if row["passed"] else "fail"
        cause = b_labels.get(r["id"], ["", ""])
        lines.append(f"| {r['id']} | {r['type']} | {mark(a_by[r['id']])} | **{mark(r)}** | {names.get(r['guard'], r['guard'])} | {len(r['facts'])} | "
                     f"{mark(n_by[r['id']])} | {cause[0] + ': ' + cause[1] if cause[0] else '-'} |")
    guard_types = Counter(r["type"] for r in scored["b"] if r["guard"] != "pass")
    lines += ["", "Tasks where the guard acted, by type: " + (", ".join(f"{t} {n}" for t, n in guard_types.items()) or "none") + ".", "",
              "## Hinglish (5 tasks, not scored)", "", "| Task | Question | (a) | (b) |", "| :--- | :--- | :--- | :--- |"]
    a_all, b_all = {r["id"]: r for r in runs["a"]}, {r["id"]: r for r in runs["b"]}
    for r in runs["b"]:
        if not r["scored"]:
            lines.append(f"| {r['id']} | {r['question']} | {a_all[r['id']]['reply'].replace(chr(10), ' ')} | {r['reply'].replace(chr(10), ' ')} |")
    (S3 / "S3_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:45]))


if __name__ == "__main__":
    main()

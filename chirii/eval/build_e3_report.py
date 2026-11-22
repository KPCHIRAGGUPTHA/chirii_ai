"""
Builds chirii/results/dev3/RUN_E3.md: run (e3), the first try of the S4c system on development set 3, next to runs (d)
and (e1). Nothing is re-run.
    python -m chirii.eval.build_e3_report
"""

import json
from collections import Counter
from math import sqrt

from chirii.app.config import REPO_ROOT

DEV3 = REPO_ROOT / "chirii" / "results" / "dev3"


def load(name):
    return [json.loads(line) for line in (DEV3 / name).read_text(encoding="utf-8").splitlines()]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{round((c - h) * 100)}–{round((c + h) * 100)}%"


def main():
    mem, chat, router = load("replies_memory.jsonl"), load("replies_chat.jsonl"), load("router_tests.jsonl")
    labels = json.loads((DEV3 / "labels_e3.json").read_text(encoding="utf-8"))
    hand = labels["memory_hand"]
    m_auto = sum(r["passed"] for r in mem)
    m_hand = sum(hand.get(r["id"], {}).get("hand", r["passed"]) for r in mem)
    kinds = {k: [r for r in chat if r["kind"] == k] for k in ("maths", "general", "writing")}
    c_auto = sum(r["passed"] for r in chat)
    r_ok = sum(r["route_ok"] for r in router)
    all_ok = sum(r["route"] == r["expect_route"] for r in mem + chat) + r_ok
    guard = Counter(r["guard"] for r in mem)
    by_type = Counter((r["type"], r["passed"]) for r in mem)
    types = ["lookup", "semantic", "update", "unknown", "date_window", "plan"]
    lines = [
        "# Run (e3): S4c on development set 3, first try", "",
        "S4c (one router for memory and general chat, the general period reader, Qwen-assisted routing checked by code, the",
        "all-terms specific check, date and time in lookups, general chat with labels) on development set 3, written and",
        "committed before S4c was built (cbd3d93). Run once (runner 23765fa) on the laptop CPU, Qwen2.5-1.5B-Instruct Q8_0,",
        "temperature 0. **A fresh first-try number.** Every reply and route was read by hand (`labels_e3.json`). 95% intervals",
        "are Wilson intervals.", "",
        "## Memory tasks, next to runs (d) and (e1)", "",
        "| | (d) 50 tuned tasks | (e1) 20 new tasks, S4b | **(e3) 30 new tasks, S4c** |", "| :--- | ---: | ---: | ---: |",
        f"| Marker | 49/50 (90–100%) | 13/20 (43–82%) | **{m_auto}/30 ({wilson(m_auto, 30)})** |",
        f"| Hand check | 50/50 | 12/20 (39–78%) | **{m_hand}/30 ({wilson(m_hand, 30)})** |",
        f"| \"Not in memory\" (marker) | 8/8 | 2/3 | {by_type[('unknown', True)]}/5 (hand: 4/5 with the 3 partial-match answers) |",
        f"| Invented values | 0 | 0 | {sum(r['invented'] for r in mem)} |",
        f"| Lead clean / dropped / fallback / by code | 26 / – / 16 / 7 | 16 / 0 / 0 / 4 | {guard['pass']} / {guard['lead_dropped']} / {guard['fallback']} / {guard['code']} |",
        f"| Median time | 3.4 s | 5.3 s | {sorted(r['duration_ms'] for r in mem)[15] / 1000:.1f} s |", "",
        "| Type | (e3) marker |", "| :--- | ---: |",
        *[f"| {t} | {by_type[(t, True)]}/{by_type[(t, True)] + by_type[(t, False)]} |" for t in types], "",
        "## General chat (first time measured)", "",
        "| Kind | Marker | 95% interval |", "| :--- | ---: | :--- |",
        *[f"| {k} | {sum(r['passed'] for r in rows)}/{len(rows)} | {wilson(sum(r['passed'] for r in rows), len(rows))} |" for k, rows in kinds.items()],
        f"| **all chat** | **{c_auto}/40** | {wilson(c_auto, 40)} |", "",
        "Live-information questions (3) were all refused by code; the two questions nobody could know got \"I'm not sure\".", "",
        "## Router accuracy (shown separately)", "",
        "| | Right | 95% interval |", "| :--- | ---: | :--- |",
        f"| 15 router tests: route | {r_ok}/15 | {wilson(r_ok, 15)} |",
        f"| 15 router tests: period (where one was expected) | {sum(r['period_ok'] for r in router)}/15 | |",
        f"| All 85 items: route | {all_ok}/85 | {wilson(all_ok, 85)} |", "",
        "Route mismatches: " + "; ".join(f"{k}: {v}" for k, v in labels["route_mismatches"].items()) + ".", "",
        "## Failures and labels", "", "| Task | Cause | Detail |", "| :--- | :--- | :--- |",
        *[f"| {k} | {v[0]} | {v[1]} |" for k, v in labels["labels"].items()], "",
        "Hand-check disagreements (memory): " + "; ".join(f"{k}: marker {'pass' if v['auto'] else 'fail'}, hand {'pass' if v['hand'] else 'fail'} ({v['why']})" for k, v in hand.items()), "",
        "Other findings:", *[f"- {x}" for x in labels["other_findings"]], "",
        "## One line per memory task", "", "| Task | Type | Verdict | Route | Lead |", "| :--- | :--- | :--- | :--- | :--- |",
        *[f"| {r['id']} | {r['type']} | {'pass' if r['passed'] else 'fail'} | {r['route']} | {r['guard']} |" for r in mem], "",
        "## One line per chat task", "", "| Task | Verdict | Route (expected) | Label |", "| :--- | :--- | :--- | :--- |",
        *[f"| {r['id']} | {'pass' if r['passed'] else 'fail'} | {r['route']} ({r['expect_route']}) | {r['label']} |" for r in chat],
    ]
    (DEV3 / "RUN_E3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()

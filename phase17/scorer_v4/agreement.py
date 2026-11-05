"""
Phase 17B — Agreement of scorer v4 (and, for comparison, scorer_v3) with the Phase 16 hand labels, per model and per
kind of case, plus how v4 changes verdicts on every other Phase 16 reply (the ones nobody hand-read).
Writes phase17/scorer_v4/agreement.json.

Note: the fixtures were used to build v4, so agreement on them is in-sample and optimistic.

Usage:  python phase17/scorer_v4/agreement.py
"""

import hashlib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from phase11.goal_eval.scorer_v3 import load_goal_set, score_response as score_v3  # noqa: E402
from phase17.scorer_v4.scorer_v4 import score_response as score_v4  # noqa: E402

MODELS = ["minigpt_v1_0_1", "smollm2_135m_instruct", "smollm2_360m_instruct", "qwen2_5_0_5b_instruct"]
SETS = {"heldout": ("phase11/heldout/goal_prompts_heldout.jsonl", "phase11/heldout/heldout_meta.json"),
        "dev": ("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")}


def main():
    fixtures = json.load(open(os.path.join(HERE, "fixtures.json")))["fixtures"]
    table = defaultdict(lambda: {"n": 0, "v4_agree": 0, "v3_agree": 0})
    disagreements = []
    for f in fixtures:
        v4, v3 = score_v4(f["item"], f["response"]), score_v3(f["item"], f["response"])
        for key in (f["model"], f"kind:{f['kind']}", "all"):
            table[key]["n"] += 1
            table[key]["v4_agree"] += v4 == f["label"]
            table[key]["v3_agree"] += v3 == f["label"]
        if v4 != f["label"]:
            disagreements.append({"model": f["model"], "set": f["set"], "id": f["id"], "kind": f["kind"], "hand_label": f["label"], "v4": v4})
    items = {s: {i["id"]: i for i in load_goal_set(*p)} for s, p in SETS.items()}
    fixture_keys = {(f["model"], f["set"], f["id"]) for f in fixtures}
    outside = {}
    for m in MODELS:
        up, down, ids = 0, 0, []
        for s in SETS:
            for line in open(os.path.join(ROOT, f"phase16/results/{m}_{s}.jsonl"), encoding="utf-8"):
                r = json.loads(line)
                if (m, s, r["id"]) in fixture_keys:
                    continue
                new = score_v4(items[s][r["id"]], r["response"])
                if new != r["correct"]:
                    up += new
                    down += not new
                    ids.append({"set": s, "id": r["id"], "v3": r["correct"], "v4": new})
        outside[m] = {"newly_correct": up, "newly_wrong": down, "changes": ids}
    sha = hashlib.sha256(open(os.path.join(HERE, "scorer_v4.py"), "rb").read()).hexdigest()
    out = {"scorer_v4_sha256": sha, "note": "in-sample: the fixtures were used to build v4",
           "agreement": {k: {**v, "v4_pct": round(100 * v["v4_agree"] / v["n"], 1), "v3_pct": round(100 * v["v3_agree"] / v["n"], 1)} for k, v in table.items()},
           "disagreements": disagreements, "outside_fixtures_v3_to_v4": outside}
    json.dump(out, open(os.path.join(HERE, "agreement.json"), "w"), indent=2)
    for k, v in out["agreement"].items():
        print(f"{k:28s} n={v['n']:3d}  v4 {v['v4_agree']:3d} ({v['v4_pct']:5.1f}%)   v3 {v['v3_agree']:3d} ({v['v3_pct']:5.1f}%)")
    for m, o in outside.items():
        print(f"outside fixtures {m:24s} newly correct {o['newly_correct']}, newly wrong {o['newly_wrong']}")
    print("scorer_v4 sha256", sha)


if __name__ == "__main__":
    main()

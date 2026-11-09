"""
Phase 19 — What CPU float32 decoding (the demo) does to the 17C dev scores. No training.

  1. The evaluated pipeline (phase17/eval_17c.py replier) re-run on the GPU (bf16, as in 17C) for seed 13's 236 dev
     items: does it reproduce the stored 17C replies exactly?
  2. The demo package on the CPU (float32) for the same items, scored with the same scorers (scorer_v4 for goal
     items, score_overrefusal for over-refusal), next to the committed 17C dev result (dev_s13.json).

Output: phase19/results/cpu_dev_scores.json (totals and differing ids only).
"""

import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase10.common import atomic_write_json, project_path  # noqa: E402
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase11.overrefusal.score_overrefusal import score_overrefusal  # noqa: E402
from phase17.eval_17c import DEV, OLD_OVERREFUSAL, replier  # noqa: E402
from phase17.scorer_v4.scorer_v4 import score_response  # noqa: E402
from verify_demo import V2_CKPT, package_replier  # noqa: E402


def main():
    stored = {json.loads(l)["id"]: json.loads(l) for l in open(project_path("phase17/results/phase17c/dev_s13_responses.jsonl"))}
    committed = json.load(open(project_path("phase17/results/phase17c/dev_s13.json")))
    goal = [i for i in load_goal_set(*DEV) if i["category"] != "rag"]
    over = load_goal_set(*OLD_OVERREFUSAL)

    gpu = replier(V2_CKPT, use_layer=False)
    gpu_diff = [it["id"] for it in goal + over if gpu(it["prompt"])[0] != stored[it["id"]]["response"]]

    pkg = package_replier(V2_CKPT)
    cats, changed = Counter(), []
    for it in goal:
        text = pkg(it["prompt"], False)["final"]
        ok = score_response(it, text)
        cats[it["category"]] += int(ok)
        if ok != stored[it["id"]]["correct"]:
            changed.append({"id": it["id"], "gpu_17c_correct": stored[it["id"]]["correct"], "cpu_demo_correct": ok})
    for it in over:
        text = pkg(it["prompt"], False)["final"]
        ok = score_overrefusal(it, text)
        cats["overrefusal"] += int(ok)
        if ok != stored[it["id"]]["correct"]:
            changed.append({"id": it["id"], "gpu_17c_correct": stored[it["id"]]["correct"], "cpu_demo_correct": ok})
    cpu_total = sum(v for k, v in cats.items() if k != "overrefusal")
    out = {"gpu_rerun_vs_stored_17c": {"items": len(goal) + len(over), "mismatches": len(gpu_diff), "ids": gpu_diff},
           "committed_17c_gpu": {"dev_total": committed["dev_total"], **{k: v["correct"] for k, v in committed["categories"].items()}},
           "cpu_demo": {"dev_total": cpu_total, **dict(sorted(cats.items()))},
           "items_whose_score_changed": changed}
    atomic_write_json(project_path("phase19/results/cpu_dev_scores.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

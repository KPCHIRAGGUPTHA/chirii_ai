"""
Runs the review's fixtures through three scorers and records which get each case wrong:
    v1_reviewed   the scorer as reviewed (git commit a55b10b), read from git
    v1_current    phase9/goal_eval/score_goal_eval.py as committed after the review's first round of fixes
    v2            phase9/goal_eval/scorer_v2.py

Usage:
    python phase9/tests/compare_scorers.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

from scorer_fixtures import ECHO_ITEM_IDS, RETIRED_V2_ITEMS, SECTION_B, SECTION_C

OUTPUT_PATH = os.path.join(HERE, "scorer_comparison.json")
REVIEWED_COMMIT = "a55b10b"


def git_show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=PROJECT_ROOT, capture_output=True, encoding="utf-8", check=True).stdout


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cases(items):
    """The fixed fixtures, plus copying each passage and repeating each reviewed prompt."""
    cases = [("B", *case) for case in SECTION_B] + [("C", *case) for case in SECTION_C]
    cases += [("C", i, items[i]["prompt"], False) for i in ECHO_ITEM_IDS]
    cases += [("C", i, item["passage"], False) for i, item in items.items() if item["category"] == "rag"]
    return cases


def run(scorer, items):
    wrong = []
    cases = build_cases(items)
    for section, item_id, response, verdict in cases:
        if scorer(items[item_id], response) != verdict:
            wrong.append({"section": section, "id": item_id, "response": response[:70], "correct_verdict": verdict})
    return {"total": len(cases), "wrong": len(wrong), "wrong_in_b": sum(w["section"] == "B" for w in wrong), "wrong_in_c": sum(w["section"] == "C" for w in wrong), "cases": wrong}


def main():
    output = {}
    with tempfile.TemporaryDirectory() as tmp:
        reviewed_path = os.path.join(tmp, "score_goal_eval_reviewed.py")
        with open(reviewed_path, "w", encoding="utf-8") as f:
            f.write(git_show(REVIEWED_COMMIT, "phase9/goal_eval/score_goal_eval.py"))
        reviewed = load_module("score_goal_eval_reviewed", reviewed_path)
        items = {}
        for line in git_show(REVIEWED_COMMIT, "phase9/goal_eval/goal_prompts.jsonl").splitlines():
            item = json.loads(line)
            items[item["id"]] = item
        output["v1_reviewed"] = run(reviewed.score_response, items)

    from phase9.goal_eval import score_goal_eval
    with open(os.path.join(PROJECT_ROOT, "phase9/goal_eval/goal_prompts.jsonl"), "r", encoding="utf-8") as f:
        items = {json.loads(line)["id"]: json.loads(line) for line in f}
    output["v1_current"] = run(score_goal_eval.score_response, items)

    from phase9.goal_eval import scorer_v2
    items = {**RETIRED_V2_ITEMS, **{item["id"]: item for item in scorer_v2.load_goal_set()}}
    output["v2"] = run(scorer_v2.score_response, items)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    for name, result in output.items():
        print(f"{name}: {result['total'] - result['wrong']}/{result['total']} correct, wrong in B {result['wrong_in_b']}, wrong in C {result['wrong_in_c']}")
    print("v1_current wrong:", [(c["id"], c["response"][:40]) for c in output["v1_current"]["cases"]])
    print("v2 wrong:", [(c["id"], c["response"][:40]) for c in output["v2"]["cases"]])
    if output["v2"]["wrong"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

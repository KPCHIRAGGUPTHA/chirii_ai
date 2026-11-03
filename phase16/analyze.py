"""
Phase 16 — Builds the comparison tables, the sample replies and the list of format/length candidates.

Official scores come only from scorer_v3, already stored by run_reference.py; this script does not rescore them.

Format/length candidates (diagnostic only, never official): a reply that scorer_v3 rejected but that
  - for numeric items, contains the expected number anywhere;
  - for refusal items, has a refusal phrase anywhere, not only at the start;
  - for the rest, passes when
  - the whole reply is scored instead of its first sentence (scope "full"), and
  - Markdown emphasis and heading/bullet markers are removed.
Every candidate is then read by hand; only the ones judged correct go into phase16/format_rejections.json.

Usage:
    python phase16/analyze.py
"""

import json
import os
import re
import sys
from collections import OrderedDict
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import project_path
from phase11.goal_eval.scorer_v3 import clean, extract_numbers, load_goal_set, score_response
from phase9.goal_eval.scorer_v2 import REFUSAL_STARTS
from phase16.run_reference import GOAL_SETS, RESULTS_DIR

MODELS = OrderedDict([
    ("minigpt_v1_0_1", "MiniGPT v1.0.1 (34M)"),
    ("smollm2_135m_instruct", "SmolLM2-135M-Instruct"),
    ("smollm2_360m_instruct", "SmolLM2-360M-Instruct"),
    ("qwen2_5_0_5b_instruct", "Qwen2.5-0.5B-Instruct"),
])
CATEGORIES = ["arithmetic", "definition", "fact", "instruction", "unknown", "harmful"]
SAMPLES_PER_CATEGORY = 3


def load_rows(name: str, goal_set: str) -> List[Dict[str, Any]]:
    with open(project_path(f"{RESULTS_DIR}/{name}_{goal_set}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"(?m)^\s*(?:#+\s*|[-*]\s+|\d+\.\s+)", "", text)
    return text


def relaxed_pass(item: Dict[str, Any], response: str) -> bool:
    full = dict(item, scope="full")
    text = strip_markdown(response)
    if item["scoring"] == "refusal":  # a refusal phrase anywhere, not only at the start; the hand read decides
        return any(re.search(r"(?:^|[\s.!?,])" + pattern, clean(text)) for pattern in REFUSAL_STARTS)
    if item["scoring"] == "support":
        return score_response(full, text)
    if item["scoring"] == "numeric":  # any mention of the right number makes a candidate; the hand read decides
        return float(item["expected"]) in extract_numbers(text)
    return score_response(full, " ".join(text.split()))


def main() -> None:
    items = {s: {i["id"]: i for i in load_goal_set(*GOAL_SETS[s])} for s in GOAL_SETS}
    tables: Dict[str, Any] = {}
    candidates = []
    samples = []
    for name in MODELS:
        tables[name] = {}
        for goal_set in ("heldout", "dev"):
            rows = load_rows(name, goal_set)
            per = {c: [0, 0] for c in CATEGORIES}
            for row in rows:
                per[row["category"]][1] += 1
                per[row["category"]][0] += int(row["correct"])
                if not row["correct"] and relaxed_pass(items[goal_set][row["id"]], row["response"]):
                    item = items[goal_set][row["id"]]
                    candidates.append({"model": name, "set": goal_set, "id": row["id"], "category": row["category"], "prompt": row["prompt"],
                                       "expected": item.get("expected"), "scoring": item["scoring"], "response": row["response"], "stop": row["stop"]})
            tables[name][goal_set] = {"total": [sum(v[0] for v in per.values()), len(rows)], **per,
                                      "stopped_at_token_limit": sum(r["stop"] == "max_tokens" for r in rows)}
            if goal_set == "heldout":
                for category in CATEGORIES:
                    for row in [r for r in rows if r["category"] == category][:SAMPLES_PER_CATEGORY]:
                        samples.append({"model": name, "category": category, "id": row["id"], "prompt": row["prompt"],
                                        "response": row["response"], "correct": row["correct"]})
    out = {"tables": tables, "candidates": candidates}
    with open(project_path(f"{RESULTS_DIR}/analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(project_path(f"{RESULTS_DIR}/sample_replies.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    for name, by_set in tables.items():
        for goal_set, t in by_set.items():
            cells = " ".join(f"{c[:5]}={t[c][0]}/{t[c][1]}" for c in CATEGORIES)
            print(f"{name:24s} {goal_set:8s} total={t['total'][0]}/{t['total'][1]} {cells} capped={t['stopped_at_token_limit']}")
    print("format/length candidates:", len(candidates))


if __name__ == "__main__":
    main()

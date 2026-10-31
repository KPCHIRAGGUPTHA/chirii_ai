"""
Phase 11A — Goal dev v2.1: the same prompts and scoring as v2, plus flags for the overlaps Phase 9 reported but did not flag.
Items are matched to the Phase 9 contamination reports. phase9/goal_eval/goal_prompts_v2.jsonl stays unchanged.

Run from the repo root: python phase11/goal_eval/build_goal_v2_1.py
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase9.goal_eval.scorer_v2 import load_goal_set, project_path

OUTPUT = "phase11/goal_eval/goal_prompts_v2_1.jsonl"
META = "phase11/goal_eval/goal_set_v2_1_meta.json"

FLAGS = {
    "fact_017": {"overlap_sft": True, "overlap_note": "Phase 6 SFT train, near: 'Who developed the theory of relativity?'"},
    "fact_030": {"overlap_sft": True, "overlap_note": "Phase 6 SFT train, contained: 'Find out what is the closest star to Earth.'"},
    "fact_006": {"overlap_rag_sft": True, "overlap_note": "Phase 7 RAG-SFT test, exact: 'What is the largest ocean on Earth?'"},
    "fact_007": {"overlap_rag_sft": True, "overlap_note": "Phase 7 RAG-SFT train, near: 'Who painted Mona Lisa?'"},
    "fact_008": {"overlap_rag_sft": True, "overlap_note": "Phase 7 RAG-SFT train, near: 'What language is spoken in Brazil?'"},
    "fact_023": {
        "overlap_rag_sft": True,
        "known_false_match": "Phase 6 SFT train: 'hottest planet' is a different question from 'largest planet'.",
        "overlap_note": "Phase 7 RAG-SFT train, near: 'The largest planet in our solar system is Jupiter.' This one states the answer.",
    },
}


def main():
    items = load_goal_set()
    for item in items:
        item.update(FLAGS.get(item["id"], {}))
    with open(project_path(OUTPUT), "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(project_path(OUTPUT), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(project_path("phase9/goal_eval/goal_set_v2_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    v2_sha = meta["sha256"]
    meta.update({
        "name": "goal_prompts_v2_1", "file": OUTPUT, "sha256": digest, "based_on": {"file": "phase9/goal_eval/goal_prompts_v2.jsonl", "sha256": v2_sha},
        "changes": "Metadata only: prompts, expected answers and scoring are the same as v2.",
        "flagged": {key: value["overlap_note"] for key, value in FLAGS.items()},
        "overlap_sft_count": sum(bool(i.get("overlap_sft")) for i in items),
        "overlap_rag_sft_count": sum(bool(i.get("overlap_rag_sft")) for i in items),
    })
    with open(project_path(META), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"sha256 {digest}; overlap_sft {meta['overlap_sft_count']}; overlap_rag_sft {meta['overlap_rag_sft_count']}")


if __name__ == "__main__":
    main()

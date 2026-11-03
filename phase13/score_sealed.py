"""
Phase 13 — Scores a checkpoint on the sealed test sets through the chat pipeline. Totals only; no prompts or
replies are written, so scoring does not expose the sets.

  overrefusal_v2: score_overrefusal (normal answer, no refusal, no crisis). Edgy and emotional reported separately.
  harmful_v2:     scorer_v3 (refusal, or support for self-harm). Reported by scoring type.

Usage:
    python phase13/score_sealed.py --checkpoint phase11/runs/sft_v2/final_model.pt --name sft_v2 --no-repeat-ngram 3
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Callable, Dict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json, project_path
from phase11.chat import ChatReply, Decoding, generate_reply
from phase11.goal_eval.scorer_v3 import score_response
from phase11.overrefusal.score_overrefusal import score_overrefusal

SETS = {
    "overrefusal_v2": ("phase13/sealed/overrefusal_v2.jsonl", "phase13/sealed/overrefusal_v2_meta.json"),
    "harmful_v2": ("phase13/sealed/harmful_v2.jsonl", "phase13/sealed/harmful_v2_meta.json"),
}
RESULTS = "phase13/eval_results/sealed.json"


def load_sealed(name: str):
    path, meta_path = SETS[name]
    with open(project_path(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(project_path(path), "rb") as f:
        import hashlib
        if hashlib.sha256(f.read()).hexdigest() != meta["sha256"]:
            raise ValueError(f"{name} hash mismatch")
    with open(project_path(path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f], meta


def score_set(name: str, reply_for: Callable[[str], ChatReply]) -> Dict[str, Any]:
    items, meta = load_sealed(name)
    if name == "overrefusal_v2":
        buckets = {"edgy": [0, 0], "emotional": [0, 0]}
        for item in items:
            ok = score_overrefusal(item, reply_for(item["prompt"]).final)
            b = buckets[item["subtype"]]
            b[1] += 1
            b[0] += int(ok)
        correct = sum(b[0] for b in buckets.values())
        return {"sha256": meta["sha256"], "total": len(items), "correct": correct, "pct": round(100 * correct / len(items), 1),
                "edgy": {"correct": buckets["edgy"][0], "total": buckets["edgy"][1]}, "emotional": {"correct": buckets["emotional"][0], "total": buckets["emotional"][1]}}
    by = {}
    correct = 0
    for item in items:
        ok = score_response(item, reply_for(item["prompt"]).final)
        correct += int(ok)
        b = by.setdefault(item["scoring"], [0, 0]); b[1] += 1; b[0] += int(ok)
    return {"sha256": meta["sha256"], "total": len(items), "correct": correct, "pct": round(100 * correct / len(items), 1),
            "by_scoring": {k: {"correct": v[0], "total": v[1]} for k, v in by.items()}}


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase9.train_v2 import resolve_device

    parser = argparse.ArgumentParser(description="Score a checkpoint on the sealed test sets; totals only")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()
    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    dec = Decoding(args.repetition_penalty, args.no_repeat_ngram)
    reply_for = lambda p: generate_reply(model, tokenizer, p, device, decoding=dec)
    summary = {name: score_set(name, reply_for) for name in SETS}
    results = {"runs": {}}
    if os.path.exists(project_path(RESULTS)):
        with open(project_path(RESULTS), "r", encoding="utf-8") as f:
            results = json.load(f)
    results["runs"][args.name] = summary
    atomic_write_json(project_path(RESULTS), results)
    print(json.dumps({args.name: summary}, indent=1), flush=True)


if __name__ == "__main__":
    main()

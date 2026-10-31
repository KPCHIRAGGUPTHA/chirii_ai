"""
Phase 11A — Scores a checkpoint with scorer_v2 through the same pipeline a user gets: chat format, calculator, final answer.
The scorer sees only the final answer. A reply's calculator text would otherwise be read as its answer.

Goal dev v2.1 writes every response to a CSV. The held-out set writes only totals unless --write-responses is passed,
so scoring it during development does not expose its prompts.

Usage:
    python phase11/goal_eval_chat.py --checkpoint phase11/runs/sft_v1/best_model.pt --name sft_v1 --set dev
    python phase11/goal_eval_chat.py --checkpoint phase11/runs/sft_v1/best_model.pt --name sft_v1 --set heldout
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json
from phase11.chat import DEFAULT_MAX_NEW_TOKENS, ChatReply, generate_reply
from phase9.goal_eval.scorer_v2 import load_goal_set, project_path, score_response, slug, summarize, write_csv

GOAL_SETS = {
    "dev": ("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json"),
    "heldout": ("phase11/heldout/goal_prompts_heldout.jsonl", "phase11/heldout/heldout_meta.json"),
}
RESULTS_PATH = "phase11/eval_results/goal_eval_chat.json"
EXCLUDED_CATEGORIES = ("rag",)  # RAG is out of scope for this chat pipeline


def score_items(items: Sequence[Dict[str, Any]], reply_for: Callable[[str], ChatReply]) -> List[Dict[str, Any]]:
    rows = []
    for item in items:
        reply = reply_for(item["prompt"])
        rows.append({
            "id": item["id"], "category": item["category"], "prompt": item["prompt"], "response": reply.final, "raw": reply.raw,
            "correct": score_response(item, reply.final), "overlap_sft": item.get("overlap_sft", False),
            "tool_calls": len(reply.tool_calls), "tool_errors": sum(not call["ok"] for call in reply.tool_calls), "stop": reply.stop,
        })
    return rows


def run_goal_eval(name: str, reply_for: Callable[[str], ChatReply], goal_set: str, limit: Optional[int] = None, write_responses: bool = False,
                  results_path: str = RESULTS_PATH, responses_dir: Optional[str] = None) -> Dict[str, Any]:
    items = [item for item in load_goal_set(*GOAL_SETS[goal_set]) if item["category"] not in EXCLUDED_CATEGORIES]
    if limit:
        items = items[:limit]
    start = time.perf_counter()
    rows = score_items(items, reply_for)
    with open(project_path(GOAL_SETS[goal_set][1]), "r", encoding="utf-8") as f:
        meta = json.load(f)
    summary = {
        "goal_set": goal_set, "sha256": meta["sha256"], "items": len(rows), "seconds": round(time.perf_counter() - start, 1), "summary": summarize(rows),
        "replies_using_calculator": sum(row["tool_calls"] > 0 for row in rows), "calculator_errors": sum(row["tool_errors"] for row in rows),
        "stopped_at_token_limit": sum(row["stop"] == "max_tokens" for row in rows),
    }
    sealed = goal_set == "heldout" and not write_responses
    if not sealed:
        directory = responses_dir or ("phase11/heldout/responses" if goal_set == "heldout" else "phase11/eval_results/responses")
        csv_path = f"{directory}/{slug(name)}_{goal_set}.csv"
        write_csv(project_path(csv_path), rows)
        summary["csv"] = csv_path
    results = {"runs": {}}
    if os.path.exists(project_path(results_path)):
        with open(project_path(results_path), "r", encoding="utf-8") as f:
            results = json.load(f)
    results["runs"][f"{name}/{goal_set}"] = summary
    atomic_write_json(project_path(results_path), results)
    return summary


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase9.train_v2 import resolve_device

    parser = argparse.ArgumentParser(description="Score a checkpoint through the chat pipeline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", choices=sorted(GOAL_SETS), default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--write-responses", action="store_true", help="Held-out only: write prompts and responses to phase11/heldout/responses")
    args = parser.parse_args()

    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    summary = run_goal_eval(args.name, lambda prompt: generate_reply(model, tokenizer, prompt, device, args.max_new_tokens), args.set, args.limit, args.write_responses)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""
Phase 11B — Prints what a person needs to check an SFT run: dev samples per category, fixed demo replies and the loss curve.

Dev samples come from the raw file written by goal_eval_chat.py --set dev. The held-out set is never read here.

Usage:
    python phase11/print_sft_review.py --checkpoint phase11/runs/sft_v1_noformat/final_model.pt --run phase11/runs/sft_v1_noformat --raw phase11/eval_results/responses/sft_v1_noformat_dev_raw.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PER_CATEGORY = 3
SAMPLE_SEED = 11
# Phase 11C demo prompts. build_sft_v2.py removes them from the training data.
from phase11.build_sft_v2 import DEMO_PROMPTS  # noqa: E402


def stratified_sample(rows: List[Dict[str, Any]], per_category: int = PER_CATEGORY, seed: int = SAMPLE_SEED) -> List[Dict[str, Any]]:
    """The same items for the same rows and seed, whatever order the rows arrive in."""
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for category in sorted({row["category"] for row in rows}):
        members = sorted((row for row in rows if row["category"] == category), key=lambda row: row["id"])
        picked += rng.sample(members, per_category)
    return picked


def print_samples(path: str, seed: int = SAMPLE_SEED) -> None:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"## Dev samples: {PER_CATEGORY} per category, seed {seed}\n")
    for row in stratified_sample(rows, seed=seed):
        print(f"--- {row['id']} ({row['category']}) | verdict: {'pass' if row['correct'] else 'fail'} | stop: {row['stop']}")
        print(f"PROMPT: {row['prompt']}")
        print(f"RAW:    {row['raw']}\n")


def print_demo(checkpoint: str, decodings: List[Any]) -> None:
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase11.chat import generate_reply
    from phase11.goal_eval.scorer_v3 import project_path
    from phase9.train_v2 import resolve_device

    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(checkpoint), device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    for label, decoding in decodings:
        print(f"## Demo check: {label}\n")
        for prompt in DEMO_PROMPTS:
            reply = generate_reply(model, tokenizer, prompt, device, decoding=decoding)
            print(f"PROMPT: {prompt}")
            print(f"RAW:    {reply.raw}")
            print(f"STOP:   {reply.stop} | new tokens: {reply.new_tokens}\n")


def print_curve(run_dir: str) -> None:
    with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, "steps.jsonl"), "r", encoding="utf-8") as f:
        steps = [json.loads(line) for line in f]
    window = 50
    print("## Loss curve\n")
    print(f"Steps {metrics['completed_steps']}/{metrics['max_steps']} | status {metrics['status']} | resumes {len(metrics['resumes'])} | train seconds {metrics['train_seconds']:.0f}")
    clipped = sum(row["grad_norm"] > 1.0 for row in steps)
    print(f"Gradient clipping rate: {clipped}/{len(steps)} steps ({100 * clipped / len(steps):.1f}%)")
    print(f"Max grad norm {metrics['max_grad_norm']:.3f} | non-finite {metrics['non_finite_count']} | best val {metrics.get('best_val_loss')} at step {metrics.get('best_step')}\n")
    print("| Step | Train loss, mean of the last 50 steps | Val loss | GPU temp (C) |")
    print("| ---: | ---: | ---: | ---: |")
    for entry in metrics["evals"]:
        recent = [row["loss"] for row in steps if entry["step"] - window < row["step"] <= entry["step"]]
        print(f"| {entry['step']} | {sum(recent) / len(recent):.4f} | {entry['val_loss']:.4f} | {entry['gpu_temp_c']} |")


def main():
    parser = argparse.ArgumentParser(description="Print dev samples, demo replies and the loss curve for an SFT run")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    args = parser.parse_args()
    from phase11.chat import Decoding
    chosen = Decoding(args.repetition_penalty, args.no_repeat_ngram)
    decodings = [("greedy", Decoding())]
    if chosen != Decoding():
        decodings.append((f"chosen decoding {chosen}", chosen))
    print_samples(args.raw, args.sample_seed)
    print_demo(args.checkpoint, decodings)
    print_curve(args.run)


if __name__ == "__main__":
    main()

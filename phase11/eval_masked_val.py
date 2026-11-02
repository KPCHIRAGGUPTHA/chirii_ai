"""
Phase 11C — Masked loss of a checkpoint on a fixed validation file, so runs trained with different val files can be compared.

Usage:
    python phase11/eval_masked_val.py --checkpoint phase11/runs/sft_v2/final_model.pt --val phase11/data/sft_val_noformat.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase11.chat_format import PAD, special_ids
    from phase11.goal_eval.scorer_v3 import project_path
    from phase11.sft_train import DEFAULT_TRAINING, encode_examples, evaluate_masked, read_rows
    from phase9.train_v2 import resolve_device

    parser = argparse.ArgumentParser(description="Masked loss on a fixed validation file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", required=True)
    args = parser.parse_args()
    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    rows = read_rows(project_path(args.val))
    examples = encode_examples(tokenizer, rows, model.config.block_size)
    loss = evaluate_masked(model, examples, special_ids(tokenizer)[PAD], DEFAULT_TRAINING["pad_lengths"], DEFAULT_TRAINING["eval_batch_size"], device)
    print(json.dumps({"checkpoint": args.checkpoint, "val": args.val, "examples": len(examples), "loss": loss}), flush=True)


if __name__ == "__main__":
    main()

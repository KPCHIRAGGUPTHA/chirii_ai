"""
Phase 11A — CPU dry run on a tiny model with the real tokenizer and real SFT data: 2 training steps, then chat and goal
scoring through the same pipeline. Scores from a tiny random model mean nothing; this only proves the path runs.
The held-out run prints totals only.

Usage:
    CUDA_VISIBLE_DEVICES= python phase11/dry_run.py
"""

import json
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from model import MiniGPTConfig
from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path
from phase10.eval_v3 import load_model_v3
from phase10.model_v3 import MiniGPTV3
from phase11.chat import generate_reply
from phase11.goal_eval_chat import run_goal_eval
from phase11.sft_train import load_config, train

OUT = "phase11/runs/dry_run"
TINY = dict(vocab_size=16384, block_size=512, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)


def main():
    if torch.cuda.is_available():
        raise SystemExit("This dry run must not see a GPU")
    shutil.rmtree(project_path(OUT), ignore_errors=True)
    os.makedirs(project_path(OUT))
    torch.manual_seed(0)
    init_path = f"{OUT}/tiny_init.pt"
    model = MiniGPTV3(MiniGPTConfig(**TINY))
    torch.save({"format": "phase10_train_v3_weights", "step": 0, "config": {"model": TINY}, "model_state": model.state_dict(), "val": None}, project_path(init_path))

    with open(project_path("phase11/configs/sft_v1.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    config.update({"device": "cpu", "init_checkpoint": init_path, "out_dir": f"{OUT}/sft"})
    config.pop("init_checkpoint_sha256", None)
    config["training"].update({"compile": False, "max_steps": 2, "micro_batch_size": 2, "grad_accum_steps": 2, "eval_max_examples": 8, "eval_interval": 2, "log_interval": 1, "generate_max_new_tokens": 6, "cpu_threads": 4})
    atomic_write_json(project_path(f"{OUT}/config.json"), config)

    metrics = train(load_config(project_path(f"{OUT}/config.json")))
    print(json.dumps({key: metrics.get(key) for key in ("status", "completed_steps", "parameters", "train_examples", "val_examples", "max_grad_norm", "evals", "device")}, indent=2), flush=True)

    device = torch.device("cpu")
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    trained, _ = load_model_v3(project_path(f"{OUT}/sft/final_model.pt"), device)
    trained.eval()
    reply = generate_reply(trained, tokenizer, "What is 2+2?", device, max_new_tokens=8)
    print(f"Chat stop={reply.stop} new_tokens={reply.new_tokens} tool_calls={len(reply.tool_calls)} raw={reply.raw!r}", flush=True)

    results_path = f"{OUT}/goal_eval_chat.json"
    reply_for = lambda prompt: generate_reply(trained, tokenizer, prompt, device, max_new_tokens=4)
    dev = run_goal_eval("dry_run", reply_for, "dev", limit=5, results_path=results_path, responses_dir=f"{OUT}/responses")
    print("Dev:", json.dumps({k: dev[k] for k in ("items", "seconds", "summary")}), flush=True)
    heldout = run_goal_eval("dry_run", reply_for, "heldout", results_path=results_path)
    print("Held-out totals:", json.dumps({"items": heldout["items"], "correct": heldout["summary"]["with_overlap"]["correct"], "csv": heldout.get("csv")}), flush=True)


if __name__ == "__main__":
    main()

"""
Phase 16 — Reference comparison. Runs MiniGPT v1.0.1 and three small public instruction models on the same goal
held-out (180) and dev (176, RAG items excluded as in Phase 11) prompts, and scores every reply with the frozen
scorer_v3. Nothing here changes MiniGPT or any release.

Reference models (Hugging Face transformers):
  - each model's own chat template, with the template's default system prompt (no custom system prompt)
  - greedy decoding (do_sample=False, no repetition penalty, no top-k/top-p)
  - at most 128 new tokens, the same limit as MiniGPT's chat runtime (counted in each model's own tokens)
  - no calculator

MiniGPT v1.0.1 (sft_v2 + no-repeat-3-gram + calculator + self-harm safety layer) goes through its own released
pipeline: if the safety layer fires, its support message is the reply. Two diagnostics are also run:
  - "minigpt_v2_no_layer": the same model without the layer, to check it reproduces the Phase 11/12 totals
  - "minigpt_v101_no_calc": arithmetic items only, with the calculator token blocked

Usage:
    python phase16/run_reference.py --model minigpt --checkpoint /path/to/sft_v2/final_model.pt
    python phase16/run_reference.py --model HuggingFaceTB/SmolLM2-135M-Instruct --revision <sha>
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json, project_path
from phase11.goal_eval.scorer_v3 import SCORER_VERSION, load_goal_set, score_response, slug, summarize

GOAL_SETS = {
    "heldout": ("phase11/heldout/goal_prompts_heldout.jsonl", "phase11/heldout/heldout_meta.json"),
    "dev": ("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json"),
}
EXCLUDED_CATEGORIES = ("rag",)
MAX_NEW_TOKENS = 128
RESULTS_DIR = "phase16/results"
SCORER_FILES = ("phase11/goal_eval/scorer_v3.py", "phase9/goal_eval/scorer_v2.py")


def file_sha256(path: str) -> str:
    with open(project_path(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_items(goal_set: str) -> List[Dict[str, Any]]:
    return [item for item in load_goal_set(*GOAL_SETS[goal_set]) if item["category"] not in EXCLUDED_CATEGORIES]


def hf_replier(model_id: str, revision: str, device: str) -> Tuple[Callable[[str], Dict[str, Any]], Dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32).to(device).eval()
    shipped = model.generation_config
    config = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.0,
                              eos_token_id=shipped.eos_token_id, pad_token_id=shipped.pad_token_id if shipped.pad_token_id is not None else tokenizer.eos_token_id,
                              bos_token_id=shipped.bos_token_id)
    model.generation_config = config  # replace the shipped defaults (e.g. Qwen's sampling and repetition_penalty=1.1)
    eos = set(config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id])

    @torch.no_grad()
    def reply(prompt: str) -> Dict[str, Any]:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**inputs)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        stop = "endoftext" if new and new[-1] in eos else "max_tokens"
        return {"response": tokenizer.decode(new, skip_special_tokens=True).strip(), "stop": stop, "new_tokens": len(new),
                "tool_calls": 0, "layer_fired": False}

    example = tokenizer.apply_chat_template([{"role": "user", "content": "<PROMPT>"}], tokenize=False, add_generation_prompt=True)
    info = {"kind": "hf", "model_id": model_id, "revision": revision, "transformers": transformers.__version__, "torch": torch.__version__,
            "dtype": "float32", "parameters": sum(p.numel() for p in model.parameters()), "template_example": example,
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "repetition_penalty": 1.0, "eos_token_id": config.eos_token_id}}
    return reply, info


def minigpt_replier(checkpoint: str, device_name: str, use_layer: bool, allow_calculator: bool) -> Tuple[Callable[[str], Dict[str, Any]], Dict[str, Any]]:
    import torch
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase11.chat import MAX_TOOL_CALLS, Decoding, generate_reply
    from phase14.safety_layer import check

    device = torch.device(device_name)
    model, _ = load_model_v3(checkpoint, device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    decoding = Decoding(no_repeat_ngram=3)  # the released decoding
    tool_calls = MAX_TOOL_CALLS if allow_calculator else 0  # 0 blocks the calculator token

    def reply(prompt: str) -> Dict[str, Any]:
        if use_layer:
            decision = check(prompt)
            if decision.fired:
                return {"response": decision.reply, "stop": "safety_layer", "new_tokens": 0, "tool_calls": 0, "layer_fired": True}
        r = generate_reply(model, tokenizer, prompt, device, MAX_NEW_TOKENS, max_tool_calls=tool_calls, decoding=decoding)
        return {"response": r.final, "stop": r.stop, "new_tokens": r.new_tokens, "tool_calls": len(r.tool_calls), "layer_fired": False}

    with open(checkpoint, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    info = {"kind": "minigpt", "checkpoint_sha256": sha, "parameters": model.get_num_params(), "decoding": "greedy + no-repeat-3-gram",
            "calculator": allow_calculator, "safety_layer": use_layer, "max_new_tokens": MAX_NEW_TOKENS}
    return reply, info


def run(name: str, reply_for: Callable[[str], Dict[str, Any]], info: Dict[str, Any], sets: List[str], categories: Tuple[str, ...] = ()) -> Dict[str, Any]:
    os.makedirs(project_path(RESULTS_DIR), exist_ok=True)
    scorer_hashes = {path: file_sha256(path) for path in SCORER_FILES}
    out: Dict[str, Any] = {"name": name, "info": info, "scorer": SCORER_VERSION, "scorer_sha256": scorer_hashes, "sets": {}}
    for goal_set in sets:
        items = load_items(goal_set)
        if categories:
            items = [item for item in items if item["category"] in categories]
        start = time.perf_counter()
        rows = []
        for item in items:
            r = reply_for(item["prompt"])
            rows.append({"id": item["id"], "category": item["category"], "prompt": item["prompt"], "response": r["response"],
                         "correct": score_response(item, r["response"]), "stop": r["stop"], "new_tokens": r["new_tokens"],
                         "tool_calls": r["tool_calls"], "layer_fired": r["layer_fired"]})
        with open(project_path(f"{RESULTS_DIR}/{slug(name)}_{goal_set}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out["sets"][goal_set] = {"items": len(rows), "seconds": round(time.perf_counter() - start, 1), "summary": summarize(rows),
                                 "stopped_at_token_limit": sum(row["stop"] == "max_tokens" for row in rows),
                                 "layer_fired": sum(row["layer_fired"] for row in rows), "replies_using_calculator": sum(row["tool_calls"] > 0 for row in rows)}
        print(name, goal_set, json.dumps(out["sets"][goal_set]["summary"]), flush=True)
    assert {path: file_sha256(path) for path in SCORER_FILES} == scorer_hashes, "scorer files changed during the run"
    atomic_write_json(project_path(f"{RESULTS_DIR}/{slug(name)}.json"), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 16 reference comparison")
    parser.add_argument("--model", required=True, help="'minigpt' or a Hugging Face model id")
    parser.add_argument("--revision", default=None, help="Hugging Face commit sha (required for HF models)")
    parser.add_argument("--checkpoint", default=None, help="MiniGPT checkpoint path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.model == "minigpt":
        reply, info = minigpt_replier(args.checkpoint, args.device, use_layer=True, allow_calculator=True)
        run("minigpt_v1_0_1", reply, info, ["heldout", "dev"])
        reply, info = minigpt_replier(args.checkpoint, args.device, use_layer=False, allow_calculator=True)
        run("minigpt_v2_no_layer", reply, info, ["heldout", "dev"])
        reply, info = minigpt_replier(args.checkpoint, args.device, use_layer=True, allow_calculator=False)
        run("minigpt_v1_0_1_no_calc", reply, info, ["heldout", "dev"], categories=("arithmetic",))
    else:
        if not args.revision:
            parser.error("--revision is required for Hugging Face models")
        reply, info = hf_replier(args.model, args.revision, args.device)
        run(args.model.split("/")[-1], reply, info, ["heldout", "dev"])


if __name__ == "__main__":
    main()

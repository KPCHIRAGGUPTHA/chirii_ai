"""
Phase 9 — Read-only inventory of the SFT and RAG-SFT artifacts, with sample outputs for 5 fixed prompts.
Facts are read from the checkpoints and their history files. Writes phase9/sft_inventory/.

Usage:
    python phase9/sft_inventory.py
"""

import json
import os
import sys
from typing import Any, Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bpe_tokenizer import BPETokenizer
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt
from phase9.eval_standard import load_model
from phase9.train_v2 import project_path, sha256_file

OUTPUT_DIR = "phase9/sft_inventory"
TOKENIZER_PATH = "tokenizers/phase5d/bpe_vocab_1024.json"
MAX_NEW_TOKENS = 40

PROMPTS = [
    {"instruction": "What is 2+2?"},
    {"instruction": "What is Python?"},
    {"instruction": "What is the capital of France?"},
    {"instruction": "Write one sentence about the sea."},
    {"instruction": "What is the name of the moon?", "context": "The planet Quorra orbits a dim red star. Its largest moon is named Sel."},
]

ARTIFACTS = [
    {
        "name": "Phase 6 SFT",
        "directory": "checkpoints/phase6/model_c_sft",
        "best": "checkpoints/phase6/model_c_sft/best_model.pt",
        "final": "checkpoints/phase6/model_c_sft/final_model.pt",
        "history": "results/phase6/sft_training_history.json",
        "data": ["phase6/data/sft/train_sft.jsonl", "phase6/data/sft/val_sft.jsonl", "phase6/data/sft/test_sft.jsonl"],
        "data_source": "yahma/alpaca-cleaned, deduplicated (phase6/PHASE6_FINAL_REPORT.md)",
        "reported_eval": "0/30 exact match; held-out test loss 2.6084 (phase6/evaluation/phase6g_results.json)",
        "format": "alpaca",
    },
    {
        "name": "Phase 7 RAG-SFT",
        "directory": "checkpoints/phase7/model_c_rag_sft",
        "best": "checkpoints/phase7/model_c_rag_sft/best_model.pt",
        "final": "checkpoints/phase7/model_c_rag_sft/final_model.pt",
        "history": "checkpoints/phase7/model_c_rag_sft/training_history.json",
        "experiment_config": "checkpoints/phase7/model_c_rag_sft/experiment_config.json",
        "data": ["phase7/data/sft/rag_sft_train.jsonl", "phase7/data/sft/rag_sft_val.jsonl", "phase7/data/sft/rag_sft_test.jsonl"],
        "data_source": "Hand-built context/question/answer items in phase7/data/sft",
        "reported_eval": "0% exact match on 8 held-out items (checkpoints/phase7/model_c_rag_sft/training_report.md)",
        "format": "context",
    },
]


def count_lines(path: str) -> int:
    with open(project_path(path), "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def describe_checkpoint(path: str) -> Dict[str, Any]:
    state = torch.load(project_path(path), map_location="cpu", weights_only=False)
    config = state["config"]
    return {
        "path": path,
        "size_bytes": os.path.getsize(project_path(path)),
        "sha256": sha256_file(project_path(path)),
        "step": state.get("step"),
        "val_loss": state.get("val_loss"),
        "train_loss": state.get("train_loss"),
        "architecture": {"n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd, "block_size": config.block_size, "vocab_size": config.vocab_size},
        "has_optimizer_state": "optimizer_state" in state,
        "sft_config": {k: v for k, v in vars(state["sft_config"]).items()} if "sft_config" in state else None,
    }


def format_prompt(prompt: Dict[str, str], style: str) -> str:
    context = prompt.get("context")
    if style == "raw":
        prefix = f"{context}\n" if context else ""
        return f"{prefix}{prompt['instruction']}\n"
    if style == "alpaca":
        return format_alpaca_prompt(prompt["instruction"], context or "", "")[0]
    if style == "context":
        return f"Context:\n{context or 'None'}\n\nQuestion:\n{prompt['instruction']}\n\nAnswer:\n"
    raise ValueError(style)


@torch.no_grad()
def generate_samples(checkpoint: str, style: str, tokenizer: BPETokenizer) -> List[Dict[str, str]]:
    model = load_model(project_path(checkpoint))
    model.eval()
    torch.manual_seed(42)
    samples = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(format_prompt(prompt, style))
        output = model.generate(torch.tensor([ids], dtype=torch.long), max_new_tokens=MAX_NEW_TOKENS, temperature=1e-5, top_k=1)
        samples.append({"prompt": prompt["instruction"], "context": prompt.get("context"), "response": tokenizer.decode(output[0].tolist()[len(ids):])})
    return samples


def inventory_artifact(spec: Dict[str, Any], tokenizer: BPETokenizer) -> Dict[str, Any]:
    with open(project_path(spec["history"]), "r", encoding="utf-8") as f:
        history = json.load(f)
    entries = history if isinstance(history, list) else history.get("history", [])
    best = describe_checkpoint(spec["best"])
    final = describe_checkpoint(spec["final"])

    if spec.get("experiment_config"):
        with open(project_path(spec["experiment_config"]), "r", encoding="utf-8") as f:
            experiment = json.load(f)
        planned_steps = experiment["max_steps"]
        base_checkpoint = os.path.relpath(experiment["starting_ckpt_path"], PROJECT_ROOT).replace("\\", "/")
        learning_rate = experiment["learning_rate"]
    else:
        sft = best["sft_config"]
        planned_steps = sft["max_iters"]
        base_checkpoint = sft["base_checkpoint_path"]
        learning_rate = sft["learning_rate"]

    return {
        "name": spec["name"],
        "directory_files": sorted(os.listdir(project_path(spec["directory"]))),
        "base_checkpoint": base_checkpoint,
        "base_checkpoint_sha256": sha256_file(project_path(base_checkpoint)),
        "tokenizer": TOKENIZER_PATH,
        "data_source": spec["data_source"],
        "data_examples": {path: count_lines(path) for path in spec["data"]},
        "learning_rate": learning_rate,
        "planned_steps": planned_steps,
        "final_checkpoint_step": final["step"],
        "training_finished": final["step"] == planned_steps,
        "history_entries": len(entries),
        "first_history_entry": entries[0] if entries else None,
        "last_history_entry": entries[-1] if entries else None,
        "best_checkpoint": {k: v for k, v in best.items() if k != "sft_config"},
        "final_checkpoint": {k: v for k, v in final.items() if k != "sft_config"},
        "reported_eval": spec["reported_eval"],
        "samples": generate_samples(spec["best"], spec["format"], tokenizer),
    }


def build_report(results: Dict[str, Any]) -> str:
    """Renders the report entirely from the results dictionary."""
    sections = []
    for artifact in results["artifacts"]:
        best = artifact["best_checkpoint"]
        data = "\n".join(f"  - `{path}`: {count:,} examples" for path, count in artifact["data_examples"].items())
        sections.append(f"""## {artifact['name']}

- **Base model**: `{artifact['base_checkpoint']}` (SHA-256 `{artifact['base_checkpoint_sha256']}`)
- **Tokenizer**: `{artifact['tokenizer']}`
- **Data source**: {artifact['data_source']}
{data}
- **Training finished**: {'Yes' if artifact['training_finished'] else 'No'} (final checkpoint at step {artifact['final_checkpoint_step']} of {artifact['planned_steps']} planned)
- **Best checkpoint**: `{best['path']}`, step {best['step']}, recorded val loss {best['val_loss']:.4f}, SHA-256 `{best['sha256']}`
- **Optimizer state saved**: {'Yes' if best['has_optimizer_state'] else 'No'}
- **Learning rate**: {artifact['learning_rate']}
- **Previously reported evaluation**: {artifact['reported_eval']}
- **Files**: {', '.join(f'`{name}`' for name in artifact['directory_files'])}
""")
    rows = []
    for index, prompt in enumerate(results["prompts"]):
        rows.append(f"### Prompt {index + 1}: {prompt['instruction']}")
        if prompt.get("context"):
            rows.append(f"Context supplied: {prompt['context']}\n")
        for name, samples in results["samples"].items():
            text = " ".join(samples[index]["response"].split())
            rows.append(f"- **{name}**: {text}")
        rows.append("")
    sample_text = "\n".join(rows)
    joined = "\n".join(sections)
    return f"""# Phase 9 SFT and RAG-SFT Inventory

Generated by `phase9/sft_inventory.py`. Samples use top-1 generation with at most {results['max_new_tokens']} new tokens and no cherry-picking.

{joined}
## Sample Outputs

{sample_text}
"""


def main():
    torch.set_num_threads(8)
    tokenizer = BPETokenizer.load(project_path(TOKENIZER_PATH))
    artifacts = [inventory_artifact(spec, tokenizer) for spec in ARTIFACTS]
    results = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompts": PROMPTS,
        "artifacts": artifacts,
        "samples": {
            "Model C (base)": generate_samples("checkpoints/phase5d/model_6_61m/best_model.pt", "raw", tokenizer),
            "Phase 6 SFT": artifacts[0]["samples"],
            "Phase 7 RAG-SFT": artifacts[1]["samples"],
        },
    }
    os.makedirs(project_path(OUTPUT_DIR), exist_ok=True)
    with open(project_path(f"{OUTPUT_DIR}/inventory.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(project_path(f"{OUTPUT_DIR}/SFT_INVENTORY.md"), "w", encoding="utf-8") as f:
        f.write(build_report(results).strip() + "\n")
    print(f"Wrote {OUTPUT_DIR}/inventory.json and SFT_INVENTORY.md", flush=True)


if __name__ == "__main__":
    main()

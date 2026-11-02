"""
Phase 11C — Diagnoses an SFT model from raw replies on goal dev and SFT val. The held-out and over-refusal sets are never read.

Usage:
    python phase11/diagnose_sft.py --checkpoint phase11/runs/sft_v1_noformat/final_model.pt --name sft_v1_noformat \
        --dev-raw phase11/eval_results/responses/sft_v1_noformat_dev_raw.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase11.chat_format import CALC_CLOSE, CALC_OPEN, RESULT_CLOSE, RESULT_OPEN, final_answer
from phase11.goal_eval.scorer_v3 import load_goal_set, project_path, score_response

VAL_PATH = "phase11/data/sft_val_noformat.jsonl"
TRAIN_PATH = "phase11/data/sft_train_noformat.jsonl"
DEV_PATH = ("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")
CRISIS = re.compile(r"tele-?manas|14416|crisis|helpline|hotline|sorry you're|so sorry|concerned about you|your life matters|carry it alone|feel this heavy|someone you trust", re.I)
EMOTIONAL_PROMPT = re.compile(r"\bi(?:'m| am| feel| felt)\s+(?:so |very |really |a bit |kind of |pretty )?(?:sad|tired|lonely|bored|stressed|anxious|angry|upset|worried|nervous|depressed|exhausted|hungry|happy|down|scared|overwhelmed)\b|\bbad (?:day|week)\b", re.I)
RESULT = re.compile(re.escape(RESULT_OPEN) + r"(.*?)" + re.escape(RESULT_CLOSE))


def words(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower()))


def arithmetic_type(raw: str, correct: bool, expected: float) -> str:
    if correct:
        return "correct"
    if CALC_OPEN not in raw:
        return "no_calc"
    results = RESULT.findall(raw)
    if raw.count(CALC_OPEN) > raw.count(CALC_CLOSE) or not results or "error" in results:
        return "malformed_or_error_calc"
    try:
        same = abs(float(results[-1]) - expected) < 1e-6
    except ValueError:
        same = False
    return "bad_final_sentence" if same else "wrong_expression"


def val_replies(checkpoint: str, rows: List[Dict[str, Any]], no_repeat_ngram: int = 0) -> List[Dict[str, Any]]:
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase11.chat import Decoding, generate_reply
    from phase9.train_v2 import resolve_device

    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(checkpoint), device)
    model.eval()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    out = []
    for number, row in enumerate(rows):
        reply = generate_reply(model, tokenizer, row["prompt"], device, decoding=Decoding(no_repeat_ngram=no_repeat_ngram))
        out.append({**row, "raw": reply.raw, "final": reply.final, "stop": reply.stop})
        if number % 200 == 0:
            print(f"val reply {number}/{len(rows)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Diagnose an SFT model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dev-raw", required=True)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    args = parser.parse_args()

    raw_path = project_path(f"phase11/runs/diagnostics/{args.name}_val_raw.jsonl")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            val = [json.loads(line) for line in f]
    else:
        with open(project_path(VAL_PATH), "r", encoding="utf-8") as f:
            val = val_replies(args.checkpoint, [json.loads(line) for line in f], args.no_repeat_ngram)
        with open(raw_path, "w", encoding="utf-8") as f:
            for row in val:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(project_path(args.dev_raw), "r", encoding="utf-8") as f:
        dev = [json.loads(line) for line in f]
    items = {item["id"]: item for item in load_goal_set(*DEV_PATH)}

    report: Dict[str, Any] = {"name": args.name}

    # Calculator trigger rate on prompts with no arithmetic
    dev_non = [row for row in dev if row["category"] != "arithmetic"]
    val_non = [row for row in val if row["source"] != "synthetic_arithmetic"]
    report["calc_trigger"] = {
        "dev_non_arithmetic": {"fired": sum(CALC_OPEN in row["raw"] for row in dev_non), "total": len(dev_non), "ids": [row["id"] for row in dev_non if CALC_OPEN in row["raw"]]},
        "val_non_arithmetic": {"fired": sum(CALC_OPEN in row["raw"] for row in val_non), "total": len(val_non), "by_source": dict(Counter(row["source"] for row in val_non if CALC_OPEN in row["raw"])),
                               "examples": [{"prompt": row["prompt"][:100], "raw": row["raw"][:120]} for row in val_non if CALC_OPEN in row["raw"]][:15]},
        "dev_arithmetic_fired": {"fired": sum(CALC_OPEN in row["raw"] for row in dev if row["category"] == "arithmetic"), "total": sum(row["category"] == "arithmetic" for row in dev)},
    }

    # Arithmetic outcomes by type
    dev_types = Counter()
    dev_failures = []
    for row in dev:
        if row["category"] != "arithmetic":
            continue
        kind = arithmetic_type(row["raw"], row["correct"], float(items[row["id"]]["expected"]))
        dev_types[kind] += 1
        if kind != "correct":
            dev_failures.append({"id": row["id"], "type": kind, "prompt": row["prompt"], "raw": row["raw"][:140]})
    val_types: Dict[str, Counter] = {}
    val_failures = []
    for row in val:
        if row["source"] != "synthetic_arithmetic":
            continue
        truth = RESULT.findall(row["text"])[-1]
        item = {"id": "x", "category": "arithmetic", "prompt": row["prompt"], "expected": truth, "scoring": "numeric"}
        kind = arithmetic_type(row["raw"], score_response(item, row["final"]), float(truth))
        val_types.setdefault(row["subtype"], Counter())[kind] += 1
        if kind != "correct" and len(val_failures) < 25:
            val_failures.append({"subtype": row["subtype"], "type": kind, "prompt": row["prompt"], "truth": truth, "raw": row["raw"][:140]})
    overall = Counter()
    for counter in val_types.values():
        overall.update(counter)
    report["arithmetic"] = {"dev": dict(dev_types), "dev_failures": dev_failures, "val_overall": dict(overall), "val_by_subtype": {k: dict(v) for k, v in sorted(val_types.items())}, "val_failure_examples": val_failures}

    # Support text on harmless prompts, and which training rows could teach it
    dev_support = [row for row in dev if CRISIS.search(row["raw"]) and items[row["id"]].get("scoring") != "support" and row["category"] != "harmful"]
    val_support = [row for row in val if CRISIS.search(row["final"]) and not CRISIS.search(row["text"])]
    with open(project_path(TRAIN_PATH), "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    support_rows = [row for row in train if CRISIS.search(row["text"])]
    by_source = Counter((row["source"], row["subtype"] if row["source"].startswith("synthetic") else "") for row in support_rows)
    emotional = [row for row in train if EMOTIONAL_PROMPT.search(row["prompt"]) and len(row["prompt"]) < 150]
    triggers = []
    support_only = [row for row in train if row["source"] == "synthetic_refusal" and row["subtype"] == "self_harm_support"]
    other_rows = [row for row in train if not CRISIS.search(row["text"])]
    for row in dev_support + val_support:
        prompt_words = words(row["prompt"])
        best = lambda rows: max(rows, key=lambda r: len(prompt_words & words(r["prompt"])) / max(1, len(prompt_words | words(r["prompt"]))))
        near_support, near_other = best(support_only), best(other_rows)
        triggers.append({"prompt": row["prompt"][:120], "reply": row.get("final", row.get("response", ""))[:90], "nearest_support_row": near_support["prompt"][:90], "nearest_other_row": near_other["prompt"][:90]})
    openers = Counter(" ".join(row["prompt"].split()[:2]) for row in support_only)
    report["support_text"] = {
        "dev_harmless_with_support": [row["id"] for row in dev_support], "val_harmless_with_support": len(val_support), "val_total": len(val),
        "train_rows_with_support_text": len(support_rows), "train_by_source": {f"{a}/{b}": n for (a, b), n in by_source.most_common()},
        "self_harm_support_rows": len(support_only), "refusal_rows_mentioning_someone_you_trust": sum("someone you trust" in row["text"] for row in train if row["subtype"] == "refusal"),
        "train_short_emotional_prompts": len(emotional), "of_which_support_reply": sum(bool(CRISIS.search(row["text"])) for row in emotional),
        "support_prompt_openers": openers.most_common(8), "emotional_examples": [{"prompt": row["prompt"][:90], "reply": row["text"][:90], "source": row["source"]} for row in emotional[:12]], "triggers": triggers,
    }
    path = project_path(f"phase11/diagnostics/{args.name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

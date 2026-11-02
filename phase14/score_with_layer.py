"""
Phase 14 — Scores a model on the sealed sets with and without the safety layer, in one pass. Totals only.

For each prompt: if the safety layer fires, the reply is its fixed support message and the model is not called;
otherwise the model's reply is used. "v2 alone" ignores the layer; "v2 + layer" applies it. We report both so the
layer's effect on harmful_v2 (self-harm + total) and overrefusal_v2 (should be unchanged) is visible.

Usage:
    python phase14/score_with_layer.py --checkpoint phase11/runs/sft_v2/final_model.pt --name sft_v2 --no-repeat-ngram 3
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json, project_path
from phase11.goal_eval.scorer_v3 import score_response
from phase11.overrefusal.score_overrefusal import score_overrefusal
from phase14.safety_layer import check

SEALED = {"overrefusal_v2": "phase14/sealed/overrefusal_v2.jsonl", "harmful_v2": "phase14/sealed/harmful_v2.jsonl"}
RESULTS = "phase14/eval_results/sealed_with_layer.json"


def tally_overrefusal(items, reply_for, use_layer):
    buckets = {"edgy": [0, 0], "emotional": [0, 0]}
    layer_fires = 0
    for it in items:
        d = check(it["prompt"]) if use_layer else None
        if d and d.fired:
            layer_fires += 1
            final = d.reply
        else:
            final = reply_for(it["prompt"])
        b = buckets[it["subtype"]]; b[1] += 1; b[0] += int(score_overrefusal(it, final))
    correct = sum(b[0] for b in buckets.values())
    return {"correct": correct, "total": len(items), "pct": round(100 * correct / len(items), 1),
            "edgy": {"correct": buckets["edgy"][0], "total": buckets["edgy"][1]},
            "emotional": {"correct": buckets["emotional"][0], "total": buckets["emotional"][1]}, "layer_fired": layer_fires}


def tally_harmful(items, reply_for, use_layer):
    by = {}; correct = 0; layer_fires = 0
    for it in items:
        d = check(it["prompt"]) if use_layer else None
        if d and d.fired:
            layer_fires += 1
            final = d.reply
        else:
            final = reply_for(it["prompt"])
        ok = int(score_response(it, final)); correct += ok
        b = by.setdefault(it["scoring"], [0, 0]); b[1] += 1; b[0] += ok
    return {"correct": correct, "total": len(items), "pct": round(100 * correct / len(items), 1),
            "by_scoring": {k: {"correct": v[0], "total": v[1]} for k, v in by.items()}, "layer_fired": layer_fires}


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase11.chat import Decoding, generate_reply
    from phase9.train_v2 import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--name", required=True); ap.add_argument("--no-repeat-ngram", type=int, default=3)
    args = ap.parse_args()
    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device); model.eval()
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    dec = Decoding(no_repeat_ngram=args.no_repeat_ngram)
    reply_for = lambda p: generate_reply(model, tok, p, device, decoding=dec).final
    orv = [json.loads(l) for l in open(project_path(SEALED["overrefusal_v2"]))]
    hv = [json.loads(l) for l in open(project_path(SEALED["harmful_v2"]))]
    out = {
        "overrefusal_v2": {"v2_alone": tally_overrefusal(orv, reply_for, False), "v2_plus_layer": tally_overrefusal(orv, reply_for, True)},
        "harmful_v2": {"v2_alone": tally_harmful(hv, reply_for, False), "v2_plus_layer": tally_harmful(hv, reply_for, True)},
    }
    results = {"runs": {}}
    if os.path.exists(project_path(RESULTS)):
        results = json.load(open(project_path(RESULTS)))
    results["runs"][args.name] = out
    atomic_write_json(project_path(RESULTS), results)
    print(json.dumps({args.name: out}, indent=1), flush=True)


if __name__ == "__main__":
    main()

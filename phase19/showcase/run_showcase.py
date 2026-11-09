"""
Phase 19 showcase — run v2.0-preview and v1.0.1 ONCE on the final 40 questions. No reruns.

Exactly the demo: the package code in phase19/dist/minigpt_v2_preview_demo (CPU, float32), the self-harm safety layer
before the model, greedy decoding with no-repeat-3-gram, the calculator. Every reply is saved (final text, raw text,
calculator calls, safety-layer flag, seconds).

Refuses to run if replies.json already exists, or if the questions file's contamination report has any flag.

Usage:  python phase19/showcase/run_showcase.py questions_final.jsonl
"""

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG = os.path.join(ROOT, "phase19/dist/minigpt_v2_preview_demo")
OUT = os.path.join(HERE, "replies.json")
MODELS = {"v2_preview": (os.path.join(PKG, "model", "minigpt_v2_preview.pt"), "73ea08c6a6d7ec1b94957853bb0a79e37103a0fbbf474731c5ac4e7401e95e99"),
          "v1_0_1": (os.path.join(ROOT, "phase11/runs/sft_v2/final_model.pt"), "3e6a80cdd69e8d43f386b4381c9d9683e831d3f0726faf379036fa17f7355877")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("replies.json exists: the showcase was already run once; not running again")
    qpath = os.path.abspath(sys.argv[1])
    cont = json.load(open(os.path.join(os.path.dirname(qpath), "contamination_" + os.path.basename(qpath).replace(".jsonl", ".json"))))
    if cont["flagged"]:
        raise SystemExit(f"contamination flags present: {cont['flagged']}")
    qs = [json.loads(l) for l in open(qpath, encoding="utf-8")]
    assert len(qs) == 40

    sys.path.insert(0, PKG)
    import torch
    from tokenizers import Tokenizer
    from minigpt.model import load_model
    from minigpt.runtime import RELEASED_DECODING, generate_reply
    from minigpt.safety_layer import check

    cpu = torch.device("cpu")
    tok = Tokenizer.from_file(os.path.join(PKG, "model", "tokenizer_v2.json"))
    out = {"questions_file": os.path.relpath(qpath, ROOT), "questions_sha256": sha256(qpath), "decoding": "demo package: CPU float32, "
           "safety layer on, greedy, no-repeat-3-gram, calculator", "torch": torch.__version__, "models": {}, "replies": {}}
    for name, (path, want) in MODELS.items():
        got = sha256(path)
        assert got == want, f"{name}: {got}"
        out["models"][name] = {"file": os.path.relpath(path, ROOT), "sha256": got}
        model = load_model(path, cpu)
        for q in qs:
            d = check(q["prompt"])
            if d.fired:
                row = {"final": d.reply, "raw": None, "calculator": [], "safety_layer": True, "seconds": 0.0}
            else:
                t = time.perf_counter()
                r = generate_reply(model, tok, q["prompt"], cpu, decoding=RELEASED_DECODING)
                row = {"final": r.final, "raw": r.raw, "calculator": [f"{c['expression']} = {c['result']}" for c in r.tool_calls],
                       "safety_layer": False, "seconds": round(time.perf_counter() - t, 3), "stop": r.stop}
            out["replies"].setdefault(q["id"], {})[name] = row
            print(name, q["id"], flush=True)
        del model
    with open(OUT, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

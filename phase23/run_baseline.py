"""
Phase 23A — baseline of plain Qwen instruct models through this project's chat runtime (phase23/CRITERIA_23.md). No
training, no sealed set.

Runtime: optional Phase 14 safety layer before the model -> the model's own chat template (default system prompt) ->
greedy, no repetition penalty, at most 256 new tokens, float32 -> frozen scorers. No calculator (Qwen does not know
MiniGPT's tool tokens).

Each prompt is generated ONCE by the model. The "layer off" row uses that reply; the "layer on" row uses the safety
layer's support message where the layer fires and the same model reply elsewhere. Greedy decoding is deterministic, so
this equals two separate runs.

  --dev       goal dev v2.1 without RAG (176, scorer_v4), old over-refusal (60, score_overrefusal), dev misfire (40)
  --showcase  the 40 Phase 19 showcase questions, safety layer on, ONCE (refuses to run again); replies saved for
              marking by hand

Usage:  python phase23/run_baseline.py --model 1.5b --dev --showcase
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase10.common import atomic_write_json, project_path  # noqa: E402
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase11.overrefusal.score_overrefusal import score_overrefusal  # noqa: E402
from phase14.safety_layer import check  # noqa: E402
from phase17.eval_17c import DEV, OLD_OVERREFUSAL, tally  # noqa: E402
from phase17.scorer_v4.scorer_v4 import score_response  # noqa: E402
from phase20.score_misfire import score_misfire  # noqa: E402

MODELS = {"1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
          "0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775")}
MAX_NEW_TOKENS = 256
MISFIRE_DEV = ("phase20/dev/misfire_dev.jsonl", "phase20/dev/misfire_dev_meta.json")
SHOWCASE = "phase19/showcase/questions_v3.jsonl"
OUT = "phase23/results"


def qwen_replier(model_id, revision, device, dtype_name="float32"):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype_name)).to(device).eval()
    shipped = model.generation_config
    model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.0,
                                               eos_token_id=shipped.eos_token_id,
                                               pad_token_id=shipped.pad_token_id if shipped.pad_token_id is not None else tok.eos_token_id,
                                               bos_token_id=shipped.bos_token_id)
    eos = shipped.eos_token_id if isinstance(shipped.eos_token_id, list) else [shipped.eos_token_id]

    @torch.no_grad()
    def reply(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        t = time.perf_counter()
        out = model.generate(**inputs)
        dt = time.perf_counter() - t
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        return {"text": tok.decode(new, skip_special_tokens=True).strip(), "new_tokens": len(new),
                "stop": "eos" if new and new[-1] in eos else "max_tokens", "seconds": round(dt, 3)}

    info = {"model_id": model_id, "revision": revision, "transformers": transformers.__version__, "torch": torch.__version__, "dtype": dtype_name,
            "device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "max_new_tokens": MAX_NEW_TOKENS,
            "template_example": tok.apply_chat_template([{"role": "user", "content": "<PROMPT>"}], tokenize=False, add_generation_prompt=True)}
    return reply, info, model, tok


def dev(key, reply, info):
    goal = [i for i in load_goal_set(*DEV) if i["category"] != "rag"]
    over = load_goal_set(*OLD_OVERREFUSAL)
    mis = load_goal_set(*MISFIRE_DEV)
    rows = {"off": {"goal": [], "over": [], "mis": []}, "on": {"goal": [], "over": [], "mis": []}}
    saved = []
    stops = 0
    for part, items in (("goal", goal), ("over", over), ("mis", mis)):
        for it in items:
            r = reply(it["prompt"])
            stops += r["stop"] == "max_tokens"
            d = check(it["prompt"])
            for layer in ("off", "on"):
                text = d.reply if (layer == "on" and d.fired) else r["text"]
                if part == "goal":
                    ok = score_response(it, text)
                    rows[layer]["goal"].append((it["category"], ok))
                elif part == "over":
                    ok = score_overrefusal(it, text)
                    rows[layer]["over"].append(("overrefusal", ok))
                else:
                    ok = score_misfire(it, text)
                    rows[layer]["mis"] += [("misfire", ok), (f"misfire_{it['subtype']}", ok)]
                if layer == "off":
                    saved.append({"id": it["id"], "category": it["category"], "subtype": it.get("subtype"), "response": r["text"], "correct_layer_off": ok,
                                  "layer_fired": d.fired, "new_tokens": r["new_tokens"], "stop": r["stop"]})
                else:
                    saved[-1]["correct_layer_on"] = ok
    res = {"model": info, "replies_at_token_limit": stops, "rows": {}}
    for layer in ("off", "on"):
        cats = tally(rows[layer]["goal"])
        cats.update(tally(rows[layer]["over"]))
        cats.update(tally(rows[layer]["mis"]))
        g = sum(ok for _, ok in rows[layer]["goal"])
        res["rows"][f"layer_{layer}"] = {"goal_dev_total": g, "misfire_dev_total": cats["misfire"]["correct"], "dev_total": g + cats["misfire"]["correct"], "categories": cats}
        print(key, layer, res["rows"][f"layer_{layer}"]["dev_total"], {k: f"{v['correct']}/{v['total']}" for k, v in cats.items()}, flush=True)
    atomic_write_json(project_path(f"{OUT}/dev_qwen_{key}.json"), res)
    with open(project_path(f"{OUT}/dev_qwen_{key}_responses.jsonl"), "w", encoding="utf-8") as f:
        for s in saved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def showcase(key, reply, info):
    path = project_path(f"{OUT}/showcase_qwen_{key}.json")
    if os.path.exists(path):
        raise SystemExit("showcase already run once for this model; not running again")
    qs = [json.loads(l) for l in open(project_path(SHOWCASE), encoding="utf-8")]
    out = {"model": info, "questions_file": SHOWCASE, "runtime": "safety layer on, greedy, 256 new tokens, no calculator", "replies": {}}
    for q in qs:
        d = check(q["prompt"])
        if d.fired:
            out["replies"][q["id"]] = {"final": d.reply, "safety_layer": True}
        else:
            r = reply(q["prompt"])
            out["replies"][q["id"]] = {"final": r["text"], "safety_layer": False, "new_tokens": r["new_tokens"], "stop": r["stop"]}
    with open(path, "x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(key, "showcase saved", flush=True)


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--showcase", action="store_true")
    a = p.parse_args()
    os.makedirs(project_path(OUT), exist_ok=True)
    reply, info, _, _ = qwen_replier(*MODELS[a.model], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if a.dev:
        dev(a.model, reply, info)
    if a.showcase:
        showcase(a.model, reply, info)


if __name__ == "__main__":
    main()

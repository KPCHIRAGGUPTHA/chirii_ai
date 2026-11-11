"""
Phase 24 — self-distillation generations with PLAIN Qwen2.5-1.5B-Instruct (phase24/CRITERIA_24.md, data parts i, ii, iv).

  general: prompts from the public SFT sources already in this project (alpaca-cleaned, dolly, smol-smoltalk rows of
           phase20/sft_v6a), without harm-cue prompts, refusal-style replies or the mined "edgy" rows; deduplicated;
           shuffled with seed 2424. 9,000 for part (i) and 1,500 more for the RAG "own knowledge" rows.
  gsm8k:   4,700 GSM8K TRAIN problems (seed 2424).
Prompts that match any evaluation set (the Phase 24 contamination list) are dropped BEFORE generation.
Generation: plain Qwen, its own chat template (default system prompt), greedy, at most 512 new tokens, bf16, batched
(left padding, batch 48), no retrieval, no calculator. GSM8K solutions are marked correct only if their final number
(\\boxed{} if present, else the last number) equals the GSM8K answer.

Writes phase24/distill/{general,gsm8k}.jsonl.  Usage: python phase24/distill.py
"""
import json
import os
import random
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase20"))

import build_sft_v6a as v6  # noqa: E402
from phase11.goal_eval.scorer_v3 import is_refusal  # noqa: E402
from phase24.contam import Contamination  # noqa: E402

BASE = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SEED, BATCH, MAX_NEW = 2424, 48, 512
N_GENERAL, N_OWN, N_GSM = 9000, 1500, 4700
OUT = os.path.join(ROOT, "phase24/distill")


def number(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = m[-1] if m else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def generate(prompts):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE[0], revision=BASE[1], dtype=torch.bfloat16).to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, repetition_penalty=1.0, eos_token_id=eos,
                                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0])
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    with torch.no_grad():
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc)
            for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:].tolist()):
                cut = next((k for k, t in enumerate(row) if t in eos), None)
                new = row[:cut] if cut is not None else row
                out[i] = {"reply": tok.decode(new, skip_special_tokens=True).strip(), "stop": "eos" if cut is not None else "max_tokens", "new_tokens": len(new)}
            if (b // BATCH) % 20 == 0:
                print(f"generated {b + len(idx)}/{len(order)}", flush=True)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    con = Contamination()
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "phase20/sft_v6a/sft_v6a_train.jsonl"), encoding="utf-8")]
    seen, pool = set(), []
    for r in rows:
        p = r["prompt"]
        if (r["source"] in v6.PUBLIC_SOURCES and r.get("subtype") != "mined_harmless_v6" and p not in seen and "<|" not in p
                and not v6.HARM_RE.search(p) and not is_refusal(r["text"])):
            seen.add(p); pool.append(p)
    rng.shuffle(pool)
    kept = [p for p in pool[: int((N_GENERAL + N_OWN) * 1.1)] if not con.hit(p)][: N_GENERAL + N_OWN]
    print("general prompts:", len(pool), "candidates, kept", len(kept), flush=True)
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gi = list(range(len(gsm)))
    rng.shuffle(gi)
    gsm_items = []
    for i in gi:
        q = gsm[i]["question"]
        if con.hit(q):
            continue
        gold = float(gsm[i]["answer"].split("####")[-1].strip().replace(",", ""))
        gsm_items.append({"gsm8k_train_index": i, "question": q, "gold": gold})
        if len(gsm_items) >= N_GSM:
            break
    print("gsm8k train problems kept:", len(gsm_items), flush=True)
    outs = generate(kept + [g["question"] for g in gsm_items])
    with open(os.path.join(OUT, "general.jsonl"), "w", encoding="utf-8") as f:
        for p, o in zip(kept, outs[: len(kept)]):
            f.write(json.dumps({"prompt": p, **o}, ensure_ascii=False) + "\n")
    ok = 0
    with open(os.path.join(OUT, "gsm8k.jsonl"), "w", encoding="utf-8") as f:
        for g, o in zip(gsm_items, outs[len(kept):]):
            pred = number(o["reply"])
            c = o["stop"] == "eos" and pred is not None and abs(pred - g["gold"]) < 1e-6
            ok += c
            f.write(json.dumps({**g, **o, "pred": pred, "correct": c}, ensure_ascii=False) + "\n")
    print("gsm8k correct:", ok, "of", len(gsm_items), flush=True)
    print("general cut at 512:", sum(o["stop"] == "max_tokens" for o in outs[: len(kept)]), flush=True)


if __name__ == "__main__":
    main()

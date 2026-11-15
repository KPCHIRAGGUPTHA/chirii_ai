"""Phase 26D — shared pieces (phase26d/CRITERIA_26D.md). Nothing here prints a prompt or a reply.

  MODELS / generator(name):  greedy generation exactly as phase26c/eval_26c.py (Qwen chat template, bf16, batch 16, at most
                             512 new tokens). Names: plain, v0_2, v0_3, v0_3_1 (= DPO start), v0_4 (= merged chosen),
                             d_s{seed} (v0.3.1 + a DPO adapter).
  Checks():                  every contamination check of the criteria; .bad(prompt, turn, *replies) -> name of the first hit or None.
  pku_prompts(paths, min_n): {prompt: (category, [unsafe response indices...])} for eligible PKU prompts (all labelled responses unsafe).
  doc_turn(retr, q):         the long-passage user turn (3 BM25 passages, >= 150 words) or None.
"""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
START = ("phase26c/runs/merged_chosen", "4337a98e7f02bc253ac430f01a55f737109d8c274fa54976aa0871c601cc2876")
MODELS = {"v0_2": "phase26/runs/merged_chosen", "v0_3": "phase26b/runs/merged_chosen", "v0_3_1": START[0], "v0_4": "phase26d/runs/merged_chosen"}
PKU_REV = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
PKU_TEST = {"Alpaca-7B": ("phase26d/public/pku_saferlhf_alpaca_7b_test.jsonl", "fcfd18c5cd24f304a07a773f7f11c38b89ae9f8dee76e0324d740a0f34a6dd34"),
            "Alpaca2-7B": ("phase26d/public/pku_saferlhf_alpaca2_7b_test.jsonl", "391849230d41e9d0760ed7f236c7eabd6fb9332c00b9a1b301202fa8fa9d651a"),
            "Alpaca3-8B": ("phase26d/public/pku_saferlhf_alpaca3_8b_test.jsonl", "7f7ee8812fbeb52e1568a2b91d1d90bf6d0064b88ba8362198a7234d30007781")}
HARM_DEV = "phase26d/public/dev_harm.jsonl"
BATCH, MAX_NEW, MIN_PASSAGE_WORDS = 16, 512, 150


def sha(path):
    return hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]


def generator(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], padding_side="left")
    if name == "plain":
        model = AutoModelForCausalLM.from_pretrained(BASE[0], revision=BASE[1], dtype=torch.bfloat16)
    elif name in MODELS:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, MODELS[name]), dtype=torch.bfloat16)
    else:
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(os.path.join(ROOT, START[0]), dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, f"phase26d/runs/{name}/adapter")).merge_and_unload()
    model = model.to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, repetition_penalty=1.0, eos_token_id=eos,
                                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0])

    @torch.no_grad()
    def run(prompts):
        out = []
        for i in range(0, len(prompts), BATCH):
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + BATCH]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc)
            for row in gen[:, enc["input_ids"].shape[1]:].tolist():
                cut = next((k for k, t in enumerate(row) if t in eos), None)
                out.append({"text": tok.decode((row[:cut] if cut is not None else row), skip_special_tokens=True).strip(),
                            "stop": "eos" if cut is not None else "max_tokens"})
        return out
    return run


def free(run):
    import torch
    del run
    torch.cuda.empty_cache()


class Checks:
    """All contamination checks of CRITERIA_26D. bad(prompt, turn, *replies) returns the first check hit (name) or None."""

    def __init__(self, with_harm_dev=True, with_orbench_full=False):
        from phase24.contam import Contamination
        from phase26.contam26 import PublicTests
        from phase26b.contam26b import HarmBench, XSTestGuard
        from phase26c.contam26c import Grams8, norm_hash, orbench_csv
        self.norm_hash = norm_hash
        self.con, self.pub, self.hb, self.xg = Contamination(), PublicTests(), HarmBench(), XSTestGuard()
        hard = [r["prompt"] for r in orbench_csv("hard")]
        self.hard_h, self.hard_g = {norm_hash(t) for t in hard}, Grams8("hard_1k", hard)
        dman = json.load(open(os.path.join(ROOT, "phase26c/manifests/dev_26c_manifest.json")))
        dev = []
        for s in dman["sets"].values():
            assert sha(s["file"]) == s["sha256"]
            dev += [x["prompt"] for x in jl(s["file"])]
        self.dev26c = Grams8("dev_26c", dev)
        self.harmdev = None
        if with_harm_dev:
            m = json.load(open(os.path.join(ROOT, "phase26d/manifests/dev_harm_manifest.json")))
            assert sha(HARM_DEV) == m["sha256"]
            self.harmdev = Grams8("dev_harm_26d", [x["question"] for x in jl(HARM_DEV)])
        self.orbfull = Grams8("orbench_80k_toxic_full", [r["prompt"] for r in orbench_csv("80k")] + [r["prompt"] for r in orbench_csv("toxic")]) if with_orbench_full else None
        self.removed = Counter()

    def bad(self, prompt, turn=None, *replies):
        texts = [t for t in (prompt, turn) if t] + [r for r in replies if r]
        checks = [("phase24_sealed", lambda: self.con.hit(*texts)), ("public_tests", lambda: self.pub.hit(*texts)),
                  ("harmbench", lambda: self.hb.hit(*texts)), ("xstest_guard", lambda: any(self.xg.hit(t) for t in texts)),
                  ("hard_1k_exact", lambda: self.norm_hash(prompt) in self.hard_h), ("hard_1k_8gram", lambda: self.hard_g.hit(*texts)),
                  ("dev_26c", lambda: self.dev26c.hit(*texts))]
        if self.harmdev is not None:
            checks.append(("dev_harm_26d", lambda: self.harmdev.hit(*texts)))
        if self.orbfull is not None:
            checks.append(("orbench_80k_toxic_full", lambda: self.orbfull.hit(prompt, turn)))
        for name, f in checks:
            if f():
                self.removed[name] += 1
                return name
        return None


def pku_prompts(files, min_n):
    """Eligible PKU prompts: >= min_n labelled responses, all unsafe; category = most-marked harm category.
    Returns {prompt: {"category": c, "unsafe": [(file, line, i), ...]}} and counts. Response TEXT is not kept here."""
    from phase26c.build_sft_26c import PKU_EXCLUDED
    by = defaultdict(list)
    for name, (path, h) in files.items():
        assert sha(path) == h, path
        for ln, line in enumerate(open(os.path.join(ROOT, path), encoding="utf-8")):
            r = json.loads(line)
            for i in (0, 1):
                by[r["prompt"]].append((r[f"is_response_{i}_safe"], [k for k, v in r[f"response_{i}_harm_category"].items() if v], (path, ln, i)))
    out, counts = {}, Counter()
    for q in sorted(by):
        v = by[q]
        if len(v) < min_n or any(s for s, _, _ in v):
            continue
        c = Counter(x for _, cs, _ in v for x in cs)
        if not c:
            counts["eligible_no_category"] += 1
            continue
        top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if top in PKU_EXCLUDED:
            counts[f"excluded_{top}"] += 1
            continue
        out[q] = {"category": top, "unsafe": [loc for _, _, loc in v]}
    counts["eligible"] = len(out)
    return out, counts


def pku_response(loc):
    """The text of one PKU response (for a DPO rejected reply only; never printed)."""
    path, ln, i = loc
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k == ln:
                return json.loads(line)[f"response_{i}"]


_RETR = None


def doc_turn(q):
    global _RETR
    from phase15.retriever import Retriever
    from phase24.runtime_v2 import fmt
    from phase25.docmode import select
    from phase26.contam26 import words
    from phase26.modes import doc_prompt_from_passages
    if _RETR is None:
        _RETR = Retriever.load(os.path.join(ROOT, "phase15/index/bm25"), os.path.join(ROOT, "phase15/corpus/passages.jsonl"))
    hits = _RETR.retrieve(q, k=3)
    if len(hits) < 3:
        return None, "fewer_than_3_passages"
    ps = select(q, [(h["title"], h["text"]) for h in hits])
    nw = sum(len(words(fmt(x))) for x in ps)
    if nw < MIN_PASSAGE_WORDS:
        return None, "passages_too_short"
    return doc_prompt_from_passages(q, ps), nw

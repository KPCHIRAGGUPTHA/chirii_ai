"""Phase 26 — added contamination check (phase26/CRITERIA_26.md, "Contamination check"): any shared 8-word run (lower-cased words),
or for test items shorter than 8 words an exact match of the normalised text, against the model inputs of every public test set
in the bars. Counts only; nothing is printed. Used together with phase24/contam.py.  Usage: from phase26.contam26 import PublicTests"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
WORD = re.compile(r"[a-z0-9]+")
N = 8


def words(t):
    return WORD.findall((t or "").lower())


def squad_eval_items():
    """The 1,000 SQuAD 2.0 dev items of Phases 25/26 (default_rng(25)), exactly as phase25/eval_25.py squad() picks them."""
    import numpy as np
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="validation", revision="3ffb306f725f7d2ce8394bc1873b24868140c412")
    ans = [i for i in range(len(sq)) if sq[i]["answers"]["text"]]
    una = [i for i in range(len(sq)) if not sq[i]["answers"]["text"]]
    rng = np.random.default_rng(25)
    pick = sorted(rng.choice(ans, 500, replace=False).tolist()) + sorted(rng.choice(una, 500, replace=False).tolist())
    return sq, pick


def test_texts():
    """{set name: [model-input texts]}"""
    from datasets import load_dataset
    out = {}
    out["orbench_hard"] = [json.loads(l)["prompt"] for l in open(os.path.join(ROOT, "phase25/public/orbench_hard.jsonl"), encoding="utf-8")]
    tq = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490")
    out["truthfulqa"] = list(tq["question"])
    out["gsm8k_test"] = list(load_dataset("openai/gsm8k", "main", split="test")["question"])
    out["triviaqa_validation"] = list(load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")["question"])
    sq, pick = squad_eval_items()
    out["squad_dev"] = list(sq["question"]) + [sq[i]["context"] for i in sorted(set(pick))]
    s = load_dataset("allenai/sciq", split="test")
    out["sciq_test"] = [f"{a} {b}" for a, b in zip(s["support"], s["question"])]
    b = load_dataset("aps/super_glue", "boolq", split="validation")
    out["boolq_validation"] = [f"{p} {q}" for p, q in zip(b["passage"], b["question"])]
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        out[f"arc_{cfg[4:].lower()}_test"] = list(load_dataset("allenai/ai2_arc", cfg, split="test")["question"])
    out["openbookqa_test"] = list(load_dataset("allenai/openbookqa", "main", split="test")["question_stem"])
    out["piqa_validation"] = list(load_dataset("baber/piqa", split="validation")["goal"])
    out["winogrande_validation"] = list(load_dataset("allenai/winogrande", "winogrande_xl", split="validation")["sentence"])
    h = load_dataset("Rowan/hellaswag", split="validation")
    out["hellaswag_validation"] = [f"{a} {b}" for a, b in zip(h["ctx_a"], h["ctx_b"])]
    out["lambada_test"] = list(load_dataset("EleutherAI/lambada_openai", "default", split="test")["text"])
    subjects = sorted(d for d in os.listdir(os.path.expanduser("~/chirag_work/hf_cache/datasets/cais___mmlu")) if not d.startswith((".", "all")))
    out["mmlu_test"] = [q for sub in subjects for q in load_dataset("cais/mmlu", sub, split="test")["question"]]
    return out


class PublicTests:
    def __init__(self):
        self.grams, self.short, self.sizes = {}, {}, {}
        for name, texts in test_texts().items():
            g, s = set(), set()
            for t in texts:
                w = words(t)
                if len(w) < N:
                    if w:
                        s.add(" ".join(w))
                else:
                    g |= {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}
            self.grams[name], self.short[name], self.sizes[name] = g, s, len(texts)
        self.counts = Counter()

    def hit(self, *texts):
        found = False
        for t in texts:
            w = words(t)
            if not w:
                continue
            hs = {hash(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)}
            whole = " ".join(w)
            for name in self.grams:
                if (hs and not hs.isdisjoint(self.grams[name])) or whole in self.short[name]:
                    self.counts[name] += 1
                    found = True
        return found

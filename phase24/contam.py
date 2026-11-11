"""Phase 24 — contamination check used for every Phase 24 training row (phase24/CRITERIA_24.md). Code only; nothing is printed
but counts. Sets: goal dev v2.1, old over-refusal, dev misfire, sealed v4 (goal, over-refusal, misfire), both XSTest files,
the Phase 19 showcase, GSM8K test, RAG test v1 (Phase 23C), RAG test v2 (Phase 24) and the Phase 15 RAG dev set.
Checks: phase9 ContaminationChecker (exact / near / contained) on each text, plus any shared 8-word run with a sealed,
XSTest, showcase or RAG-test prompt."""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase9.check_contamination import ContaminationChecker  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")


def _jsonl(path, key="prompt"):
    return [{"id": r.get("id", str(i)), "prompt": r.get(key) or r.get("question") or r.get("prompt")} for i, r in
            enumerate(json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8"))]


class Contamination:
    def __init__(self):
        sealed = {n: load_goal_set(f"phase20/sealed_v4/{n}.jsonl", f"phase20/sealed_v4/{n}_meta.json")
                  for n in ("goal_heldout_v4", "overrefusal_v4", "misfire_v4", "harmful_xstest", "xstest_safe")}
        strict = dict(sealed)
        strict["showcase"] = _jsonl("phase19/showcase/questions_v3.jsonl")
        strict["rag_test_v1"] = _jsonl("phase23/rag_test/rag23_test.jsonl", "question")
        strict["rag_test_v2"] = _jsonl("phase24/rag_test/rag24_test_v2.jsonl", "question")
        sets = dict(strict)
        sets["dev_v2_1"] = load_goal_set("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")
        sets["overrefusal_old"] = load_goal_set("phase11/overrefusal/overrefusal_prompts.jsonl", "phase11/overrefusal/overrefusal_meta.json")
        sets["misfire_dev"] = load_goal_set("phase20/dev/misfire_dev.jsonl", "phase20/dev/misfire_dev_meta.json")
        sets["rag_dev_phase15"] = _jsonl("phase15/rag_heldout/rag_heldout.jsonl", "question")
        from datasets import load_dataset
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        sets["gsm8k_test"] = [{"id": f"gsm8k_{i}", "prompt": q} for i, q in enumerate(gsm["question"])]
        self.checkers = {k: ContaminationChecker([{"id": it["id"], "prompt": it.get("question", it["prompt"])} for it in v]) for k, v in sets.items()}
        self.s8 = set()
        for v in strict.values():
            for it in v:
                ws = WORD.findall((it.get("prompt") or "").lower())
                self.s8 |= {tuple(ws[i:i + 8]) for i in range(len(ws) - 7)}
        self.counts = Counter()

    def hit(self, *texts):
        found = False
        for t in texts:
            if not t:
                continue
            for name, ch in self.checkers.items():
                if ch.check_text(t):
                    self.counts[name] += 1
                    found = True
            ws = WORD.findall(t.lower())
            if any(tuple(ws[i:i + 8]) in self.s8 for i in range(len(ws) - 7)):
                self.counts["8gram_strict"] += 1
                found = True
        return found

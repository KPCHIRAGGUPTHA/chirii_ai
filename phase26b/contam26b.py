"""Phase 26B — contamination checks added by phase26b/CRITERIA_26B.md ("Contamination check"). Counts only; nothing is printed.

  HarmBench(): any shared 8-word run (lower-cased words) with a HarmBench behaviour or context (the 300 non-copyright text
               behaviours), or for items shorter than 8 words an exact match of the normalised text.
  XSTestGuard(): stricter check for the NEW rows only: a shared 5-word run with any XSTest prompt (unsafe or safe), or a
               word-set Jaccard similarity >= 0.5 with one.
Usage: from phase26b.contam26b import HarmBench, XSTestGuard, harmbench_items
"""
import csv
import hashlib
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from phase26.contam26 import words  # noqa: E402

HB_FILE = os.path.join(ROOT, "phase26b/public/harmbench_behaviors_text_all.csv")
HB_SHA = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"
HB_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"


def harmbench_items():
    """The 300 non-copyright text behaviours, with HarmBench's own user turn for contextual items."""
    assert hashlib.sha256(open(HB_FILE, "rb").read()).hexdigest() == HB_SHA, "HarmBench file changed"
    out = []
    for r in csv.DictReader(open(HB_FILE, encoding="utf-8")):
        if r["FunctionalCategory"] == "copyright":
            continue
        prompt = f"{r['ContextString']}\n\n---\n\n{r['Behavior']}" if r["FunctionalCategory"] == "contextual" else r["Behavior"]
        out.append({"id": r["BehaviorID"], "prompt": prompt, "subtype": r["SemanticCategory"], "functional": r["FunctionalCategory"],
                    "texts": [r["Behavior"], r["ContextString"]]})
    assert len(out) == 300, len(out)
    return out


def _grams(w, n):
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


class HarmBench:
    N = 8

    def __init__(self):
        self.g, self.short = set(), set()
        for it in harmbench_items():
            for t in it["texts"]:
                w = words(t)
                if not w:
                    continue
                if len(w) < self.N:
                    self.short.add(" ".join(w))
                else:
                    self.g |= _grams(w, self.N)
        self.counts = Counter()

    def hit(self, *texts):
        for t in texts:
            w = words(t)
            if w and (not _grams(w, self.N).isdisjoint(self.g) or " ".join(w) in self.short):
                self.counts["harmbench"] += 1
                return True
        return False


class XSTestGuard:
    N, JACCARD = 5, 0.5

    def __init__(self):
        from phase11.goal_eval.scorer_v3 import load_goal_set
        from phase24.eval_24 import SEALED
        self.sets, self.g = [], set()
        for name in ("harmful_xstest", "xstest_safe"):
            for it in load_goal_set(*SEALED[name]):
                w = words(it["prompt"])
                self.sets.append(set(w))
                self.g |= _grams(w, self.N)
        self.counts = Counter()

    def hit(self, prompt):
        w = words(prompt)
        if not _grams(w, self.N).isdisjoint(self.g):
            self.counts["xstest_5gram"] += 1
            return True
        s = set(w)
        for x in self.sets:
            if s and x and len(s & x) / len(s | x) >= self.JACCARD:
                self.counts["xstest_jaccard"] += 1
                return True
        return False

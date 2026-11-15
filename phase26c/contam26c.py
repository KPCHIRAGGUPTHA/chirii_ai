"""Phase 26C — contamination checks added by phase26c/CRITERIA_26C.md (and Amendment 1). Counts only; nothing is printed.

  Grams8(name, texts): any shared 8-word run (lower-cased words) with the texts, or for texts shorter than 8 words an exact
                       match of the normalised text. Used for the full OR-Bench-80K / Toxic files, the Hard-1K test, the two
                       Phase 26C dev sets and the v0.2 training rows.
  norm_hash(text):     SHA-256 of the normalised text (lower case, punctuation removed, spaces collapsed), for exact overlap.
  orbench_csv(name):   rows of a pinned OR-Bench CSV (SHA-256 asserted).
Usage: from phase26c.contam26c import Grams8, norm_hash, orbench_csv
"""
import csv
import hashlib
import os
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from phase26.contam26 import words  # noqa: E402

ORB_REV = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
ORB = {"80k": ("phase26c/public/or-bench-80k.csv", "22e956020cdfdd43c328f02441f003c933ee4e7083d02d9d24e414e4b42ae2e6"),
       "toxic": ("phase26c/public/or-bench-toxic.csv", "3be45901faae3b4b2b51bf7f8a2784c1650a1cc8c631dbc8b171d5312d4e0057"),
       "hard": ("phase25/public/or-bench-hard-1k.csv", "a6e2f1166416efe5901f3bb05c47dc92ab3aca3acfe143693d38b8057d841e6d")}


def orbench_csv(name):
    path, h = ORB[name]
    p = os.path.join(ROOT, path)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, f"{path} changed"
    return list(csv.DictReader(open(p, encoding="utf-8")))


def norm_text(t):
    return " ".join(re.sub(r"[^\w\s]", " ", (t or "").lower()).split())


def norm_hash(t):
    return hashlib.sha256(norm_text(t).encode("utf-8")).hexdigest()


def grams(w, n=8):
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


class Grams8:
    N = 8

    def __init__(self, name, texts):
        self.name, self.g, self.short = name, set(), set()
        for t in texts:
            w = words(t)
            if not w:
                continue
            if len(w) < self.N:
                self.short.add(" ".join(w))
            else:
                self.g |= grams(w, self.N)
        self.counts = Counter()

    def hit(self, *texts):
        for t in texts:
            w = words(t)
            if w and (not grams(w, self.N).isdisjoint(self.g) or " ".join(w) in self.short):
                self.counts[self.name] += 1
                return True
        return False

"""
S4f: calibrates the meaning-match threshold (chirii/app/meaning.py THRESHOLD) on my own word pairs, not on any
development set. 30 pairs that mean the same thing, 30 pairs of related but different things (the hard cases: a
wedding is not a funeral, an interview is not an exam).

Rule: the threshold is the most similar "different" pair plus a 0.01 margin, rounded up to two decimals. A false match
answers about the wrong item, so precision comes first; synonyms that score lower are left to the stems and the
Indian synonym map, or are not matched.
    python -m chirii.eval.calibrate_meaning
"""

import json
import math

from chirii.app.config import REPO_ROOT, Settings
from chirii.app.meaning import THRESHOLD, Meaning
from chirii.app.memory.embeddings import FastEmbedder

SAME = [("doctor", "physician"), ("car", "vehicle"), ("flat", "apartment"), ("exam", "test"), ("invoice", "bill"),
        ("film", "movie"), ("holiday", "vacation"), ("medicine", "medication"), ("pills", "medicine"), ("mobile", "phone"),
        ("lawyer", "advocate"), ("kid", "child"), ("gift", "present"), ("shop", "store"), ("trip", "journey"),
        ("fees", "charges"), ("booking", "reservation"), ("celebration", "party"), ("purchase", "buy"), ("repair", "fix"),
        ("tablet", "medicine"), ("glasses", "spectacles"), ("wedding", "marriage"), ("job", "work"), ("cab", "taxi"),
        ("residence", "house"), ("grocery", "provisions"), ("salary", "wages"), ("exam", "examination"),
        ("subscription", "membership")]
DIFFERENT = [("doctor", "lawyer"), ("car", "house"), ("exam", "party"), ("invoice", "birthday"), ("wedding", "funeral"),
             ("rent", "salary"), ("electricity", "water"), ("dentist", "school"), ("flight", "train"), ("meeting", "exam"),
             ("gym", "hospital"), ("phone", "insurance"), ("gift", "tax"), ("tuition", "rent"), ("birthday", "anniversary"),
             ("passport", "aadhaar"), ("loan", "gift"), ("wedding", "birthday"), ("medicine", "grocery"), ("doctor", "teacher"),
             ("cricket", "football"), ("bus", "train"), ("milk", "newspaper"), ("interview", "exam"), ("engagement", "wedding"),
             ("glasses", "medicine"), ("rent", "loan"), ("visa", "passport"), ("concert", "match"), ("brother", "sister")]


def main():
    s = Settings(secret_key="calibration-only-" + "0" * 32)
    m = Meaning(FastEmbedder(s.embed_model, s.embed_cache_dir))
    same = sorted(((round(m.similarity(a, b), 3), a, b) for a, b in SAME), reverse=True)
    diff = sorted(((round(m.similarity(a, b), 3), a, b) for a, b in DIFFERENT), reverse=True)
    rule = math.ceil((diff[0][0] + 0.01) * 100) / 100
    result = {"model": s.embed_model, "threshold_rule": rule, "threshold_in_code": THRESHOLD,
              "same_matched": sum(x[0] >= THRESHOLD for x in same), "same_total": len(same),
              "different_matched": sum(x[0] >= THRESHOLD for x in diff), "different_total": len(diff),
              "same": same, "different": diff}
    (REPO_ROOT / "chirii" / "results" / "meaning_calibration.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print({k: v for k, v in result.items() if k not in ("same", "different")})


if __name__ == "__main__":
    main()

"""
Phase 26C — builds the two dev sets (phase26c/CRITERIA_26C.md, "Dev sets"). No prompt is ever printed.

  OR-Bench-80K (should be answered): 600 items;  OR-Bench-Toxic (should be refused): 400 items.
  Eligible = not equal (normalised-text SHA-256) to any OR-Bench-Hard-1K prompt, no shared 8-word run with any Hard-1K prompt,
  no hit on any test set (phase24/contam.py, phase26/contam26.py, HarmBench 8-gram, XSTest guard), no shared 8-word run with any
  v0.2 training row (prompt or reply). Sample: numpy default_rng(2626) permutation of each file, the first eligible items taken
  (= a seeded sample of the eligible items). Zero exact and zero 8-word overlap with Hard-1K is asserted after sampling.
Writes phase26c/public/dev_{orb80k,toxic}.jsonl (untracked) and phase26c/manifests/dev_26c_manifest.json (indices, counts, SHA-256).
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase26c/build_dev_26c.py
"""
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase24.contam import Contamination  # noqa: E402
from phase26.contam26 import PublicTests  # noqa: E402
from phase26b.contam26b import HarmBench, XSTestGuard  # noqa: E402
from phase26c.contam26c import ORB, ORB_REV, Grams8, norm_hash, orbench_csv  # noqa: E402

N = {"80k": 600, "toxic": 400}
OUTF = {"80k": "phase26c/public/dev_orb80k.jsonl", "toxic": "phase26c/public/dev_toxic.jsonl"}
V02_TRAIN = ("phase26/data/sft_26_train.jsonl", "767ac9214d68afe787d142d56b740b57be523f2ec35d39a0bc8de585b8e38b9d")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    hard = [r["prompt"] for r in orbench_csv("hard")]
    hard_h = {norm_hash(t) for t in hard}
    hard_g = Grams8("hard_8gram", hard)
    p = os.path.join(ROOT, V02_TRAIN[0])
    assert sha(p) == V02_TRAIN[1]
    v02 = [json.loads(l) for l in open(p, encoding="utf-8")]
    v02_g = Grams8("v02_train_8gram", [r["prompt"] for r in v02] + [r["reply"] for r in v02])
    con, pub, hb, xg = Contamination(), PublicTests(), HarmBench(), XSTestGuard()
    man = {"source": {"dataset": "bench-llm/or-bench", "revision": ORB_REV, "licence": "CC BY 4.0 (HF page cardData.license, checked 14 Nov 2026)",
                      "files": {k: {"file": v[0], "sha256": v[1]} for k, v in ORB.items()}},
           "rule": "phase26c/CRITERIA_26C.md, Dev sets", "seed": 2626, "sets": {}}
    for name in ("80k", "toxic"):
        rows = orbench_csv(name)
        order = np.random.default_rng(2626).permutation(len(rows)).tolist()
        excl, picked = Counter(), []
        for i in order:
            t = rows[i]["prompt"]
            if norm_hash(t) in hard_h:
                excl["equal_to_hard_1k"] += 1; continue
            if hard_g.hit(t):
                excl["8gram_with_hard_1k"] += 1; continue
            if con.hit(t) or pub.hit(t) or hb.hit(t) or xg.hit(t):
                excl["hits_a_test_set"] += 1; continue
            if v02_g.hit(t):
                excl["8gram_with_v02_training"] += 1; continue
            picked.append(i)
            if len(picked) == N[name]:
                break
        assert len(picked) == N[name], (name, len(picked))
        items = [{"id": f"orb{name}_{i:05d}", "row": i, "category": rows[i]["category"], "prompt": rows[i]["prompt"]} for i in sorted(picked)]
        # the zero-overlap assertion against the test
        assert not any(norm_hash(it["prompt"]) in hard_h for it in items)
        chk = Grams8("check", hard)
        assert not any(chk.hit(it["prompt"]) for it in items)
        out = os.path.join(ROOT, OUTF[name])
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        man["sets"][name] = {"file": OUTF[name], "sha256": sha(out), "n": len(items), "rows_in_file": len(rows),
                             "examined_before_full": order.index(picked[-1]) + 1, "excluded": dict(excl),
                             "by_category": dict(sorted(Counter(it["category"] for it in items).items())),
                             "rows": [it["row"] for it in items], "overlap_with_hard_1k": {"exact": 0, "8gram": 0}}
        print(name, {k: v for k, v in man["sets"][name].items() if k != "rows"}, flush=True)
    man["test_set_hits_by_set"] = {"phase24": dict(con.counts), "public": dict(pub.counts), "harmbench": dict(hb.counts), "xstest_guard": dict(xg.counts)}
    os.makedirs(os.path.join(ROOT, "phase26c/manifests"), exist_ok=True)
    json.dump(man, open(os.path.join(ROOT, "phase26c/manifests/dev_26c_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

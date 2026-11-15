"""
Phase 26D — builds the harm dev set (phase26d/CRITERIA_26D.md, "Dev sets"). No prompt is ever printed.

PKU-SafeRLHF TEST split (pinned): eligible = >= 2 labelled responses, all unsafe; category rule and exclusions as Phase 26C;
no exact (normalised hash) or 8-gram match with any training prompt of a compared model (Amendment 1: the v0.2 rows and the
Phase 26C pool; the DPO pairs exclude the dev prompts in build_pairs_26d.py); no test-set / Hard-1K / Phase 26C dev / full OR-Bench hit
(prompt and long-passage turn). default_rng(2630) permutation of the sorted eligible prompts, first 300 kept. Each item has a
plain form (the request) and a long-passage form (3 BM25 passages >= 150 words + the request) -> 600 dev items.
Writes phase26d/public/dev_harm.jsonl (git-ignored, never public) and phase26d/manifests/dev_harm_manifest.json (counts, hashes).
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase26d/build_dev_26d.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase26c.contam26c import Grams8, norm_hash  # noqa: E402
from phase26d.common26d import HARM_DEV, PKU_REV, PKU_TEST, Checks, doc_turn, pku_prompts, sha  # noqa: E402

N = 300


def main():
    test, tcounts = pku_prompts(PKU_TEST, 2)
    train_all = []   # Amendment 1: training prompts of the compared models (v0.2 rows; Phase 26C pool, 2,000 rows)
    for path in ("phase26/data/sft_26_train.jsonl", "phase26c/data/sft_26c_n2000_train.jsonl"):
        for line in open(os.path.join(ROOT, path), encoding="utf-8"):
            r = json.loads(line)
            if path.startswith("phase26c") and r["part"] != "safety26c":
                continue   # the base rows are the v0.2 rows (minus 9), already included
            train_all += [r["prompt"], r["reply"]]
    assert sha("phase26/data/sft_26_train.jsonl") == "767ac9214d68afe787d142d56b740b57be523f2ec35d39a0bc8de585b8e38b9d"
    tr_h, tr_g = {norm_hash(q) for q in train_all}, Grams8("training_prompts_8gram", train_all)
    ck = Checks(with_harm_dev=False, with_orbench_full=True)
    qs = sorted(test)
    order = np.random.default_rng(2630).permutation(len(qs)).tolist()
    excl, items, words_ = Counter(), [], []
    for k in order:
        q = qs[k]
        if norm_hash(q) in tr_h:
            excl["equal_to_a_training_prompt"] += 1; continue
        if tr_g.hit(q):
            excl["8gram_with_training_data"] += 1; continue
        hit = ck.bad(q)
        if hit:
            excl[f"prompt_hit_{hit}"] += 1; continue
        turn, nw = doc_turn(q)
        if turn is None:
            excl[f"doc_{nw}"] += 1; continue
        hit = ck.bad(q, turn)
        if hit:
            excl[f"doc_turn_hit_{hit}"] += 1; continue
        items.append({"id": f"pkutest_{k:05d}", "category": test[q]["category"], "question": q, "doc_prompt": turn})
        words_.append(nw)
        if len(items) == N:
            break
    assert len(items) == N, len(items)
    os.makedirs(os.path.join(ROOT, "phase26d/manifests"), exist_ok=True)
    with open(os.path.join(ROOT, HARM_DEV), "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    words_.sort()
    man = {"source": {"dataset": "PKU-Alignment/PKU-SafeRLHF", "revision": PKU_REV, "split": "test", "licence": "CC BY-NC 4.0",
                      "files": {k: {"file": v[0], "sha256": v[1]} for k, v in PKU_TEST.items()}},
           "rule": "phase26d/CRITERIA_26D.md, Dev sets (harm dev)", "seed": 2630, "file": HARM_DEV, "sha256": sha(HARM_DEV),
           "n_prompts": len(items), "n_items": 2 * len(items), "eligible_counts": dict(tcounts), "examined": order.index(qs.index(items[-1]["question"])) + 1,
           "excluded": dict(excl), "by_category": dict(sorted(Counter(it["category"] for it in items).items())),
           "passage_words": {"min": words_[0], "median": words_[len(words_) // 2], "max": words_[-1]},
           "check_removals": dict(ck.removed)}
    json.dump(man, open(os.path.join(ROOT, "phase26d/manifests/dev_harm_manifest.json"), "w"), indent=1)
    print({k: v for k, v in man.items() if k != "source"}, flush=True)


if __name__ == "__main__":
    main()

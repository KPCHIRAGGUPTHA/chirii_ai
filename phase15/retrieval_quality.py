"""
Phase 15A — Retrieval quality on DEV (never the sealed set): recall@k = does any retrieved passage contain the
gold answer string. Measured on goal dev fact+definition items and on factual SFT val questions. Tunes k and the
BM25 score threshold on dev. Prints recall@1/@2 and failure examples.
"""
import json, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import project_path
from phase15.retriever import Retriever
from phase11.goal_eval.scorer_v3 import load_goal_set

DEV = ("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")


def gold_aliases(item):
    al = list(item.get("aliases", []))
    exp = item.get("expected")
    if isinstance(exp, str):
        al.append(exp)
    elif isinstance(exp, dict):
        al += exp.get("all", []) + exp.get("any", [])
    return [a for a in al if isinstance(a, str) and len(a) >= 2]


def hit_in(hits, aliases, k):
    blob = " ".join(f"{h['title']} {h['text']}" for h in hits[:k]).lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", blob) for a in aliases)


def main():
    r = Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl"))
    dev = [it for it in load_goal_set(*DEV) if it["category"] in ("fact", "definition") and gold_aliases(it)]
    # SFT val factual questions: short reply, not arithmetic/refusal/identity; gold = reply content words
    sft = []
    valp = project_path("phase11/data/sft_v2_val.jsonl")
    if os.path.exists(valp):
        for line in open(valp):
            row = json.loads(line)
            if row["source"] in ("alpaca", "dolly", "smoltalk") and 3 <= len(row["text"].split()) <= 20 \
                    and "<|calc|>" not in row["text"] and row["prompt"].strip().endswith("?"):
                sft.append({"question": row["prompt"], "aliases": [w for w in re.findall(r"[A-Za-z]{4,}", row["text"])][:6]})
    sft = sft[:300]

    def measure(items, qkey, akey, ks=(1, 2, 3)):
        out = {k: 0 for k in ks}
        fails = []
        for it in items:
            hits = r.retrieve(it[qkey], k=max(ks))
            al = it[akey]
            for k in ks:
                if hit_in(hits, al, k):
                    out[k] += 1
            if not hit_in(hits, al, 2):
                fails.append((it[qkey], al[:4], hits[0]["title"] if hits else None))
        return out, fails, len(items)

    dev_items = [{"q": it["prompt"], "a": gold_aliases(it), "cat": it["category"]} for it in dev]
    dres, dfail, dn = measure(dev_items, "q", "a")
    sres, sfail, sn = measure(sft, "question", "aliases")
    print("=== DEV goal fact+definition (n=%d) ===" % dn)
    for k in (1, 2, 3):
        print(f"  recall@{k}: {dres[k]}/{dn} = {100*dres[k]/dn:.1f}%")
    print("  failure examples (gold not in top-2):")
    for q, a, t in dfail[:10]:
        print(f"    Q={q!r} gold={a} top1={t!r}")
    print("=== SFT val factual questions (n=%d, noisy gold) ===" % sn)
    for k in (1, 2, 3):
        print(f"  recall@{k}: {sres[k]}/{sn} = {100*sres[k]/sn:.1f}%" if sn else "  (none)")

    # threshold probe: distribution of top-1 scores on dev (to pick an "unanswerable" cutoff later)
    import numpy as np
    top1 = [r.retrieve(it["q"], k=1)[0]["score"] for it in dev_items if r.retrieve(it["q"], k=1)]
    print("=== DEV top-1 BM25 score distribution ===")
    print("  min %.2f p10 %.2f median %.2f p90 %.2f max %.2f" % (min(top1), np.percentile(top1,10), np.median(top1), np.percentile(top1,90), max(top1)))
    print("Chosen for 15B: k=2 (recall@2 >> recall@1, and 2 passages fit the 512 context). Threshold left at 0 for the model to judge via 'sources don't say'.")


if __name__ == "__main__":
    main()

"""
Phase 26 — plain Qwen's own answers to SciQ TRAIN and BoolQ TRAIN inputs (phase26/CRITERIA_26.md, "Replay").

  SciQ:  user turn "{support}\\n\\nQuestion: {question}\\n\\nOptions: {4 options shuffled, ', '}" (no support line if empty);
         correct = the first option mentioned in the reply is the correct answer.
  BoolQ: user turn "{passage}\\n\\nQuestion: {question}?"; correct = the first "yes"/"no" in the reply matches the label.
Train items are shuffled with seed 2626; any input that hits the Phase 24 checker or the Phase 26 public-test check is skipped
BEFORE generation; up to 3,000 of each are generated with phase24/distill.py generate() (greedy, 512 new tokens, batch 48).
Writes phase26/distill/{sciq,boolq}.jsonl.   Usage: python phase26/distill_26.py
"""
import json
import os
import random
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase24.contam import Contamination  # noqa: E402
from phase24.distill import generate  # noqa: E402
from phase26.contam26 import PublicTests  # noqa: E402

SEED, N_EACH = 2626, 3000
OUT = os.path.join(ROOT, "phase26/distill")


def first_option(reply, options):
    best = None
    for o in options:
        m = re.search(r"(?<![a-z0-9])" + re.escape(o.lower()) + r"(?![a-z0-9])", reply.lower())
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(o) > len(best[1]))):
            best = (m.start(), o)
    return best[1] if best else None


def first_yes_no(reply):
    m = re.search(r"\b(yes|no)\b", reply.lower())
    return m.group(1) if m else None


def main():
    if os.path.exists(os.path.join(OUT, "boolq.jsonl")):
        raise SystemExit("already distilled")
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    con, pub = Contamination(), PublicTests()
    from datasets import load_dataset
    sc = load_dataset("allenai/sciq", split="train")
    bq = load_dataset("aps/super_glue", "boolq", split="train")
    skipped = {"sciq": 0, "boolq": 0}
    sciq = []
    idx = list(range(len(sc)))
    rng.shuffle(idx)
    for i in idx:
        r = sc[i]
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        rng.shuffle(opts)
        sup = r["support"].strip()
        p = (f"{sup}\n\n" if sup else "") + f"Question: {r['question']}\n\nOptions: {', '.join(opts)}"
        if con.hit(r["question"]) or pub.hit(p):
            skipped["sciq"] += 1
            continue
        sciq.append({"sciq_train_index": i, "prompt": p, "options": opts, "gold": r["correct_answer"]})
        if len(sciq) >= N_EACH:
            break
    boolq = []
    idx = list(range(len(bq)))
    rng.shuffle(idx)
    for i in idx:
        r = bq[i]
        p = f"{r['passage']}\n\nQuestion: {r['question']}?"
        if con.hit(r["question"]) or pub.hit(p):
            skipped["boolq"] += 1
            continue
        boolq.append({"boolq_train_index": i, "prompt": p, "gold": "yes" if r["label"] == 1 else "no"})
        if len(boolq) >= N_EACH:
            break
    print("skipped by contamination:", skipped, "phase24 checker:", dict(con.counts), "public:", dict(pub.counts), flush=True)
    outs = generate([x["prompt"] for x in sciq + boolq])
    stats = {}
    for name, items, outs_, judge in (("sciq", sciq, outs[:len(sciq)], lambda it, t: first_option(t, it["options"])),
                                      ("boolq", boolq, outs[len(sciq):], lambda it, t: first_yes_no(t))):
        ok = 0
        with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for it, o in zip(items, outs_):
                pred = judge(it, o["reply"])
                c = o["stop"] == "eos" and pred is not None and pred.lower() == it["gold"].lower()
                ok += c
                f.write(json.dumps({**it, **o, "pred": pred, "correct": c}, ensure_ascii=False) + "\n")
        stats[name] = {"generated": len(items), "correct": ok, "cut": sum(o["stop"] == "max_tokens" for o in outs_)}
    stats["skipped_contamination"] = skipped
    stats["contamination_counts"] = {"phase24": dict(con.counts), "public": dict(pub.counts)}
    json.dump(stats, open(os.path.join(OUT, "distill_26_stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

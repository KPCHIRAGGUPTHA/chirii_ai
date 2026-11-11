"""
Phase 24 — builds fine-tune v2 data (phase24/CRITERIA_24.md). No training.

  (i)   self-distilled general replay: plain Qwen's own answers (phase24/distill/general.jsonl, first 9,000 prompts), so that
        replay is ~35% of all rows. Dropped: answers cut at 512 tokens, empty answers, and answers naming Qwen / Alibaba
        (they would contradict the Chirii identity rows).
  (ii)  GSM8K train: plain Qwen's own worked solutions that reached the right final number (phase24/distill/gsm8k.jsonl), up to 3,000.
  (iii) 23B behaviour rows (phase23/data/sft_23b_train.jsonl): traps, unknowable, identity (all); refusals halved (every
        second row); harmless edgy rows doubled (each twice). 23B arithmetic and RAG rows are not used.
  (iv)  RAG rows in EXACTLY the runtime v2 format (phase24/runtime_v2.py, 3 passages):
          ~2,500 SQuAD 2.0 train answerable: the gold context (<= 200 words) at a random position + 2 BM25 neighbours;
                 target = the gold sentence containing the answer (as 23B)
          ~1,500 own knowledge: the next 1,500 distilled prompts with their real BM25 top-3 (not filtered for relevance);
                 target = plain Qwen's own answer without passages
          ~500   "I don't know": 23B unknowable rows with their BM25 top-3
          ~300   refusals with passages: the refusal rows NOT used in (iii) (selected in code, never printed), BM25 top-3
Contamination: phase24/contam.py on every row (prompt / question and target). Counts only.
Writes phase24/data/sft_24_{train,val}.jsonl and phase24/manifests/sft_24_manifest.json.
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase24/build_sft_24.py
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase24.contam import Contamination  # noqa: E402
from phase24.runtime_v2 import K, rag_prompt  # noqa: E402

SEED = 2424
SQUAD_REV = "3ffb306f725f7d2ce8394bc1873b24868140c412"
D23 = ("phase23/data/sft_23b_train.jsonl", "7a8eb74289047d00633294846a1f8a53621f1d4b4536d4d45463362c3674755e")
RAG_PREFIX = ["", "", "The passage says: ", "From the passages: ", "According to the passages: "]
QWEN = re.compile(r"\bqwen\b|\balibaba\b", re.I)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    rng = random.Random(SEED)
    con = Contamination()
    from phase15.retriever import Retriever
    retr = Retriever.load(os.path.join(ROOT, "phase15/index/bm25"), os.path.join(ROOT, "phase15/corpus/passages.jsonl"))
    gen = [json.loads(l) for l in open(os.path.join(ROOT, "phase24/distill/general.jsonl"), encoding="utf-8")]
    gsm = [json.loads(l) for l in open(os.path.join(ROOT, "phase24/distill/gsm8k.jsonl"), encoding="utf-8")]
    assert sha(os.path.join(ROOT, D23[0])) == D23[1]
    d23 = [json.loads(l) for l in open(os.path.join(ROOT, D23[0]), encoding="utf-8")]
    counts = Counter()
    parts = defaultdict(list)

    from phase17.scorer_v4.scorer_v4 import is_refusal_v4
    from phase11.goal_eval.scorer_v3 import is_refusal
    ok_gen = lambda g: (g["stop"] == "eos" and g["reply"].strip() and not QWEN.search(g["reply"])
                        and not is_refusal_v4(g["reply"]) and not is_refusal(g["reply"]))  # deviation log, 11 Nov 16:40
    general, own = gen[:9000], gen[9000:]
    counts["general_dropped_cut_empty_qwen_or_refusal"] = sum(not ok_gen(g) for g in general)
    counts["own_dropped_cut_empty_qwen_or_refusal"] = sum(not ok_gen(g) for g in own)
    # (ii)
    for g in [x for x in gsm if x["correct"]][:3000]:
        parts["gsm8k"].append({"prompt": g["question"], "reply": g["reply"], "part": "gsm8k", "kind": "self_distilled"})
    counts["gsm8k_correct_available"] = sum(x["correct"] for x in gsm)
    # (iii)
    by = defaultdict(list)
    for r in d23:
        by[r["part"]].append(r)
    for p in ("traps", "unknowable", "identity"):
        parts["behaviour"] += [{"prompt": r["prompt"], "reply": r["reply"], "part": "behaviour", "kind": f"{p}/{r['kind']}"} for r in by[p]]
    refusals = by["refusal"]
    kept_ref, spare_ref = refusals[0::2], refusals[1::2]
    parts["behaviour"] += [{"prompt": r["prompt"], "reply": r["reply"], "part": "behaviour", "kind": f"refusal/{r['kind']}"} for r in kept_ref]
    for r in by["edgy"]:
        for copy in (0, 1):
            parts["behaviour"].append({"prompt": r["prompt"], "reply": r["reply"], "part": "behaviour", "kind": f"edgy/{r['kind']}", "copy": copy})
    # (iv)
    def neighbours(q, n, exclude_title=None):
        hits = retr.retrieve(q, k=n + 2)
        return [h for h in hits if h["title"] != exclude_title][:n]
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="train", revision=SQUAD_REV)
    idx = list(range(len(sq)))
    rng.shuffle(idx)
    for i in idx:
        if len(parts["rag_passage"]) >= 2500:
            break
        row = sq[i]
        ctx, q, ans = row["context"], row["question"], row["answers"]
        if not ans["text"] or len(ctx.split()) > 200 or len(q.split()) < 4:
            continue
        a, start = ans["text"][0], ans["answer_start"][0]
        pos, sent = 0, None
        for s in re.split(r"(?<=[.!?])\s+", ctx):
            j = ctx.find(s, pos)
            if j <= start < j + len(s) + 1:
                sent = s
                break
            pos = j + len(s)
        if not sent or a not in sent or len(sent.split()) > 60:
            continue
        gold = {"title": row["title"].replace("_", " "), "text": ctx}
        ps = neighbours(q, 2, gold["title"])
        if len(ps) < 2:
            continue
        ps.insert(rng.randrange(3), gold)
        parts["rag_passage"].append({"prompt": rag_prompt(q, ps), "question": q, "reply": rng.choice(RAG_PREFIX) + sent.strip(),
                                     "part": "rag", "kind": "answer_from_passage", "gold": a})
    for g in [x for x in own if ok_gen(x)][:1500]:
        ps = neighbours(g["prompt"], 3)
        if len(ps) == K:
            parts["rag_own"].append({"prompt": rag_prompt(g["prompt"], ps), "question": g["prompt"], "reply": g["reply"], "part": "rag", "kind": "own_knowledge"})
    for r in by["unknowable"][:500]:
        ps = neighbours(r["prompt"], 3)
        if len(ps) == K:
            parts["rag_idk"].append({"prompt": rag_prompt(r["prompt"], ps), "question": r["prompt"], "reply": r["reply"], "part": "rag", "kind": "idk_unknowable"})
    for r in spare_ref[:300]:
        ps = neighbours(r["prompt"], 3)
        if len(ps) == K:
            parts["rag_refusal"].append({"prompt": rag_prompt(r["prompt"], ps), "question": r["prompt"], "reply": r["reply"], "part": "rag", "kind": "refusal_with_passages"})
    # (i): replay = 35% of all rows
    non_replay = sum(len(v) for v in parts.values())
    want = round(non_replay * 0.35 / 0.65)
    parts["replay"] = [{"prompt": g["prompt"], "reply": g["reply"], "part": "replay", "kind": "self_distilled"} for g in general if ok_gen(g)][:want]
    counts["replay_wanted"] = want
    # contamination
    final = []
    for name, rows in parts.items():
        for r in rows:
            if con.hit(r.get("question", r["prompt"]), r["reply"]):
                counts[f"removed_{name}"] += 1
                continue
            final.append(r)
    counts.update({f"removed_against_{k}": v for k, v in con.counts.items()})
    rng.shuffle(final)
    val, train = final[:400], final[400:]
    os.makedirs(os.path.join(ROOT, "phase24/data"), exist_ok=True)
    files = {}
    for split, rows in (("train", train), ("val", val)):
        path = os.path.join(ROOT, f"phase24/data/sft_24_{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps(dict(r, id=f"sft24_{split}_{i:05d}"), ensure_ascii=False) + "\n")
        files[split] = {"file": f"phase24/data/sft_24_{split}.jsonl", "rows": len(rows), "sha256": sha(path)}
    by_part = Counter(r["part"] for r in final)
    by_kind = Counter(f"{r['part']}/{r['kind'].split('/')[0]}" for r in final)
    m = {"seed": SEED, "files": files, "rows_total": len(final), "by_part": dict(by_part), "by_kind": dict(sorted(by_kind.items())),
         "replay_share": round(by_part["replay"] / len(final), 4), "max_identical_reply": max(Counter(r["reply"] for r in final).values()),
         "counts": dict(counts), "squad_revision": SQUAD_REV, "sources": {"sft_23b": D23}}
    os.makedirs(os.path.join(ROOT, "phase24/manifests"), exist_ok=True)
    json.dump(m, open(os.path.join(ROOT, "phase24/manifests/sft_24_manifest.json"), "w"), indent=1)
    print(json.dumps({k: m[k] for k in ("files", "rows_total", "by_part", "by_kind", "replay_share", "max_identical_reply", "counts")}, indent=1))
    ex = defaultdict(list)
    for r in final:
        key = f"{r['part']}/{r['kind'].split('/')[0]}"
        if len(ex[key]) < 5:
            ex[key].append(r)
    for key, rows in sorted(ex.items()):
        print(f"--- {key}")
        for r in rows:
            hide = "refusal" in r["kind"] or "self_harm" in r["kind"]
            q = "[prompt not shown]" if hide else r.get("question", r["prompt"])
            print(f"  {q[:100]!r} -> {r['reply'][:140]!r}")


if __name__ == "__main__":
    main()

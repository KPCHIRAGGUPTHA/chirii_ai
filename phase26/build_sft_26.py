"""
Phase 26 — builds the v0.2 fine-tune data (phase26/CRITERIA_26.md, "Data"). No training.

  general mode (bare message):  behaviour (23B rows, Phase 24 rule), GSM8K self-distilled (3,000), replay = 50% of all rows
                                (Phase 24 distilled general prompts, all 10,500, then SciQ / BoolQ train correct answers, equal halves)
  document mode (DOC_TEMPLATE): 2,500 SQuAD 2.0 train answerable (target = gold span), 1,200 SQuAD 2.0 train unanswerable
                                (target "The documents don't say."), 300 refusals with BM25 top-3 Simple Wikipedia passages
  held back for dev:            300 SQuAD 2.0 train questions (150 answerable, 150 unanswerable) from whole held-back articles
Contamination: phase24/contam.py (question and reply) and phase26/contam26.py (full user turn and reply) on every row; counts only.
Writes phase26/data/sft_26_{train,val}.jsonl, phase26/data/dev_squad_300.jsonl and phase26/manifests/sft_26_manifest.json.
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase26/build_sft_26.py
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
from phase25.docmode import select  # noqa: E402
from phase26.contam26 import PublicTests  # noqa: E402
from phase26.modes import ABSTAIN, doc_prompt_from_passages  # noqa: E402

SEED = 2626
SQUAD_REV = "3ffb306f725f7d2ce8394bc1873b24868140c412"
D23 = ("phase23/data/sft_23b_train.jsonl", "7a8eb74289047d00633294846a1f8a53621f1d4b4536d4d45463362c3674755e")
QWEN = re.compile(r"\bqwen\b|\balibaba\b", re.I)
N_ANS, N_UNA, N_REF, N_DEV_EACH = 2500, 1200, 300, 150


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]


def main():
    rng = random.Random(SEED)
    con, pub = Contamination(), PublicTests()
    removed = Counter()

    def clean(name, rows, question_key="question"):
        out = []
        for r in rows:
            a = con.hit(r.get(question_key) or r["prompt"], r["reply"])
            b = pub.hit(r["prompt"], r["reply"])
            if a or b:
                removed[name] += 1
                continue
            out.append(r)
        return out

    from phase11.goal_eval.scorer_v3 import is_refusal
    from phase17.scorer_v4.scorer_v4 import is_refusal_v4
    ok_gen = lambda g: (g["stop"] == "eos" and g["reply"].strip() and not QWEN.search(g["reply"])
                        and not is_refusal_v4(g["reply"]) and not is_refusal(g["reply"]))
    parts = {}
    counts = Counter()
    # behaviour and GSM8K: the Phase 24 rule
    assert sha(os.path.join(ROOT, D23[0])) == D23[1]
    by = defaultdict(list)
    for r in jl(D23[0]):
        by[r["part"]].append(r)
    beh = []
    for p in ("traps", "unknowable", "identity"):
        beh += [{"prompt": r["prompt"], "reply": r["reply"], "mode": "general", "part": "behaviour", "kind": f"{p}/{r['kind']}"} for r in by[p]]
    kept_ref, spare_ref = by["refusal"][0::2], by["refusal"][1::2]
    beh += [{"prompt": r["prompt"], "reply": r["reply"], "mode": "general", "part": "behaviour", "kind": f"refusal/{r['kind']}"} for r in kept_ref]
    for r in by["edgy"]:
        for copy in (0, 1):
            beh.append({"prompt": r["prompt"], "reply": r["reply"], "mode": "general", "part": "behaviour", "kind": f"edgy/{r['kind']}", "copy": copy})
    parts["behaviour"] = clean("behaviour", beh)
    gsm = [x for x in jl("phase24/distill/gsm8k.jsonl") if x["correct"]][:3000]
    parts["gsm8k"] = clean("gsm8k", [{"prompt": g["question"], "reply": g["reply"], "mode": "general", "part": "gsm8k", "kind": "self_distilled"} for g in gsm])

    # SQuAD 2.0 train: held-back articles for dev, then document-mode rows
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="train", revision=SQUAD_REV)
    titles = sorted(set(sq["title"]))
    rng.shuffle(titles)
    by_title = defaultdict(list)
    for i, t in enumerate(sq["title"]):
        by_title[t].append(i)
    held, dev_a, dev_u = set(), [], []
    for t in titles:
        if len(dev_a) >= N_DEV_EACH and len(dev_u) >= N_DEV_EACH:
            break
        held.add(t)
        for i in by_title[t]:
            (dev_a if sq[i]["answers"]["text"] else dev_u).append(i)
    dev_rows = []
    for i in dev_a[:N_DEV_EACH] + dev_u[:N_DEV_EACH]:
        r = sq[i]
        ps = select(r["question"], [(r["title"].replace("_", " "), r["context"])])
        dev_rows.append({"id": r["id"], "question": r["question"], "prompt": doc_prompt_from_passages(r["question"], ps),
                         "type": "answerable" if r["answers"]["text"] else "unanswerable", "aliases": list(r["answers"]["text"])})
    counts["dev_held_back_articles"] = len(held)
    pool = [i for t in titles if t not in held for i in by_title[t]]
    rng.shuffle(pool)
    other_titles = [t for t in titles if t not in held]
    first_ctx = {t: sq[by_title[t][0]]["context"] for t in other_titles}

    def docs_for(i):
        r = sq[i]
        title = r["title"].replace("_", " ")
        docs = [(title, r["context"])]
        if rng.random() < 0.5:
            t2 = rng.choice(other_titles)
            while t2 == r["title"]:
                t2 = rng.choice(other_titles)
            docs.insert(rng.randrange(2), (t2.replace("_", " "), first_ctx[t2]))
        return select(r["question"], docs)

    ans, una = [], []
    for i in pool:
        if len(ans) >= int(N_ANS * 1.05) and len(una) >= int(N_UNA * 1.05):
            break
        r = sq[i]
        q = r["question"]
        if len(q.split()) < 4:
            continue
        if r["answers"]["text"]:
            if len(ans) >= int(N_ANS * 1.05):
                continue
            a = r["answers"]["text"][0]
            ps = docs_for(i)
            if not any(a in p["text"] for p in ps):
                counts["answerable_answer_not_in_selected_passages"] += 1
                continue
            ans.append({"prompt": doc_prompt_from_passages(q, ps), "question": q, "reply": a, "mode": "document", "part": "document", "kind": "answerable", "squad_id": r["id"]})
        else:
            if len(una) >= int(N_UNA * 1.05):
                continue
            ps = docs_for(i)
            una.append({"prompt": doc_prompt_from_passages(q, ps), "question": q, "reply": ABSTAIN, "mode": "document", "part": "document", "kind": "unanswerable", "squad_id": r["id"]})
    parts["doc_answerable"] = clean("doc_answerable", ans)[:N_ANS]
    parts["doc_unanswerable"] = clean("doc_unanswerable", una)[:N_UNA]
    from phase15.retriever import Retriever
    retr = Retriever.load(os.path.join(ROOT, "phase15/index/bm25"), os.path.join(ROOT, "phase15/corpus/passages.jsonl"))
    ref = []
    for r in spare_ref[:N_REF]:
        hits = retr.retrieve(r["prompt"], k=3)
        if len(hits) == 3:
            ps = select(r["prompt"], [(h["title"], h["text"]) for h in hits])
            ref.append({"prompt": doc_prompt_from_passages(r["prompt"], ps), "question": r["prompt"], "reply": r["reply"], "mode": "document", "part": "document", "kind": "refusal_with_documents"})
    parts["doc_refusal"] = clean("doc_refusal", ref)

    # replay = 50% of all rows
    non_replay = sum(len(v) for v in parts.values())
    want = non_replay
    gen = [g for g in jl("phase24/distill/general.jsonl") if ok_gen(g)]
    counts["general_distilled_usable"] = len(gen)
    rep_gen = clean("replay_general", [{"prompt": g["prompt"], "reply": g["reply"], "mode": "general", "part": "replay", "kind": "general"} for g in gen])
    sb = {}
    for name in ("sciq", "boolq"):
        rows = [x for x in jl(f"phase26/distill/{name}.jsonl") if x["correct"] and ok_gen(x)]
        counts[f"{name}_correct_usable"] = len(rows)
        sb[name] = clean(f"replay_{name}", [{"prompt": x["prompt"], "reply": x["reply"], "mode": "general", "part": "replay", "kind": name} for x in rows])
    rest = max(0, want - len(rep_gen))
    half = rest // 2
    n_s = min(len(sb["sciq"]), max(half, rest - len(sb["boolq"])))
    n_b = min(len(sb["boolq"]), rest - n_s)
    parts["replay"] = rep_gen[:want] + sb["sciq"][:n_s] + sb["boolq"][:n_b]
    counts["replay_wanted"] = want
    counts["replay_sciq"], counts["replay_boolq"] = n_s, n_b

    final = [r for rows in parts.values() for r in rows]
    rng.shuffle(final)
    val, train = final[:400], final[400:]
    os.makedirs(os.path.join(ROOT, "phase26/data"), exist_ok=True)
    files = {}
    for split, rows in (("train", train), ("val", val)):
        path = os.path.join(ROOT, f"phase26/data/sft_26_{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps(dict(r, id=f"sft26_{split}_{i:05d}"), ensure_ascii=False) + "\n")
        files[split] = {"file": f"phase26/data/sft_26_{split}.jsonl", "rows": len(rows), "sha256": sha(path)}
    dpath = os.path.join(ROOT, "phase26/data/dev_squad_300.jsonl")
    with open(dpath, "w", encoding="utf-8", newline="\n") as f:
        for r in dev_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    files["dev_squad_300"] = {"file": "phase26/data/dev_squad_300.jsonl", "rows": len(dev_rows), "sha256": sha(dpath),
                              "answerable": sum(r["type"] == "answerable" for r in dev_rows)}
    by_part = Counter(r["part"] for r in final)
    by_kind = Counter(f"{r['part']}/{r['kind'].split('/')[0]}" for r in final)
    by_mode = Counter(r["mode"] for r in final)
    doc = [r for r in final if r["mode"] == "document"]
    m = {"seed": SEED, "files": files, "rows_total": len(final), "by_mode": dict(by_mode), "by_part": dict(by_part), "by_kind": dict(sorted(by_kind.items())),
         "replay_share": round(by_part["replay"] / len(final), 4),
         "document_mode_unanswerable_share": round(sum(r["kind"] == "unanswerable" for r in doc) / max(1, len(doc)), 4),
         "refusal_rows_by_mode": {"general": sum(r["kind"].startswith("refusal") for r in final if r["mode"] == "general"),
                                  "document": sum(r["kind"].startswith("refusal") for r in doc)},
         "removed_by_contamination": dict(removed), "contamination_hits": {"phase24": dict(con.counts), "public": dict(pub.counts)},
         "counts": dict(counts), "squad_revision": SQUAD_REV, "held_back_titles": sorted(held), "sources": {"sft_23b": D23}}
    os.makedirs(os.path.join(ROOT, "phase26/manifests"), exist_ok=True)
    json.dump(m, open(os.path.join(ROOT, "phase26/manifests/sft_26_manifest.json"), "w"), indent=1)
    print(json.dumps({k: m[k] for k in m if k != "held_back_titles"}, indent=1))
    ex = defaultdict(list)
    for r in final:
        key = f"{r['part']}/{r['kind'].split('/')[0]}"
        if len(ex[key]) < 3:
            ex[key].append(r)
    for key, rows in sorted(ex.items()):
        print(f"--- {key}")
        for r in rows:
            hide = "refusal" in r["kind"] or "self_harm" in r["kind"] or "edgy" in r["kind"]
            q = "[prompt not shown]" if hide else r.get("question", r["prompt"])
            rep = "[reply not shown]" if hide else r["reply"]
            print(f"  {q[:100]!r} -> {rep[:140]!r}")


if __name__ == "__main__":
    main()

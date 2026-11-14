"""
Phase 26B — builds the v0.3 fine-tune data (phase26b/CRITERIA_26B.md, "Data"). No training.

  v0.2 rows:   phase26/data/sft_26_{train,val}.jsonl unchanged, minus rows that hit HarmBench (counted)
  new rows:    1,000 general-mode + 500 document-mode discriminatory requests (frames: phase26b/private/disc_frames.json, private,
               never printed) -> decline + brief correction of the premise (every reply passes is_refusal_v4; no reply > 3 times);
               500 general-mode guard rows: benign questions about the same groups, answered by plain Qwen (self-distilled,
               Phase 24 filters), cached in phase26b/distill/guard.jsonl
Contamination on every new row: phase24/contam.py, phase26/contam26.py, HarmBench 8-gram, XSTest guard (5-gram / Jaccard >= 0.5).
Prints counts, and example REPLIES of discriminatory rows (never their prompts); guard rows are printed in full.
Writes phase26b/data/sft_26b_{train,val}.jsonl and phase26b/manifests/sft_26b_manifest.json.
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase26b/build_sft_26b.py
"""
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase24.contam import Contamination  # noqa: E402
from phase25.docmode import select  # noqa: E402
from phase26.contam26 import PublicTests  # noqa: E402
from phase26.modes import doc_prompt_from_passages  # noqa: E402
from phase26b.contam26b import HarmBench, XSTestGuard  # noqa: E402

SEED = 2627
V02 = {"train": ("phase26/data/sft_26_train.jsonl", "767ac9214d68afe787d142d56b740b57be523f2ec35d39a0bc8de585b8e38b9d"),
       "val": ("phase26/data/sft_26_val.jsonl", "860e66fcea0d58b4efa99433b25206666a956e61de33b8108bd5e6ecb42f6a8c")}
FRAMES = "phase26b/private/disc_frames.json"
N_GEN, N_DOC, N_GUARD = 1000, 500, 500
QWEN = re.compile(r"\bqwen\b|\balibaba\b", re.I)

OPENERS = ["I won't help with that.", "I can't help with that.", "I won't write that.", "I'm not going to do that.",
           "Sorry, I can't help with that.", "I won't do that.", "I can't do that.", "I'm sorry, but I won't help with that."]
CORRECTIONS = {
    "rights": ["{Gc} have the same rights as everyone else, and denying someone the chance to {R} because of their {A} is discrimination.",
               "A person's {A} says nothing about whether they should be able to {R}. Excluding people on that basis is unfair, and in many places it is illegal.",
               "Whether someone can {R} should not depend on their {A}. Treating {G} differently here would be discrimination."],
    "inferior": ["No group of people is inferior because of their {A}. People differ as individuals, and claims like this have long been used to justify unfair treatment.",
                 "The premise is wrong: {G} are not less capable or less valuable than anyone else. A person's {A} tells you nothing about their worth.",
                 "Claims that {G} are inferior are not supported by evidence. Abilities vary from person to person, not by {A}."],
    "exclusion": ["Turning people away because of their {A} is discrimination, and in many countries it is against the law.",
                  "A rule that excludes {G} singles people out for their {A}. Everyone should be treated by the same fair rules.",
                  "Keeping {G} out because of their {A} is unfair, and it is illegal in many places."],
    "threat": ["Portraying {G} as a danger because of their {A} spreads prejudice, and it can lead to real harm against innocent people.",
               "{Gc} are not a threat because of their {A}. Blaming a whole group for problems is scapegoating, and it has fuelled serious harm in the past.",
               "Content that casts {G} as enemies targets people for their {A}. It is unfair and it can put real people in danger."],
    "jokes": ["Jokes and insults that put {G} down for their {A} reinforce prejudice and hurt real people.",
              "Mocking people for their {A} isn't harmless fun; it singles out {G} and feeds prejudice.",
              "Putting {G} down because of their {A} is hurtful and unfair."],
    "screening": ["Rejecting people for a job or a home because of their {A} is discrimination and is illegal in many countries. Decisions should rest on each person's qualifications or circumstances.",
                  "Screening out {G} would be discrimination based on {A}, and hiding it does not make it fair or legal.",
                  "Treating applicants differently because of their {A} is unfair and, in many places, against the law. Judge each person on what matters for the job or the tenancy."],
}
OFFERS = {
    "rights": ["I'd be glad to explain how equal rights protections work instead.", "I'm happy to talk about the history of equal rights instead."],
    "inferior": ["I'm happy to talk about what research says about human diversity instead.", ""],
    "exclusion": ["I'd be glad to help write a rule that treats everyone fairly instead.", "I'm happy to help with a welcoming notice for everyone instead."],
    "threat": ["I'd be glad to help write something about how communities can stand up to prejudice instead.", ""],
    "jokes": ["I'm happy to help with a joke that doesn't target anyone instead.", "I'd be glad to help with some light-hearted humour that doesn't put anyone down."],
    "screening": ["I'd be glad to help write fair, skills-based selection criteria instead.", "I'm happy to help draft a fair screening process that applies to everyone."],
}
GUARD = [
    "What protections do anti-discrimination laws usually give {G} at work?",
    "How can a company make its hiring process fair for {G}?",
    "Why is it wrong to deny {G} the right to {R}?",
    "Write a short welcome message for {G} joining our neighbourhood association.",
    "What challenges do {G} often face when looking for housing, and how can landlords be fair?",
    "How can I be a better ally to {G}?",
    "Why are stereotypes about {G} harmful?",
    "What does discrimination against {G} mean, and why is it illegal in many countries?",
    "How can a school make sure {G} feel included?",
    "What should I do if I see someone treating {G} unfairly?",
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]


def cap(s):
    return s[0].upper() + s[1:]


def main():
    from phase11.goal_eval.scorer_v3 import is_refusal
    from phase17.scorer_v4.scorer_v4 import is_refusal_v4
    rng = random.Random(SEED)
    fr = json.load(open(os.path.join(ROOT, FRAMES), encoding="utf-8"))
    con, pub, hb, xg = Contamination(), PublicTests(), HarmBench(), XSTestGuard()
    removed, counts = Counter(), Counter()

    # ---------------- discriminatory requests (prompts never printed)
    kinds = sorted(fr["frames"])
    reply_uses = Counter()
    seen, rejected = set(), set()

    def make(kind):
        g, a = rng.choice(fr["groups"])
        r, p = rng.choice(fr["rights"]), rng.choice(fr["places"])
        prompt = rng.choice(fr["frames"][kind]).format(G=g, R=r, P=p)
        prompt = cap(prompt)
        corr = rng.choice(CORRECTIONS[kind]).format(G=g, Gc=cap(g), A=a, R=r)
        offer = rng.choice(OFFERS[kind])
        reply = " ".join(x for x in (rng.choice(OPENERS), corr, offer) if x)
        return prompt, reply

    def disc_rows(n, mode, retr=None):
        out, active, misses = [], list(kinds), Counter()
        while len(out) < n and active:
            kind = active[len(out) % len(active)]
            prompt, reply = make(kind)
            if prompt in seen or prompt in rejected or reply_uses[reply] >= 3:
                misses[kind] += 1
                if misses[kind] > 3000:  # this kind's combinations are used up: continue with the others
                    active.remove(kind)
                    counts[f"kind_exhausted_{mode}_{kind}"] += 1
                continue
            misses[kind] = 0
            assert is_refusal_v4(reply), "a written reply is not recognised as a refusal"
            if con.hit(prompt, reply) or pub.hit(prompt, reply) or hb.hit(prompt, reply) or xg.hit(prompt):
                rejected.add(prompt)
                removed[f"disc_{mode}"] += 1
                continue
            row = {"prompt": prompt, "question": prompt, "reply": reply, "mode": mode, "part": "discrimination", "kind": f"discrimination/{kind}"}
            if mode == "document":
                hits = retr.retrieve(prompt, k=3)
                if len(hits) < 3:
                    rejected.add(prompt)
                    counts["doc_no_3_passages"] += 1
                    continue
                ps = select(prompt, [(h["title"], h["text"]) for h in hits])
                row["prompt"] = doc_prompt_from_passages(prompt, ps)
                if pub.hit(row["prompt"]) or hb.hit(row["prompt"]):
                    rejected.add(prompt)
                    removed["disc_document_passages"] += 1
                    continue
            seen.add(prompt)
            reply_uses[reply] += 1
            out.append(row)
        assert len(out) == n, (mode, len(out))
        return out

    disc_gen = disc_rows(N_GEN, "general")
    from phase15.retriever import Retriever
    retr = Retriever.load(os.path.join(ROOT, "phase15/index/bm25"), os.path.join(ROOT, "phase15/corpus/passages.jsonl"))
    disc_doc = disc_rows(N_DOC, "document", retr)
    counts["max_reply_repeats"] = max(reply_uses.values())

    # ---------------- guard rows (benign; plain Qwen's own answers)
    gpath = os.path.join(ROOT, "phase26b/distill/guard.jsonl")
    if not os.path.exists(gpath):
        cands, s = [], set()
        while len(cands) < int(N_GUARD * 1.3):
            g, _ = rng.choice(fr["groups"])
            q = rng.choice(GUARD).format(G=g, R=rng.choice(fr["rights"]))
            if q in s:
                continue
            s.add(q)
            if con.hit(q, "") or pub.hit(q) or hb.hit(q) or xg.hit(q):
                removed["guard_prompt"] += 1
                continue
            cands.append(q)
        from phase24.distill import generate
        gen = generate(cands)
        os.makedirs(os.path.dirname(gpath), exist_ok=True)
        with open(gpath, "w", encoding="utf-8") as f:
            for q, g in zip(cands, gen):
                f.write(json.dumps({"prompt": q, **g}, ensure_ascii=False) + "\n")
    guard = []
    for g in jl("phase26b/distill/guard.jsonl"):
        ok = g["stop"] == "eos" and g["reply"].strip() and not QWEN.search(g["reply"]) and not is_refusal_v4(g["reply"]) and not is_refusal(g["reply"])
        if not ok:
            counts["guard_filtered_out"] += 1
            continue
        if con.hit(g["prompt"], g["reply"]) or pub.hit(g["prompt"], g["reply"]) or hb.hit(g["prompt"], g["reply"]):
            removed["guard"] += 1
            continue
        guard.append({"prompt": g["prompt"], "reply": g["reply"], "mode": "general", "part": "guard", "kind": "guard/self_distilled"})
    counts["guard_usable"] = len(guard)
    guard = guard[:N_GUARD]

    # ---------------- v0.2 rows, unchanged except HarmBench hits
    v02 = {}
    for split, (path, h) in V02.items():
        assert sha(os.path.join(ROOT, path)) == h, path
        keep = []
        for r in jl(path):
            if hb.hit(r["prompt"], r["reply"]):
                removed[f"v02_{split}_harmbench"] += 1
                continue
            keep.append(r)
        v02[split] = keep

    new = disc_gen + disc_doc + guard
    train = [dict(r) for r in v02["train"]] + new
    rng.shuffle(train)
    val = v02["val"]
    os.makedirs(os.path.join(ROOT, "phase26b/data"), exist_ok=True)
    files = {}
    for split, rows in (("train", train), ("val", val)):
        path = os.path.join(ROOT, f"phase26b/data/sft_26b_{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(rows):
                r = dict(r)
                r.setdefault("id", f"sft26b_{split}_{i:05d}")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[split] = {"file": f"phase26b/data/sft_26b_{split}.jsonl", "rows": len(rows), "sha256": sha(path)}
    allr = train + val
    by_part = Counter(r["part"] for r in allr)
    m = {"seed": SEED, "files": files, "rows_total": len(allr), "by_mode": dict(Counter(r["mode"] for r in allr)),
         "by_part": dict(by_part), "new_rows": {"discrimination_general": len(disc_gen), "discrimination_document": len(disc_doc), "guard": len(guard)},
         "new_by_kind": dict(sorted(Counter(f"{r['mode']}/{r['kind']}" for r in new).items())),
         "replay_share": round(by_part["replay"] / len(allr), 4),
         "removed_by_contamination": dict(removed),
         "contamination_hits": {"phase24": dict(con.counts), "public": dict(pub.counts), "harmbench": dict(hb.counts), "xstest_guard": dict(xg.counts)},
         "counts": dict(counts), "sources": {"v0_2": V02, "frames_sha256": sha(os.path.join(ROOT, FRAMES))}}
    os.makedirs(os.path.join(ROOT, "phase26b/manifests"), exist_ok=True)
    json.dump(m, open(os.path.join(ROOT, "phase26b/manifests/sft_26b_manifest.json"), "w"), indent=1)
    print(json.dumps(m, indent=1))
    print("--- example replies of discriminatory rows (prompts not shown)")
    for r in disc_gen[:6] + disc_doc[:3]:
        print(f"  [{r['mode']}/{r['kind']}] [prompt not shown] -> {r['reply']!r}")
    print("--- guard rows")
    for r in guard[:5]:
        print(f"  {r['prompt']!r} -> {r['reply'][:200]!r}")


if __name__ == "__main__":
    main()

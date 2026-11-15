"""
Phase 26C — builds the 2,000-row pool of new safety rows and the four mixtures (phase26c/CRITERIA_26C.md + Amendment 1). No training.

Requests are NEVER written here and NEVER printed. They come, by code, from:
  - phase23/data/sft_23b_train.jsonl 'refusal'-kind rows not used by v0.2 (by["refusal"][1::2][300:]);
  - the Phase 26B discrimination frames (phase26b/private/disc_frames.json), composed exactly as in Phase 26B;
  - PKU-SafeRLHF train PROMPTS only (responses never read into training or saved): eligible if >= 4 labelled responses and all
    unsafe; category = most-marked harm category; Copyright Issues / Sexual Content / Psychological Harm excluded.
Only the replies are written here: opener + one 'why' sentence (premise correction for discrimination) + optional alternative;
every reply passes is_refusal_v4; no reply > 3 times.
Pool order: blocks of 20 positions (6 disc-general, 2 disc-document, 6 harm-general, 6 harm-document), shuffled within each
block (seed 2629); mixture N = base rows + pool[:N], shuffled (seed 2628).
Contamination on every request and full user turn: phase24/contam.py, phase26/contam26.py, HarmBench, XSTest guard, the full
OR-Bench-80K / Toxic files and the two Phase 26C dev sets (8-gram).
Writes phase26c/data/sft_26c_n{N}_{train,val}.jsonl, phase26c/manifests/sft_26c_n{N}_manifest.json, phase26c/manifests/pool_26c.json.
Prints counts and example REPLIES only.
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase26c/build_sft_26c.py
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase24.contam import Contamination  # noqa: E402
from phase24.runtime_v2 import fmt  # noqa: E402
from phase25.docmode import select  # noqa: E402
from phase26.contam26 import PublicTests, words  # noqa: E402
from phase26.modes import doc_prompt_from_passages  # noqa: E402
from phase26b.contam26b import HarmBench, XSTestGuard  # noqa: E402
from phase26c.contam26c import Grams8, orbench_csv  # noqa: E402

SEED_POOL, SEED_MIX = 2629, 2628
MIXES = (0, 500, 1000, 2000)
BLOCK = ["dg"] * 6 + ["dd"] * 2 + ["hg"] * 6 + ["hd"] * 6
V02 = {"train": ("phase26/data/sft_26_train.jsonl", "767ac9214d68afe787d142d56b740b57be523f2ec35d39a0bc8de585b8e38b9d"),
       "val": ("phase26/data/sft_26_val.jsonl", "860e66fcea0d58b4efa99433b25206666a956e61de33b8108bd5e6ecb42f6a8c")}
P23 = ("phase23/data/sft_23b_train.jsonl", "7a8eb74289047d00633294846a1f8a53621f1d4b4536d4d45463362c3674755e")
FRAMES = ("phase26b/private/disc_frames.json", "43a02a2d")
PKU_REV = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
PKU = {"Alpaca-7B": ("phase26c/public/pku_saferlhf_alpaca_7b_train.jsonl", "e9aeebe2bbbed37d7a28561809f47c6c0622e48b170c80bc2eb72d908d2c4666"),
       "Alpaca2-7B": ("phase26c/public/pku_saferlhf_alpaca2_7b_train.jsonl", "f7845941c570087a320121d222cbdc500189c59310c60aadaf23b14877c92e0a"),
       "Alpaca3-8B": ("phase26c/public/pku_saferlhf_alpaca3_8b_train.jsonl", "21c3b4f2572444e42d161f09166471d14208c9e153fcc0ec900ae05013e47d4c")}
PKU_EXCLUDED = {"Copyright Issues", "Sexual Content", "Psychological Harm"}
PKU_GROUP = {"Economic Crime": "fraud", "White-Collar Crime": "fraud", "Mental Manipulation": "manipulation", "Privacy Violation": "privacy",
             "Drugs": "drugs", "Endangering National Security": "public_order", "Disrupting Public Order": "public_order", "Cybercrime": "cyber",
             "Insulting Behavior": "insult", "Violence": "violence", "Physical Harm": "violence", "Human Trafficking": "trafficking",
             "Endangering Public Health": "public_health", "Environmental Damage": "public_health", "Animal Abuse": "animals",
             "Discriminatory Behavior": "disc_general"}
MIN_PASSAGE_WORDS = 150

# ---------------------------------------------------------------- the only newly written text: replies
OPENERS = ["I won't help with that.", "I can't help with this.", "I'm not able to help with that.", "Sorry, I can't do that.",
           "I won't do this.", "I'm not going to help with that.", "I can't assist with that request.", "I'm sorry, but I can't help with that.",
           "I won't be writing that.", "No, I can't help with that.", "I have to decline this one.", "I can't help you with that one.",
           "I won't assist with this.", "I can't provide that."]
WHY = {
    "general": ["It could seriously hurt people.", "Helping with this could put people in real danger.",
                "This could cause real harm to others, and in many places it is illegal.", "That kind of help could be used to hurt someone.",
                "It risks real harm to people who never agreed to it.", "Doing this would likely break the law and could hurt others."],
    "fraud": ["It would mean deceiving people out of their money, which is fraud and illegal.", "Schemes like this cheat real people and can ruin them financially.",
              "That is financial crime, and the losses fall on real people.", "Tricking people or institutions for money is illegal and does real damage."],
    "manipulation": ["Manipulating someone like this takes away their ability to make their own choices.",
                     "Deliberately deceiving or pressuring a person like that can do lasting harm to them.",
                     "It treats another person as something to be controlled, which is harmful and unfair to them.",
                     "Psychological manipulation can seriously damage someone's wellbeing and trust."],
    "privacy": ["It would invade someone's privacy without their consent.", "Getting or sharing someone's personal information like that can put them at risk.",
                "People have a right to keep that information private, and exposing it can cause real harm.",
                "Tracking or exposing someone without permission is a serious privacy violation and often illegal."],
    "drugs": ["Making, selling or misusing illegal drugs puts people's health and lives at risk.", "That would mean helping with illegal drug activity, which can hurt people badly.",
              "Drugs handled this way can cause overdoses and serious harm.", "This is illegal and dangerous, both for you and for others."],
    "public_order": ["It could endanger many people and public safety.", "Actions like this can threaten public safety and are serious crimes.",
                     "This could cause widespread harm and disorder.", "Undermining public safety or security like this can put many lives at risk."],
    "cyber": ["Getting into systems or accounts without permission is illegal and harms the people who own them.",
              "Unauthorised access like this can do real damage, and it is a crime in most countries.",
              "Attacking computers or data you don't own can hurt real people and organisations.",
              "Malicious software and break-ins put other people's data and safety at risk."],
    "insult": ["Insulting or demeaning people causes real hurt and doesn't solve anything.", "Words meant to humiliate someone can do lasting damage.",
               "Putting people down like that is cruel, and I don't want to help hurt anyone.", "Content designed to demean a person is harmful to them."],
    "violence": ["It could lead to someone being seriously hurt or killed.", "Hurting another person is wrong and against the law.",
                 "Violence like this causes real injury and suffering.", "I won't help with anything meant to physically harm someone."],
    "trafficking": ["Trafficking or exploiting people is a grave crime that causes immense suffering.", "It would mean exploiting vulnerable people, which is deeply harmful and illegal.",
                    "Forcing or smuggling people for profit violates their basic rights.", "Exploiting people this way does lifelong harm to victims."],
    "public_health": ["It could put people's health or the environment at serious risk.", "This could harm many people's health and safety.",
                      "Actions like this can spread harm well beyond the people directly involved.", "Endangering health or the environment like that is dangerous and often illegal."],
    "animals": ["Hurting animals is cruel and, in most places, illegal.", "Animals can suffer, and causing them harm on purpose is wrong.",
                "That would mean treating an animal cruelly, which I won't support.", "Animal cruelty causes real suffering and is against the law in many countries."],
    "disc_general": ["Treating people as lesser because of who they are is discrimination, and the idea behind it is wrong.",
                     "No group of people is worth less than another; judging people by group membership is unfair.",
                     "The premise is mistaken: people should be judged as individuals, not by the group they belong to.",
                     "Claims that a whole group deserves worse treatment are prejudice, not fact.",
                     "Singling out people for their identity spreads prejudice and can lead to real harm.",
                     "Everyone deserves equal respect; stereotypes about a group don't describe the people in it."],
}
ALT = {
    "general": ["If something is worrying you, I'm happy to talk about safe options.", "I'm glad to help with something else.", ""],
    "fraud": ["I'm happy to explain how to spot and avoid scams instead.", "I can help with legitimate ways to earn or save money instead.", ""],
    "manipulation": ["I'm happy to help with honest, respectful ways to communicate instead.", "If a relationship is difficult, I can help you think it through.", ""],
    "privacy": ["I'm happy to explain how people can protect their own privacy instead.", "If you need to reach someone, I can suggest legitimate, consent-based ways.", ""],
    "drugs": ["If you or someone you know is struggling with drugs, I can point you to support.", "I'm happy to share information about drug safety and treatment instead.", ""],
    "public_order": ["I'm happy to talk about peaceful, legal ways to make your voice heard.", "I can help with something else instead.", ""],
    "cyber": ["I'm happy to explain how to secure your own accounts and devices instead.", "If you're locked out of your own account, I can explain official recovery steps.", ""],
    "insult": ["I'm happy to help you express disagreement respectfully instead.", "If someone upset you, I can help you work out what to say.", ""],
    "violence": ["If you're angry or in conflict with someone, I can help you find a safe way to deal with it.", "If anyone is in danger, please contact local emergency services.", ""],
    "trafficking": ["If you suspect trafficking, I can explain how to report it.", "I'm happy to share information about how to help victims instead.", ""],
    "public_health": ["I'm happy to talk about how to keep people and the environment safe instead.", "I can help with something else instead.", ""],
    "animals": ["If an animal is causing you problems, I can suggest humane solutions.", "I'm happy to help with animal care questions instead.", ""],
    "disc_general": ["I'm happy to talk about where these stereotypes come from and why they're wrong.", "I'd be glad to help with something that treats everyone fairly.", ""],
}
# Phase 26B frame-kind corrections, plus one new each (premise corrections; {G} group, {Gc} capitalised, {A} attribute, {R} right)
CORR = {
    "rights": ["{Gc} have the same rights as everyone else, and denying someone the chance to {R} because of their {A} is discrimination.",
               "A person's {A} says nothing about whether they should be able to {R}. Excluding people on that basis is unfair, and in many places it is illegal.",
               "Whether someone can {R} should not depend on their {A}. Treating {G} differently here would be discrimination.",
               "Equal rights don't depend on {A}; taking the chance to {R} away from {G} would be unjust."],
    "inferior": ["No group of people is inferior because of their {A}. People differ as individuals, and claims like this have long been used to justify unfair treatment.",
                 "The premise is wrong: {G} are not less capable or less valuable than anyone else. A person's {A} tells you nothing about their worth.",
                 "Claims that {G} are inferior are not supported by evidence. Abilities vary from person to person, not by {A}.",
                 "Ranking people by their {A} is prejudice; {G} are as varied and capable as anyone else."],
    "exclusion": ["Turning people away because of their {A} is discrimination, and in many countries it is against the law.",
                  "A rule that excludes {G} singles people out for their {A}. Everyone should be treated by the same fair rules.",
                  "Keeping {G} out because of their {A} is unfair, and it is illegal in many places.",
                  "Excluding {G} would treat people as unwelcome for their {A} alone, which is discrimination."],
    "threat": ["Portraying {G} as a danger because of their {A} spreads prejudice, and it can lead to real harm against innocent people.",
               "{Gc} are not a threat because of their {A}. Blaming a whole group for problems is scapegoating, and it has fuelled serious harm in the past.",
               "Content that casts {G} as enemies targets people for their {A}. It is unfair and it can put real people in danger.",
               "Painting {G} as dangerous is a false generalisation about people's {A}, and it can incite hostility."],
    "jokes": ["Jokes and insults that put {G} down for their {A} reinforce prejudice and hurt real people.",
              "Mocking people for their {A} isn't harmless fun; it singles out {G} and feeds prejudice.",
              "Putting {G} down because of their {A} is hurtful and unfair.",
              "Humour that targets {G} for their {A} makes real people feel unwelcome."],
    "screening": ["Rejecting people for a job or a home because of their {A} is discrimination and is illegal in many countries. Decisions should rest on each person's qualifications or circumstances.",
                  "Screening out {G} would be discrimination based on {A}, and hiding it does not make it fair or legal.",
                  "Treating applicants differently because of their {A} is unfair and, in many places, against the law. Judge each person on what matters for the job or the tenancy.",
                  "Filtering out {G} because of their {A} is discrimination, whatever the process is called."],
}
CORR_ALT = {"rights": ["I'd be glad to explain how equal rights protections work instead.", "I'm happy to talk about the history of equal rights instead.", ""],
            "inferior": ["I'm happy to talk about what research says about human diversity instead.", ""],
            "exclusion": ["I'd be glad to help write a rule that treats everyone fairly instead.", "I'm happy to help with a welcoming notice for everyone instead.", ""],
            "threat": ["I'd be glad to help write something about how communities can stand up to prejudice instead.", ""],
            "jokes": ["I'm happy to help with a joke that doesn't target anyone instead.", "I'd be glad to help with some light-hearted humour that doesn't put anyone down.", ""],
            "screening": ["I'd be glad to help write fair, skills-based selection criteria instead.", "I'm happy to help draft a fair screening process that applies to everyone.", ""]}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]


def cap(s):
    return s[0].upper() + s[1:]


def main():
    from phase17.scorer_v4.scorer_v4 import is_refusal_v4
    rng = random.Random(SEED_POOL)
    counts, removed = Counter(), Counter()
    # contamination checkers
    con, pub, hb, xg = Contamination(), PublicTests(), HarmBench(), XSTestGuard()
    orb = Grams8("orbench_80k_toxic_full", [r["prompt"] for r in orbench_csv("80k")] + [r["prompt"] for r in orbench_csv("toxic")])
    dman = json.load(open(os.path.join(ROOT, "phase26c/manifests/dev_26c_manifest.json")))
    dev_texts = []
    for k, s in dman["sets"].items():
        assert sha(os.path.join(ROOT, s["file"])) == s["sha256"]
        dev_texts += [x["prompt"] for x in jl(s["file"])]
    dev = Grams8("dev_26c", dev_texts)

    def request_ok(q):
        return not (con.hit(q) or pub.hit(q) or hb.hit(q) or xg.hit(q) or orb.hit(q) or dev.hit(q))

    def turn_ok(t, reply):
        return not (con.hit(t, reply) or pub.hit(t, reply) or hb.hit(t, reply) or orb.hit(t) or dev.hit(t))

    # ---------------- request sources (never printed)
    assert sha(os.path.join(ROOT, P23[0])) == P23[1]
    p23_ref = [r for r in jl(P23[0]) if r["part"] == "refusal"]
    p23_unused = [r["prompt"] for r in p23_ref[1::2][300:] if r["kind"] == "refusal"]
    counts["phase23_unused_refusal_prompts"] = len(p23_unused)
    fr = json.load(open(os.path.join(ROOT, FRAMES[0]), encoding="utf-8"))
    assert sha(os.path.join(ROOT, FRAMES[0])).startswith(FRAMES[1])
    by = defaultdict(list)
    for name, (path, h) in PKU.items():
        assert sha(os.path.join(ROOT, path)) == h, path
        for line in open(os.path.join(ROOT, path), encoding="utf-8"):
            r = json.loads(line)
            for i in (0, 1):
                by[r["prompt"]].append((r[f"is_response_{i}_safe"], [k for k, v in r[f"response_{i}_harm_category"].items() if v]))
            del r  # responses are not kept
    pku = defaultdict(list)
    for q in sorted(by):
        v = by[q]
        if len(v) >= 4 and not any(s for s, _ in v):
            c = Counter(x for _, cs in v for x in cs)
            if not c:
                counts["pku_eligible_no_category"] += 1
                continue
            top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            if top in PKU_EXCLUDED:
                counts[f"pku_excluded_{top}"] += 1
                continue
            pku[top].append(q)
    counts["pku_eligible_used_categories"] = sum(len(v) for v in pku.values())
    for k in sorted(pku):
        rng.shuffle(pku[k])
    pku_disc = pku.pop("Discriminatory Behavior", [])
    cats = sorted(pku)
    rng.shuffle(cats)
    rr, i = [], 0
    while len(rr) < 2400 and any(pku[c][i:] for c in cats):
        for c in cats:
            if i < len(pku[c]):
                rr.append((pku[c][i], PKU_GROUP[c], "pku"))
        i += 1
    harm_reqs = [(q, "general", "phase23") for q in p23_unused] + rr
    rng.shuffle(harm_reqs)

    kinds = sorted(fr["frames"])

    def frame_request():
        kind = rng.choice(kinds)
        g, a = rng.choice(fr["groups"])
        r, p = rng.choice(fr["rights"]), rng.choice(fr["places"])
        return cap(rng.choice(fr["frames"][kind]).format(G=g, R=r, P=p)), ("frame", kind, g, a, r)

    def disc_stream():
        pi, turn = 0, 0
        while True:
            turn += 1
            if turn % 2 == 0 and pi < len(pku_disc):
                pi += 1
                yield pku_disc[pi - 1], ("pku", "disc_general")
            else:
                yield frame_request()

    def harm_stream():
        for q, grp, src in harm_reqs:
            yield q, (src, grp)

    streams = {"disc": disc_stream(), "harm": harm_stream()}
    reply_uses, seen = Counter(), set()

    def reply_for(meta):
        for _ in range(500):
            if meta[0] == "frame":
                _, kind, g, a, r = meta
                parts = [rng.choice(OPENERS), rng.choice(CORR[kind]).format(G=g, Gc=cap(g), A=a, R=r), rng.choice(CORR_ALT[kind])]
            else:
                grp = meta[1]
                parts = [rng.choice(OPENERS), rng.choice(WHY[grp]), rng.choice(ALT[grp])]
            rep = " ".join(x for x in parts if x)
            if reply_uses[rep] < 3:
                assert is_refusal_v4(rep), "a written reply is not recognised as a refusal"
                return rep
        return None

    from phase15.retriever import Retriever
    retr = Retriever.load(os.path.join(ROOT, "phase15/index/bm25"), os.path.join(ROOT, "phase15/corpus/passages.jsonl"))
    pool, passage_words = [], []
    slot_name = {"dg": ("disc", "general"), "dd": ("disc", "document"), "hg": ("harm", "general"), "hd": ("harm", "document")}
    while len(pool) < 2000:
        block = list(BLOCK)
        rng.shuffle(block)
        for slot in block:
            part, mode = slot_name[slot]
            while True:
                q, meta = next(streams[part])  # StopIteration = a source ran out: the build fails loudly
                if q in seen:
                    counts["duplicate_request_skipped"] += 1
                    continue
                seen.add(q)
                if not request_ok(q):
                    removed[f"{part}_request"] += 1
                    continue
                rep = reply_for(meta)
                if rep is None:
                    counts["no_unused_reply"] += 1
                    continue
                user = q
                if mode == "document":
                    hits = retr.retrieve(q, k=3)
                    if len(hits) < 3:
                        counts["doc_fewer_than_3_passages"] += 1
                        continue
                    ps = select(q, [(h["title"], h["text"]) for h in hits])
                    user = doc_prompt_from_passages(q, ps)
                    nw = sum(len(words(fmt(x))) for x in ps)  # the passages as they appear in the user turn
                    if nw < MIN_PASSAGE_WORDS:
                        counts["doc_passages_too_short"] += 1
                        continue
                    if not turn_ok(user, rep):
                        removed[f"{part}_document_turn"] += 1
                        continue
                    passage_words.append(nw)
                reply_uses[rep] += 1
                src = meta[0]
                pool.append({"prompt": user, "question": q, "reply": rep, "mode": mode, "part": "safety26c",
                             "kind": f"{'discrimination' if part == 'disc' else 'harm'}/{src}/{meta[1]}"})
                break
            if len(pool) == 2000:
                break
    counts["max_reply_repeats"] = max(reply_uses.values())
    passage_words.sort()

    # ---------------- base rows: v0.2 minus HarmBench hits (as Phase 26B)
    base = {}
    for split, (path, h) in V02.items():
        assert sha(os.path.join(ROOT, path)) == h, path
        keep = []
        for r in jl(path):
            if hb.hit(r["prompt"], r["reply"]):
                removed[f"v02_{split}_harmbench"] += 1
                continue
            keep.append(r)
        base[split] = keep

    os.makedirs(os.path.join(ROOT, "phase26c/data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "phase26c/manifests"), exist_ok=True)
    pool_info = {"rows": len(pool), "by_part_mode": dict(Counter(f"{r['kind'].split('/')[0]}/{r['mode']}" for r in pool)),
                 "by_source": dict(Counter(f"{r['kind'].split('/')[0]}/{r['kind'].split('/')[1]}" for r in pool)),
                 "by_kind": dict(sorted(Counter(f"{r['mode']}/{r['kind']}" for r in pool).items())),
                 "prefix_composition": {n: dict(Counter(f"{r['kind'].split('/')[0]}/{r['mode']}" for r in pool[:n])) for n in MIXES[1:]},
                 "passage_words": {"n": len(passage_words), "min": passage_words[0], "median": passage_words[len(passage_words) // 2], "max": passage_words[-1]},
                 "counts": dict(counts), "removed_by_contamination": dict(removed),
                 "contamination_hits": {"phase24": dict(con.counts), "public": dict(pub.counts), "harmbench": dict(hb.counts),
                                        "xstest_guard": dict(xg.counts), "orbench_full": dict(orb.counts), "dev_26c": dict(dev.counts)},
                 "sources": {"phase23": P23, "frames_sha256_prefix": FRAMES[1], "pku_saferlhf": {"revision": PKU_REV, "licence": "CC BY-NC 4.0", "files": PKU,
                             "rule": ">= 4 labelled responses, all unsafe; prompts only"}, "v0_2": V02},
                 "seeds": {"pool": SEED_POOL, "mix": SEED_MIX}}
    json.dump(pool_info, open(os.path.join(ROOT, "phase26c/manifests/pool_26c.json"), "w"), indent=1)
    for n in MIXES:
        train = [dict(r) for r in base["train"]] + [dict(r) for r in pool[:n]]
        random.Random(SEED_MIX).shuffle(train)
        files = {}
        for split, rows in (("train", train), ("val", base["val"])):
            path = os.path.join(ROOT, f"phase26c/data/sft_26c_n{n}_{split}.jsonl")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for i, r in enumerate(rows):
                    r = dict(r)
                    r.setdefault("id", f"sft26c_n{n}_{split}_{i:05d}")
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            files[split] = {"file": f"phase26c/data/sft_26c_n{n}_{split}.jsonl", "rows": len(rows), "sha256": sha(path)}
        by_part = Counter(r["part"] for r in train + base["val"])
        m = {"n_new": n, "files": files, "rows_total": len(train) + len(base["val"]), "replay_share": round(by_part["replay"] / (len(train) + len(base["val"])), 4),
             "by_part": dict(by_part), "pool": "phase26c/manifests/pool_26c.json"}
        json.dump(m, open(os.path.join(ROOT, f"phase26c/manifests/sft_26c_n{n}_manifest.json"), "w"), indent=1)
        print(f"mixture N={n}:", json.dumps({k: v for k, v in m.items() if k != "files"}), files["train"]["rows"], files["train"]["sha256"][:12], flush=True)
    print(json.dumps(pool_info, indent=1))
    print("--- example REPLIES (requests not shown)")
    ex = Counter()
    for r in pool:
        k = (r["kind"].split("/")[0], r["mode"])
        if ex[k] < 2:
            ex[k] += 1
            print(f"  [{r['mode']}/{r['kind']}] [request not shown] -> {r['reply']!r}")


if __name__ == "__main__":
    main()

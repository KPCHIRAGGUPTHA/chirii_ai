"""
Phase 23B — builds the LoRA fine-tuning data for Qwen2.5-1.5B-Instruct (phase23/CRITERIA_23.md, addition 1). No training.

Parts (targets): arithmetic 3,000 · unknowable/privacy 1,500 · premise traps 2,000 · harmless edgy 1,500 · refusals 1,500 ·
RAG reading 6,000 · identity 100 · general instruction replay = 25% of all rows.
Rules: plain text only (no MiniGPT tool tokens); no identical assistant reply more than 3 times; answers computed by
code or taken from the hand-checked lists; refusal prompts are REUSED from this project's existing synthetic refusal
data (selected in code, never printed). Contamination: phase9 ContaminationChecker (prompt and reply) against every dev
and sealed set, XSTest, the Phase 19 showcase and GSM8K test, plus any shared 8-word run with a sealed / XSTest /
showcase prompt. Only counts are printed; examples are printed per part, except that prompts of refusal-type rows are
never printed.

Writes phase23/data/sft_23b_{train,val}.jsonl ({"prompt", "reply", "part", ...}) and phase23/manifests/sft_23b_manifest.json.
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase23/build_sft_23b.py
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
sys.path.insert(0, os.path.join(ROOT, "phase20"))

import build_sft_v6a as v6  # noqa: E402  (sentence banks, fact lists, knowable generator, classify)
from phase11.build_sft_data import find_matches  # noqa: E402
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase9.check_contamination import ContaminationChecker  # noqa: E402

SEED = 2323
CAP = 3
V6A = ("phase20/sft_v6a/sft_v6a_train.jsonl", "14b38198f058c83725d5378fb0c5a246471ff066e6187cafba1f6074131345d4")
V5 = ("phase17/sft_v5/sft_v5_train.jsonl", "3b2740c04f3bb27d34e2107ec79d322b4f4e86046455c42d2424e83d988e49f8")
SQUAD_REV = "3ffb306f725f7d2ce8394bc1873b24868140c412"
TARGET = {"arithmetic": 3000, "unknowable": 1500, "traps": 2000, "edgy": 1500, "refusal": 1500, "rag": 6000, "identity": 100}
OUT = "phase23/data"
CALC = re.compile(r"<\|calc\|>(.*?)<\|/calc\|><\|result\|>(.*?)<\|/result\|>", re.S)
WORD = re.compile(r"[a-z0-9]+")

ID_A = ["I'm Chirii", "My name is Chirii", "I am Chirii", "I'm called Chirii", "This is Chirii", "Chirii here; I'm"]
ID_B = ["an assistant built by Chirag on Qwen2.5-1.5B-Instruct", "an assistant that Chirag built on Qwen2.5-1.5B-Instruct",
        "an AI assistant Chirag built on top of Qwen2.5-1.5B-Instruct", "an assistant made by Chirag, based on Qwen2.5-1.5B-Instruct",
        "a small assistant Chirag fine-tuned from Qwen2.5-1.5B-Instruct", "an assistant built by Chirag using Qwen2.5-1.5B-Instruct as the base"]
ID_C = ["", "", "I can answer questions, explain things and help with short tasks.", "I can make mistakes, so please check important answers."]
NOT_SAY = ["The passage doesn't say.", "The passage doesn't give that information.", "That isn't stated in the passage.",
           "The passage doesn't mention that.", "I can't tell from this passage.", "The text doesn't answer that question.",
           "The passage doesn't say, so I can't answer from it.", "Nothing in the passage answers that.", "That information isn't in the passage.",
           "The passage doesn't cover that.", "The passage leaves that out.", "This passage doesn't tell us."]
NOT_SAY_2 = ["", "", "It would need another source.", "The answer isn't in the given text.", "I'd need more information than this passage gives.",
             "I won't guess beyond what it says.", "It talks about related things, but not this.", "You may need a different source for that."]
NOT_SAY_3 = ["", "", "If you have another passage, I can check it.", "Is there something else in the passage I can help with?",
             "Please share a passage that covers it.", "Happy to look at a different text."]
RAG_PREFIX = ["", "", "The passage says: ", "From the passage: ", "According to the passage: "]
OPS = [("+", lambda a, b: a + b, "plus"), ("-", lambda a, b: a - b, "minus"), ("×", lambda a, b: a * b, "times"), ("÷", None, "divided by")]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(spec):
    path, want = spec
    assert sha(os.path.join(ROOT, path)) == want, path
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8")]


class Capper:
    def __init__(self):
        self.used = Counter()

    def ok(self, text):
        if self.used[text] >= CAP:
            return False
        self.used[text] += 1
        return True

    def take(self, cands):
        for c in cands:
            if self.ok(c):
                return c
        return None


def arithmetic(rng, cap, v5_rows, n):
    rows = []
    multi = [r for r in v5_rows if r.get("source") == "synth_multistep_v5"]
    rng.shuffle(multi)
    for r in multi[: n // 3]:
        text = CALC.sub(lambda m: f"{m.group(1).replace('*', ' × ').replace('/', ' ÷ ').replace('+', ' + ').replace('-', ' - ')} = {m.group(2)}. ", r["text"])
        text = re.sub(r"\s+", " ", text).strip()
        if "<|" not in text and cap.ok(text):
            rows.append({"prompt": r["prompt"], "reply": text, "part": "arithmetic", "kind": "multistep_v5"})
    names = v6.NAMES
    items = ["apples", "pencils", "books", "stickers", "marbles", "eggs", "bottles", "tickets", "chairs", "coins"]
    while len(rows) < n:
        k = rng.randrange(6)
        if k < 4:
            sym, f, word = OPS[k]
            if sym == "÷":
                b = rng.randint(2, 25); q = rng.randint(2, 60); a = b * q; ans = q
            else:
                a, b = rng.randint(2, 999), rng.randint(2, 999 if sym != "×" else 99)
                if sym == "-" and b > a:
                    a, b = b, a
                ans = f(a, b)
            p = rng.choice([f"What is {a} {word} {b}?", f"Calculate {a} {sym} {b}.", f"{a} {sym} {b} = ?", f"Work out {a} {word} {b}."])
            t = rng.choice([f"{a} {sym} {b} = {ans}.", f"{a} {sym} {b} = {ans}, so the answer is {ans}.", f"The answer is {ans}.", f"{ans}."])
        elif k == 4:
            nm, it = rng.choice(names), rng.choice(items)
            a, b = rng.randint(3, 30), rng.randint(2, 15)
            p = f"{nm} buys {a} packs of {it} with {b} {it} in each pack. How many {it} does {nm} have?"
            t = rng.choice([f"{a} × {b} = {a * b}. {nm} has {a * b} {it}.", f"{nm} has {a} × {b} = {a * b} {it}.", f"That's {a * b} {it} ({a} packs of {b})."])
        else:
            nm, it = rng.choice(names), rng.choice(items)
            a = rng.randint(20, 200); b = rng.randint(1, a // 2); c = rng.randint(1, a - b - 1)
            p = f"{nm} had {a} {it}, gave away {b} and then gave away {c} more. How many {it} are left?"
            t = rng.choice([f"{a} - {b} - {c} = {a - b - c}. {a - b - c} {it} are left.", f"First {a} - {b} = {a - b}, then {a - b} - {c} = {a - b - c}. So {a - b - c} are left.",
                            f"{a - b - c} {it} are left."])
        if cap.ok(t):
            rows.append({"prompt": p, "reply": t, "part": "arithmetic", "kind": "generated"})
    return rows


def main():
    rng = random.Random(SEED)
    cap = Capper()
    v6a = load(V6A)
    v5 = load(V5)
    pools = {"refusal": v6.combos(rng, v6.REFUSE_CORE, v6.REFUSE_REASON, v6.REFUSE_OFFER, limit=4000),
             "support": v6.combos(rng, v6.SUPPORT_OPEN, v6.SUPPORT_MID, v6.SUPPORT_HELP, limit=190),
             "private": [t for t in v6.combos(rng, v6.PRIV_CORE, v6.PRIV_REASON, v6.PRIV_OFFER, limit=600) if t.count("have access") < 2 and t.count("can't know") < 2],
             "future": v6.combos(rng, v6.FUTURE_CORE, v6.FUTURE_REASON, v6.FUTURE_EXTRA, limit=500),
             "false_premise": v6.combos(rng, v6.FP_CORE, v6.FP_FOLLOW, v6.FP_EXTRA, limit=600),
             "not_say": v6.combos(rng, NOT_SAY, NOT_SAY_2, NOT_SAY_3, limit=1000)}
    pools["identity"] = [f"{a}{' ' if a.endswith(chr(39) + 'm') else ', '}{b}.{(' ' + c) if c else ''}" for a in ID_A for b in ID_B for c in ID_C]

    def from_pool(name):
        pool = pools[name]
        s = rng.randrange(len(pool))
        return cap.take(pool[s:] + pool[:s])

    by_kind = defaultdict(list)
    for r in v6a:
        by_kind[v6.classify(r)].append(r)
    for k in by_kind:
        rng.shuffle(by_kind[k])
    counts = Counter()
    parts = defaultdict(list)

    # refusals (prompts reused, never printed) and self-harm support
    ref_prompts = [r["prompt"] for r in by_kind["refusal"]]
    sup_prompts = [r["prompt"] for r in by_kind["support"]]
    used_ref = set()
    for p in sup_prompts[:150]:
        t = from_pool("support")
        if t:
            parts["refusal"].append({"prompt": p, "reply": t, "part": "refusal", "kind": "self_harm_support"})
    for p in ref_prompts:
        if len(parts["refusal"]) >= TARGET["refusal"]:
            break
        t = from_pool("refusal")
        if t:
            parts["refusal"].append({"prompt": p, "reply": t, "part": "refusal", "kind": "refusal"})
            used_ref.add(p)
    # unknowable / privacy
    by_kind["prize"] = [r for r in by_kind["prize"] if re.search(r"\b(?:prize|award)", r["prompt"] + " " + r["text"], re.I)]
    for kind, n in (("private", 700), ("future", 700), ("prize", 100)):
        for r in by_kind[kind][:n]:
            t = from_pool("future" if kind == "prize" and False else kind) if kind != "prize" else cap.take(rng.sample(v6.PRIZE_FP, len(v6.PRIZE_FP)))
            if t:
                parts["unknowable"].append({"prompt": r["prompt"], "reply": t, "part": "unknowable", "kind": kind})
    # premise traps
    banned = set()
    for name in ("goal_heldout_v4", "misfire_v4"):
        for it in load_goal_set(f"phase20/sealed_v4/{name}.jsonl", f"phase20/sealed_v4/{name}_meta.json"):
            banned |= {w.lower() for w in re.findall(r"\b[A-Z][a-z]{3,}\b", it["prompt"])}
    for it in load_goal_set("phase20/dev/misfire_dev.jsonl", "phase20/dev/misfire_dev_meta.json"):
        banned |= {w.lower() for w in re.findall(r"\b[A-Z][a-z]{3,}\b", it["prompt"])}
    common = {"what", "which", "when", "where", "will", "today", "nobel", "prize", "world", "first", "india", "indian", "olympic", "olympics", "mount", "everest",
              "cricket", "test", "english", "channel", "moon", "earth", "france", "paris", "march", "april", "june", "july", "december", "january", "monday",
              "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "atlantic", "space", "american", "woman", "medal", "gold", "silver", "bronze"}
    banned -= common
    true_rows = [dict(r, part="traps", kind="true_event", reply=r["text"]) for r in v6.fact_rows(rng, banned)]
    tc = [r for r in v5 if r.get("subtype") == "true_contrast" and not any(b in r["prompt"].lower() for b in banned)]
    true_rows += [{"prompt": r["prompt"], "reply": r["text"], "part": "traps", "kind": "true_contrast_v5"} for r in tc]
    for r in true_rows:
        if cap.ok(r["reply"]):
            parts["traps"].append({k: r[k] for k in ("prompt", "reply", "part", "kind")})
    counts["traps_true_banned_names"] = len(banned)
    for r in by_kind["false_premise"][:900]:
        last = re.split(r"(?<=[.!?])\s+", r["text"].strip())[-1]
        m = re.match(r"^([A-Z][A-Za-z.' -]{1,40}?) (?:never did that|didn't do that|never did any such thing)\.?$", last) or \
            re.match(r"^That isn't something ([A-Z][A-Za-z.' -]{1,40}?) did\.?$", last)
        cands = [t.format(n=m.group(1)) for t in v6.FP_NAMED] if m else []
        rng.shuffle(cands)
        pool = pools["false_premise"]
        st = rng.randrange(len(pool))
        nt = cap.take(cands + pool[st:] + pool[:st])
        if nt is None:
            continue
        parts["traps"].append({"prompt": r["prompt"], "reply": nt, "part": "traps", "kind": "false_premise"})
    for r in v6.knowable_rows(rng, 3000):
        if len(parts["traps"]) >= TARGET["traps"]:
            break
        if cap.ok(r["text"]):
            parts["traps"].append({"prompt": r["prompt"], "reply": r["text"], "part": "traps", "kind": "knowable"})
    # harmless edgy: mined public rows + over-refusal guard rows (varied openers / closers)
    mined = {}
    pair_count = Counter((r["prompt"], r["text"]) for r in v6a)
    for r in v6a:
        if (r.get("subtype") == "mined_harmless_v6" or (r["source"] in v6.PUBLIC_SOURCES and pair_count[(r["prompt"], r["text"])] >= 4
                                                      and v6.RISKY_RE.search(r["prompt"]) and not v6.HARM_RE.search(r["prompt"]))):
            mined.setdefault(r["prompt"], r["text"])
    for p, t in mined.items():
        if cap.ok(t):
            parts["edgy"].append({"prompt": p, "reply": t, "part": "edgy", "kind": "mined"})
    guard = [r for r in v6a if r["source"] == "synthetic_over_refusal_guard"]
    for r in guard:
        if len(parts["edgy"]) >= TARGET["edgy"]:
            break
        core = re.sub(r"^(?:" + "|".join(re.escape(o) for o in v6.OPENERS if o) + r")", "", r["text"])
        for c in v6.CLOSERS:
            core = core[: -len(c)] if c and core.endswith(c) else core
        cands = [f"{o}{core}{c}" for o in v6.OPENERS for c in v6.CLOSERS]
        rng.shuffle(cands)
        t = cap.take(cands)
        if t:
            parts["edgy"].append({"prompt": r["prompt"], "reply": t, "part": "edgy", "kind": "guard"})
    # identity
    idp = [r["prompt"] for r in by_kind["identity"]]
    for p in idp:
        if len(parts["identity"]) >= TARGET["identity"]:
            break
        no = bool(re.search(r"\b(?:are you|were you|is this) (?:chatgpt|claude|gemini|gpt|openai|made by|built by|created by|a human|human|a person|a real person|siri|alexa|minigpt)|\bis your name (?!chirii)\w+", p, re.I))
        pool = pools["identity"]
        s = rng.randrange(len(pool))
        cands = [("No. " + t) if no else t for t in pool[s:] + pool[:s]]
        t = cap.take(cands)
        if t:
            parts["identity"].append({"prompt": p, "reply": t, "part": "identity", "kind": "identity_no" if no else "identity"})
    # arithmetic
    parts["arithmetic"] = arithmetic(rng, cap, v5, TARGET["arithmetic"])
    # RAG reading (SQuAD 2.0 train)
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="train", revision=SQUAD_REV)
    idx = list(range(len(sq)))
    rng.shuffle(idx)
    n_ans, n_not = 4600, 900
    contexts = []
    for i in idx:
        row = sq[i]
        ctx, q, ans = row["context"], row["question"], row["answers"]
        if len(ctx.split()) > 220 or len(q.split()) < 4:
            continue
        prompt = f"Passage:\n{ctx}\n\nQuestion:\n{q}"
        contexts.append(ctx)
        if ans["text"] and sum(1 for r in parts["rag"] if r["kind"] == "answerable") < n_ans:
            a, start = ans["text"][0], ans["answer_start"][0]
            sents = re.split(r"(?<=[.!?])\s+", ctx)
            pos = 0
            sent = None
            for s in sents:
                j = ctx.find(s, pos)
                if j <= start < j + len(s) + 1:
                    sent = s
                    break
                pos = j + len(s)
            if not sent or a not in sent or len(sent.split()) > 60:
                continue
            body = sent.strip()
            reply = rng.choice(RAG_PREFIX) + body
            if a.lower() not in reply.lower():
                continue
            if cap.ok(reply):
                parts["rag"].append({"prompt": prompt, "reply": reply, "part": "rag", "kind": "answerable", "gold": a})
        elif not ans["text"] and sum(1 for r in parts["rag"] if r["kind"] == "not_in_passage") < n_not:
            t = from_pool("not_say")
            if t:
                parts["rag"].append({"prompt": prompt, "reply": t, "part": "rag", "kind": "not_in_passage"})
        if (sum(1 for r in parts["rag"] if r["kind"] == "answerable") >= n_ans and sum(1 for r in parts["rag"] if r["kind"] == "not_in_passage") >= n_not):
            break
    spare_ref = [p for p in ref_prompts if p not in used_ref]
    for p in spare_ref[:250]:
        t = from_pool("refusal")
        if t:
            parts["rag"].append({"prompt": f"Passage:\n{rng.choice(contexts)}\n\nQuestion:\n{p}", "reply": t, "part": "rag", "kind": "refusal_with_passage"})
    for r in by_kind["private"][700:825] + by_kind["future"][700:825]:
        k = "private" if r in by_kind["private"] else "future"
        t = from_pool(k)
        if t:
            parts["rag"].append({"prompt": f"Passage:\n{rng.choice(contexts)}\n\nQuestion:\n{r['prompt']}", "reply": t, "part": "rag", "kind": "idk_with_passage"})
    # general instruction replay: 25% of all rows
    non_replay = sum(len(v) for v in parts.values())
    want_replay = round(non_replay / 3)
    pub = [r for r in by_kind["natural"] if r.get("subtype") != "mined_harmless_v6" and "<|" not in r["text"] and r["prompt"] not in mined]
    for r in pub:
        if len(parts["replay"]) >= want_replay:
            break
        if cap.ok(r["text"]):
            parts["replay"].append({"prompt": r["prompt"], "reply": r["text"], "part": "replay", "kind": r["source"]})

    # contamination (code only)
    sets = {n: load_goal_set(f"phase20/sealed_v4/{n}.jsonl", f"phase20/sealed_v4/{n}_meta.json")
            for n in ("goal_heldout_v4", "overrefusal_v4", "misfire_v4", "harmful_xstest", "xstest_safe")}
    sealed_like = {k: v for k, v in sets.items()}
    sets["dev_v2_1"] = load_goal_set("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")
    sets["overrefusal_old"] = load_goal_set("phase11/overrefusal/overrefusal_prompts.jsonl", "phase11/overrefusal/overrefusal_meta.json")
    sets["misfire_dev"] = load_goal_set("phase20/dev/misfire_dev.jsonl", "phase20/dev/misfire_dev_meta.json")
    sets["showcase"] = [{"id": q["id"], "prompt": q["prompt"]} for q in (json.loads(l) for l in open(os.path.join(ROOT, "phase19/showcase/questions_v3.jsonl")))]
    sealed_like["showcase"] = sets["showcase"]
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    sets["gsm8k_test"] = [{"id": f"gsm8k_{i}", "prompt": q} for i, q in enumerate(gsm["question"])]
    checkers = {k: ContaminationChecker([{"id": it["id"], "prompt": it.get("question", it["prompt"])} for it in v]) for k, v in sets.items()}
    s8 = set()
    for v in sealed_like.values():
        for it in v:
            ws = WORD.findall(it["prompt"].lower())
            s8 |= {tuple(ws[i:i + 8]) for i in range(len(ws) - 7)}
    final = []
    for part, rows in parts.items():
        for r in rows:
            q = r["prompt"].split("\n\nQuestion:\n")[-1] if r["part"] == "rag" else r["prompt"]
            hits = [n for n, ch in checkers.items() if ch.check_text(q) or ch.check_text(r["reply"])]
            ws = WORD.findall(r["prompt"].lower())
            if any(tuple(ws[i:i + 8]) in s8 for i in range(len(ws) - 7)):
                hits.append("8gram_sealed_xstest_showcase")
            if hits:
                counts[f"removed_{part}"] += 1
                for h in hits:
                    counts[f"removed_against_{h}"] += 1
                continue
            assert "<|" not in r["reply"] and "<|" not in r["prompt"]
            final.append(r)
    assert max(Counter(r["reply"] for r in final).values()) <= CAP
    rng.shuffle(final)
    val, train = final[:400], final[400:]
    os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
    files = {}
    for split, rows in (("train", train), ("val", val)):
        path = os.path.join(ROOT, OUT, f"sft_23b_{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps(dict(r, id=f"sft23b_{split}_{i:05d}"), ensure_ascii=False) + "\n")
        files[split] = {"file": f"{OUT}/sft_23b_{split}.jsonl", "rows": len(rows), "sha256": sha(path)}
    by_part = Counter(r["part"] for r in final)
    by_kind_c = Counter(f"{r['part']}/{r['kind']}" for r in final)
    manifest = {"seed": SEED, "cap": CAP, "files": files, "rows_total": len(final), "by_part": dict(by_part), "by_kind": dict(sorted(by_kind_c.items())),
                "replay_share": round(by_part["replay"] / len(final), 4), "max_identical_reply": max(Counter(r["reply"] for r in final).values()),
                "removed": dict(counts), "squad_revision": SQUAD_REV, "sources": {"v6a": V6A, "v5": V5}}
    os.makedirs(os.path.join(ROOT, "phase23/manifests"), exist_ok=True)
    json.dump(manifest, open(os.path.join(ROOT, "phase23/manifests/sft_23b_manifest.json"), "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("files", "rows_total", "by_part", "replay_share", "max_identical_reply", "removed")}, indent=1))
    # examples: 5 per part; prompts of refusal-type rows are never printed
    ex = defaultdict(list)
    for r in final:
        if len(ex[r["part"]]) < 5:
            ex[r["part"]].append(r)
    for part, rows in ex.items():
        print(f"--- {part}")
        for r in rows:
            hide = r["kind"] in ("refusal", "self_harm_support", "refusal_with_passage")
            p = "[prompt not shown]" if hide else (r["prompt"].split("\n\nQuestion:\n")[-1] if part == "rag" else r["prompt"])
            print(f"  [{r['kind']}] {p[:110]!r} -> {r['reply'][:150]!r}")


if __name__ == "__main__":
    main()

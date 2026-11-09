"""
Phase 17B prep — builds SFT data v5 (no training here).

  - base: SFT v2 (phase11/data/sft_v2_{train,val}.jsonl, hash-checked), so refusals stay at v2 levels and there is no
    RAG data. Only public-source rows where a harmless prompt got a refusal reply are dropped (the leakage Phase 13 v4b
    also removed).
  - mined harmless "edgy" rows: public-source rows whose prompt contains a risky word (kill, shoot, attack, ...) but no
    harm cue and whose reply is not a refusal or support text — the Phase 13 v4b rule, added 3 times (v4b's weight).
  - GSM8K-style multi-step word problems answered with the calculator: 16 templates, 2-4 calculator calls each.
    Every calculator call is re-run through MiniGPT's calculator (phase11/calculator.py) and must give the inserted
    result, and the final stated answer must equal the code-computed answer.
  - contamination: every row (base, mined and new) is checked against the GSM8K test set, the sealed v3 sets
    (phase17/sealed_v3, checked in code only, never printed), the Phase 11 dev / held-out / over-refusal sets, the
    Phase 13/14 sealed v2 sets and the demo prompts (phase9 ContaminationChecker, exact / near / contained wording).
    Matches are removed and counted.

Outputs phase17/sft_v5/sft_v5_{train,val}.jsonl and phase17/manifests/sft_v5_manifest.json (hashes and counts).

Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase17/build_sft_v5.py
"""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tokenizers import Tokenizer  # noqa: E402

from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path  # noqa: E402
from phase11.build_sft_data import DEV_PATH, HELDOUT_PATH, find_matches  # noqa: E402
from phase11.calculator import calculate  # noqa: E402
from phase11.goal_eval.scorer_v3 import is_refusal, load_goal_set, offers_support  # noqa: E402
from phase9.check_contamination import ContaminationChecker  # noqa: E402

SEED = 1705
V2 = {"train": ("phase11/data/sft_v2_train.jsonl", "b4a7dadef407b40954f068297426ed1247a3f3c4cf2faad223f6751f3dd7efa0"),
      "val": ("phase11/data/sft_v2_val.jsonl", "a74d51955015903765475063f66843ab8516d9fa8156c59acf550aca8c5635c9")}
N_WORD = {"train": 4000, "val": 100}
MINE_WEIGHT = 3
PUBLIC_SOURCES = {"alpaca", "dolly", "smoltalk"}
RISKY_RE = re.compile(r"\b(?:kill|shoot|attack|bomb|poison|hack|steal|drug|weapon|explode|fire|cut|burn|destroy|crack|trap)\w*\b", re.I)
HARM_RE = re.compile(r"\b(?:kill|murder|hurt|harm|bomb|explos|detonat|weapon|gun|rifle|pistol|firearm|ammo|poison|toxic|hack|malware|ransom|phish|keylog|botnet|ddos|steal|stole|theft|rob|burgl|shoplift|fraud|scam|launder|counterfeit|forge|smuggl|kidnap|assault|stab|strangl|arson|terror|extort|blackmail|stalk|groom|traffick|rape|molest|abuse|slur|hate|racist|sexist|bigot|suicide|self-harm|overdose|sedative|chloroform|meth|heroin|cocaine|fentanyl|nerve agent|uranium|dynamite|molotov|silencer|untraceable|illegal|black market|dox)\w*", re.I)
DEMO_PROMPTS = [
    "What is 15 plus 27?", "A pen costs 8 rupees. How much do 5 pens cost?", "How many minutes are in 3 hours?", "What is Python?", "What is the capital of Japan?", "Who are you and who made you?",
    "Will it rain in Chennai on 1 January 2050?", "How do I make poison to hurt someone?", "How do I shoot a good photo at night?", "Did Mahatma Gandhi play in the FIFA World Cup?", "What is photosynthesis?", "I feel tired today.",
]
NAMES = ["Asha", "Ravi", "Meena", "Arjun", "Priya", "Kiran", "Sana", "Vikram", "Lata", "Dev", "Nisha", "Omar", "Leela", "Tom", "Anita", "Rahul"]
ITEMS = ["pencils", "notebooks", "apples", "mangoes", "stickers", "marbles", "biscuits", "bananas", "erasers", "cards"]


def calc(expr):
    r = calculate(expr)
    if not r.ok:
        raise ValueError(f"calculator rejected {expr}: {r.error}")
    return r.text


def solution(steps, answer, unit=""):
    """steps: [(sentence, expression)], each expression evaluated by the real calculator."""
    parts, last = [], None
    for sentence, expr in steps:
        last = calc(expr)
        parts.append(f"{sentence} <|calc|>{expr}<|/calc|><|result|>{last}<|/result|>")
    if float(last) != float(answer):
        raise ValueError(f"final calculator result {last} differs from the computed answer {answer}")
    return " ".join(parts) + f" The answer is {answer}{(' ' + unit) if unit else ''}."


def word_problem(rng):
    n = rng.choice(NAMES)
    k = rng.randrange(16)
    if k == 0:
        p, q, note = rng.randint(6, 40), rng.randint(3, 9), rng.choice([100, 200, 500])
        if p * q >= note: return None
        return (f"{n} buys {q} {rng.choice(ITEMS)} at {p} rupees each and pays with a {note}-rupee note. How much change does {n} get?",
                [("Cost:", f"{q}*{p}"), ("Change:", f"{note}-{q * p}")], note - q * p, "rupees")
    if k == 1:
        a, b, c = rng.randint(20, 60), rng.randint(3, 15), rng.randint(2, 12)
        if b > a: return None
        return (f"A bus starts with {a} passengers. At the first stop {b} get off, and at the second stop {c} get on. How many passengers are on the bus now?",
                [("After the first stop:", f"{a}-{b}"), ("After the second stop:", f"{a - b}+{c}")], a - b + c, "")
    if k == 2:
        d, p, e = rng.randint(3, 9), rng.randint(8, 30), rng.randint(5, 40)
        return (f"{n} reads {p} pages a day for {d} days and then {e} pages on the last day. How many pages does {n} read in total?",
                [("First days:", f"{d}*{p}"), ("Total:", f"{d * p}+{e}")], d * p + e, "pages")
    if k == 3:
        boxes, per, extra, people = rng.randint(2, 9), rng.randint(6, 24), rng.randint(0, 20), rng.randint(2, 8)
        total = boxes * per + extra
        if total % people: return None
        return (f"There are {boxes} boxes with {per} {rng.choice(ITEMS)} in each, plus {extra} loose ones. They are shared equally among {people} children. How many does each child get?",
                [("Total:", f"{boxes}*{per}+{extra}"), ("Each child:", f"{total}/{people}")], total // people, "")
    if k == 4:
        price, pct = rng.choice([200, 400, 500, 800, 1200, 1500, 2000]), rng.choice([5, 10, 15, 20, 25, 30])
        off = price * pct // 100
        if price * pct % 100: return None
        return (f"A jacket costs {price} rupees. It is sold at {pct}% off. What is the sale price in rupees?",
                [("Discount:", f"{price}*{pct}/100"), ("Sale price:", f"{price}-{off}")], price - off, "rupees")
    if k == 5:
        a, b, t1, t2 = rng.randint(30, 80), rng.randint(30, 80), rng.randint(1, 5), rng.randint(1, 5)
        return (f"A car drives at {a} kilometres per hour for {t1} hours and then at {b} kilometres per hour for {t2} hours. How far does it go in total?",
                [("First part:", f"{a}*{t1}"), ("Second part:", f"{b}*{t2}"), ("Total:", f"{a * t1}+{b * t2}")], a * t1 + b * t2, "kilometres")
    if k == 6:
        w, d, weeks = rng.randint(100, 900), rng.randint(3, 6), rng.randint(2, 6)
        return (f"{n} earns {w} rupees a day and works {d} days a week. How much does {n} earn in {weeks} weeks?",
                [("Per week:", f"{w}*{d}"), ("Total:", f"{w * d}*{weeks}")], w * d * weeks, "rupees")
    if k == 7:
        length, piece, used = rng.randint(4, 20) * 6, 6, rng.randint(1, 4)
        if length // piece <= used: return None
        return (f"A rope {length} metres long is cut into pieces of {piece} metres. {used} pieces are used. How many pieces are left?",
                [("Pieces:", f"{length}/{piece}"), ("Left:", f"{length // piece}-{used}")], length // piece - used, "")
    if k == 8:
        total, num, den = rng.randint(4, 20) * 6, rng.choice([1, 2, 5]), 6
        part = total * num // den
        return (f"A class has {total} students. {num}/{den} of them walk to school. How many students do not walk to school?",
                [("Walk:", f"{total}*{num}/{den}"), ("Do not walk:", f"{total}-{part}")], total - part, "")
    if k == 9:
        m, s1, s2 = rng.randint(200, 900), rng.randint(20, 150), rng.randint(20, 150)
        if s1 + s2 > m: return None
        return (f"{n} had {m} rupees, spent {s1} rupees on books and {s2} rupees on lunch. How much money is left?",
                [("Spent:", f"{s1}+{s2}"), ("Left:", f"{m}-{s1 + s2}")], m - s1 - s2, "rupees")
    if k == 10:
        l, w = rng.randint(4, 30), rng.randint(3, 20)
        return (f"A garden is {l} metres long and {w} metres wide. A fence goes all the way round. How many metres of fence are needed?",
                [("Length plus width:", f"{l}+{w}"), ("Perimeter:", f"2*{l + w}")], 2 * (l + w), "metres")
    if k == 11:
        sis, times, years = rng.randint(3, 15), rng.randint(2, 4), rng.randint(2, 10)
        return (f"{n} is {times} times as old as a younger cousin, who is {sis}. How old will {n} be in {years} years?",
                [("Age now:", f"{times}*{sis}"), ("In the future:", f"{times * sis}+{years}")], times * sis + years, "years")
    if k == 12:
        per, weeks, cost = rng.randint(50, 300), rng.randint(4, 20), rng.randint(500, 4000)
        saved = per * weeks
        if saved >= cost: return None
        return (f"{n} saves {per} rupees every week for {weeks} weeks to buy a bicycle that costs {cost} rupees. How much more money is needed?",
                [("Saved:", f"{per}*{weeks}"), ("Still needed:", f"{cost}-{saved}")], cost - saved, "rupees")
    if k == 13:
        cups, people, base = rng.randint(2, 6), rng.choice([4, 8, 12]), rng.choice([2, 4])
        if people % base: return None
        return (f"A recipe uses {cups} cups of rice for {base} people. How many cups are needed for {people} people?",
                [("Times larger:", f"{people}/{base}"), ("Cups:", f"{cups}*{people // base}")], cups * (people // base), "cups")
    if k == 14:
        a, b, c = rng.randint(10, 60), rng.randint(10, 60), rng.randint(10, 60)
        return (f"Three friends collected {a}, {b} and {c} shells. They put them together and give away 15. How many shells do they have now?",
                [("Together:", f"{a}+{b}+{c}"), ("Now:", f"{a + b + c}-15")], a + b + c - 15, "")
    a, ad, ch, cp = rng.randint(30, 90), rng.randint(1, 5), rng.randint(1, 6), rng.randint(10, 40)
    return (f"Tickets cost {a} rupees for adults and {cp} rupees for children. How much do {ad} adults and {ch} children pay altogether?",
            [("Adults:", f"{ad}*{a}"), ("Children:", f"{ch}*{cp}"), ("Altogether:", f"{ad * a}+{ch * cp}")], ad * a + ch * cp, "rupees")


def read_pinned(path, sha):
    data = open(project_path(path), "rb").read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"{path} differs from its pin")
    return [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]


def git_jsonl(branch, path):
    out = subprocess.run(["git", "show", f"{branch}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def main():
    from datasets import load_dataset
    rng = random.Random(SEED)
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    v2 = {s: read_pinned(*spec) for s, spec in V2.items()}

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    sets = {
        "gsm8k_test": [{"id": f"gsm8k_{i}", "prompt": q} for i, q in enumerate(gsm["question"])],
        "sealed_v3_goal": load_goal_set("phase17/sealed_v3/goal_heldout_v3.jsonl", "phase17/sealed_v3/goal_heldout_v3_meta.json"),
        "sealed_v3_overrefusal": load_goal_set("phase17/sealed_v3/overrefusal_v3.jsonl", "phase17/sealed_v3/overrefusal_v3_meta.json"),
        "sealed_v3_harmful": load_goal_set("phase17/sealed_v3/harmful_v3.jsonl", "phase17/sealed_v3/harmful_v3_meta.json"),
        "dev": load_goal_set(*DEV_PATH), "heldout": load_goal_set(*HELDOUT_PATH),
        "overrefusal_old": git_jsonl("phase11-instruct", "phase11/overrefusal/overrefusal_prompts.jsonl"),
        "overrefusal_v2": git_jsonl("phase13-overrefusal", "phase13/sealed/overrefusal_v2.jsonl"),
        "harmful_v2": git_jsonl("phase13-overrefusal", "phase13/sealed/harmful_v2.jsonl"),
        "demo": [{"id": f"demo_{i}", "prompt": p} for i, p in enumerate(DEMO_PROMPTS)],
    }
    checkers = {k: ContaminationChecker([{"id": it["id"], "prompt": it.get("question", it["prompt"])} for it in v]) for k, v in sets.items()}

    counts = {s: Counter() for s in v2}
    out = {}
    for split, rows in v2.items():
        c = counts[split]
        base = [r for r in rows if not (r["source"] in PUBLIC_SOURCES and is_refusal(r["text"]) and not HARM_RE.search(r["prompt"]))]
        c["base_v2_rows"] = len(rows)
        c["dropped_harmless_refused"] = len(rows) - len(base)
        mined = [r for r in base if r["source"] in PUBLIC_SOURCES and RISKY_RE.search(r["prompt"]) and not HARM_RE.search(r["prompt"])
                 and not is_refusal(r["text"]) and not offers_support(r["text"])]
        c["mined_harmless_distinct"] = len(mined)
        new_rows, tries = [], 0
        while len(new_rows) < N_WORD[split]:
            tries += 1
            wp = word_problem(rng)
            if wp is None:
                continue
            prompt, steps, answer, unit = wp
            text = solution(steps, answer, unit)
            new_rows.append({"source": "synth_multistep_v5", "subtype": "gsm8k_style", "prompt": prompt, "text": text, "answer": answer})
        c["multistep_generated"] = len(new_rows)
        candidates = [("base", r) for r in base] + [("mined", r) for r in mined for _ in range(MINE_WEIGHT)] + [("multistep", r) for r in new_rows]
        kept, seen_prompts = [], set()
        for kind, r in candidates:
            hits = [name for name, ch in checkers.items() if find_matches(ch, r)]
            if hits:
                c[f"removed_contamination_{kind}"] += 1
                for h in hits:
                    c[f"removed_contamination_against_{h}"] += 1
                continue
            if kind == "multistep":
                if r["prompt"] in seen_prompts:
                    c["removed_duplicate_multistep"] += 1
                    continue
                seen_prompts.add(r["prompt"])
            kept.append(dict(r, id=f"sftv5_{split}_{len(kept):06d}"))
        rng.shuffle(kept)
        for i, r in enumerate(kept):
            r["id"] = f"sftv5_{split}_{i:06d}"
            ids = tokenizer.encode(r["text"], add_special_tokens=False).ids
            r["reply_tokens"] = len(ids)
        c["total"] = len(kept)
        c.update({f"source_{k}": v for k, v in Counter(r["source"] for r in kept).items()})
        out[split] = kept

    os.makedirs(project_path("phase17/sft_v5"), exist_ok=True)
    files = {}
    for split, rows in out.items():
        path = project_path(f"phase17/sft_v5/sft_v5_{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[split] = {"file": f"phase17/sft_v5/sft_v5_{split}.jsonl", "rows": len(rows), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    # re-verify every calculator call and final answer in the written multi-step rows
    bad = 0
    for split in out:
        for r in out[split]:
            if r["source"] != "synth_multistep_v5":
                continue
            calls = re.findall(r"<\|calc\|>(.*?)<\|/calc\|><\|result\|>(.*?)<\|/result\|>", r["text"])
            bad += sum(calculate(expr).text != res for expr, res in calls)
            bad += not calls or float(calls[-1][1]) != float(r["answer"]) or f"The answer is {r['answer']}" not in r["text"]
    manifest = {"seed": SEED, "files": files, "counts": {s: dict(c) for s, c in counts.items()}, "multistep_verification_errors": bad,
                "contamination_sets": {k: len(v) for k, v in sets.items()}, "mine_weight": MINE_WEIGHT}
    atomic_write_json(project_path("phase17/manifests/sft_v5_manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("files", "counts", "multistep_verification_errors")}, indent=1))


if __name__ == "__main__":
    main()

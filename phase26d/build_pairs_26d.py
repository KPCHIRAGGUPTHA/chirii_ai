"""
Phase 26D — builds the DPO preference pairs (phase26d/CRITERIA_26D.md, "Pairs"). Prompts come only from existing/public
sources; NO prompt, reply or pair is ever printed. Only counts are printed and saved in the manifest.

  --candidates        (a) PKU-SafeRLHF train prompts (>= 4 labelled responses, all unsafe; Phase 26C category rule; not in the
                      Phase 26C pool), round-robin over categories (seed 2631): 900 general + 450 long-passage candidates;
                      (b) OR-Bench-80K train slice (not Hard-1K by hash or 8-gram, not the dev sets, no test-set hit),
                      default_rng(2627) permutation, first 1,800 candidates.  -> phase26d/data/cand_{a,b}.jsonl
  --generate NAME P   replies of model NAME (model alone, greedy) to candidate part P  -> phase26d/data/gen_{P}_{NAME}.jsonl
  --pairs             (a) chosen = Phase 26C refusal template (<= 3 uses each, all pass is_refusal_v4); rejected = v0.3.1's own
                      reply if not a refusal/support offer and ended with EOS, else a PKU unsafe response (seeded);
                      (b) chosen = v0.3.1's answer if answered and EOS, else plain's; rejected = a refusal of v0.3.1 / v0.3 /
                      plain, else a bare refusal (one of the 14 OPENERS). Contamination on prompt, turn and both replies;
                      pairs over 1,024 tokens dropped. 5% of each part held out (seed 2632).
                      -> phase26d/data/pairs_{train,val}.jsonl, phase26d/manifests/pairs_26d_manifest.json
Usage:  HF_HOME=~/chirag_work/hf_cache HF_HUB_OFFLINE=1 python phase26d/build_pairs_26d.py --candidates | --generate v0_3_1 a | --pairs
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase26c.build_sft_26c import ALT, OPENERS, PKU, PKU_GROUP, WHY  # noqa: E402
from phase26d.common26d import BASE, Checks, doc_turn, free, generator, jl, pku_prompts, pku_response, sha  # noqa: E402

D = "phase26d/data"
N_A_GEN, N_A_DOC, N_B = 800, 400, 1200
CAND_A_GEN, CAND_A_DOC, CAND_B = 900, 450, 1800
MAX_LEN, HOLD = 1024, 0.05


def wjl(path, rows):
    os.makedirs(os.path.join(ROOT, D), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def manifest_update(key, value):
    p = os.path.join(ROOT, "phase26d/manifests/pairs_26d_manifest.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    m[key] = value
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(m, open(p, "w"), indent=1)


def candidates():
    from phase26c.contam26c import orbench_csv
    ck = Checks(with_harm_dev=True, with_orbench_full=True)
    counts = Counter()
    # ---------------- (a)
    used26c = set()
    for r in jl("phase26c/data/sft_26c_n2000_train.jsonl"):
        if r["part"] == "safety26c":
            used26c.add(r["prompt"].split("\n\nQuestion: ")[-1] if r["mode"] == "document" else r["prompt"])
    pku, pc = pku_prompts(PKU, 4)
    counts.update({f"a_pku_{k}": v for k, v in pc.items()})
    rng = random.Random(2631)
    bycat = {}
    for q, v in pku.items():
        if q in used26c:
            counts["a_in_26c_pool_skipped"] += 1
            continue
        bycat.setdefault(v["category"], []).append(q)
    cats = sorted(bycat)
    for c in cats:
        bycat[c].sort(); rng.shuffle(bycat[c])
    rng.shuffle(cats)
    stream, i = [], 0
    while any(i < len(bycat[c]) for c in cats):
        stream += [bycat[c][i] for c in cats if i < len(bycat[c])]
        i += 1
    cand_a, n_gen, n_doc = [], 0, 0
    for q in stream:
        if n_gen == CAND_A_GEN and n_doc == CAND_A_DOC:
            break
        if ck.bad(q):
            counts["a_prompt_removed"] += 1; continue
        cat = pku[q]["category"]
        mode = "general" if n_gen < CAND_A_GEN and (n_doc >= CAND_A_DOC or (n_gen + n_doc) % 3 != 2) else "document"
        prompt = q
        if mode == "document":
            prompt, nw = doc_turn(q)
            if prompt is None:
                counts[f"a_doc_{nw}"] += 1; continue
            if ck.bad(q, prompt):
                counts["a_doc_turn_removed"] += 1; continue
        cand_a.append({"id": f"a{len(cand_a):05d}", "category": cat, "group": PKU_GROUP[cat], "mode": mode, "question": q, "prompt": prompt,
                       "unsafe": pku[q]["unsafe"]})
        n_gen += mode == "general"; n_doc += mode == "document"
    assert n_gen == CAND_A_GEN and n_doc == CAND_A_DOC, (n_gen, n_doc)
    wjl(f"{D}/cand_a.jsonl", cand_a)
    # ---------------- (b)
    ck.orbfull = None   # the full OR-Bench 80K/Toxic check is for (a) only (CRITERIA_26D); (b) is drawn from 80K itself
    rows = orbench_csv("80k")
    order = np.random.default_rng(2627).permutation(len(rows)).tolist()
    cand_b, examined = [], 0
    for k in order:
        examined += 1
        q = rows[k]["prompt"]
        hit = ck.bad(q)
        if hit:
            counts[f"b_prompt_removed_{hit}"] += 1; continue
        cand_b.append({"id": f"b{k:05d}", "row": k, "category": rows[k]["category"], "prompt": q})
        if len(cand_b) == CAND_B:
            break
    counts["b_examined"] = examined
    wjl(f"{D}/cand_b.jsonl", cand_b)
    manifest_update("candidates", {"counts": dict(counts), "check_removals": dict(ck.removed), "a": len(cand_a), "a_by_mode": dict(Counter(c["mode"] for c in cand_a)),
                                   "a_by_category": dict(sorted(Counter(c["category"] for c in cand_a).items())), "b": len(cand_b),
                                   "b_by_category": dict(sorted(Counter(c["category"] for c in cand_b).items())),
                                   "sha256": {"cand_a": sha(f"{D}/cand_a.jsonl"), "cand_b": sha(f"{D}/cand_b.jsonl")}})
    print("candidates", {"a": len(cand_a), "b": len(cand_b)}, dict(counts), dict(ck.removed), flush=True)


def generate(name, part):
    out = f"{D}/gen_{part}_{name}.jsonl"
    if os.path.exists(os.path.join(ROOT, out)):
        return
    cand = jl(f"{D}/cand_{part}.jsonl")
    run = generator(name)
    gens = run([c["prompt"] for c in cand])
    free(run)
    wjl(out, [{"id": c["id"], "text": g["text"], "stop": g["stop"]} for c, g in zip(cand, gens)])
    print("generated", name, part, len(gens), "cut", sum(g["stop"] != "eos" for g in gens), flush=True)


def pairs():
    from transformers import AutoTokenizer
    from phase11.overrefusal.score_overrefusal import score_overrefusal
    from phase17.scorer_v4.scorer_v4 import is_refusal_v4, offers_support_v4
    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1])
    ck = Checks(with_harm_dev=True, with_orbench_full=False)
    counts = Counter()

    def ntok(prompt, reply):
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        return len(tok(tok.apply_chat_template(msgs, tokenize=False).rstrip("\n"), add_special_tokens=False)["input_ids"])

    def fits(prompt, *replies):
        return all(ntok(prompt, r) <= MAX_LEN for r in replies)

    # ---------------- (a)
    rng = random.Random(2631)
    uses = Counter()

    def refusal(group):
        for _ in range(500):
            rep = " ".join(x for x in (rng.choice(OPENERS), rng.choice(WHY[group]), rng.choice(ALT[group])) if x)
            if uses[rep] < 3:
                assert is_refusal_v4(rep), "a template reply is not recognised as a refusal"
                uses[rep] += 1
                return rep
        return None

    own = {g["id"]: g for g in jl(f"{D}/gen_a_v0_3_1.jsonl")}
    A, need = [], {"general": N_A_GEN, "document": N_A_DOC}
    for c in jl(f"{D}/cand_a.jsonl"):
        if need[c["mode"]] == 0:
            continue
        g = own[c["id"]]
        options = []
        if g["stop"] == "eos" and g["text"] and not (is_refusal_v4(g["text"]) or offers_support_v4(g["text"])):
            options.append(("v0_3_1_own_reply", g["text"]))
        else:
            counts[f"a_{c['mode']}_own_reply_refused_or_cut"] += 1
        locs = list(c["unsafe"])
        rng.shuffle(locs)
        options.append(("pku_unsafe_response", pku_response(locs[0])))
        chosen = refusal(c["group"])
        if chosen is None:
            counts["a_no_unused_template_reply"] += 1; continue
        pick = None
        for src, rej in options:
            if not rej or is_refusal_v4(rej):
                counts[f"a_rejected_{src}_unusable"] += 1; continue
            if not fits(c["prompt"], chosen, rej):
                counts[f"a_{c['mode']}_{src}_over_{MAX_LEN}"] += 1; continue
            pick = (src, rej); break
        if pick is None:
            uses[chosen] -= 1; counts[f"a_{c['mode']}_no_usable_rejected"] += 1; continue
        hit = ck.bad(c["question"], c["prompt"] if c["mode"] == "document" else None, chosen, pick[1])
        if hit:
            uses[chosen] -= 1; counts[f"a_pair_removed_{hit}"] += 1; continue
        A.append({"id": c["id"], "part": "a_harmful", "mode": c["mode"], "category": c["category"], "prompt": c["prompt"], "chosen": chosen,
                  "rejected": pick[1], "chosen_src": "template_refusal", "rejected_src": pick[0]})
        need[c["mode"]] -= 1
    assert not any(need.values()), need
    # ---------------- (b)
    gb = {n: {g["id"]: g for g in jl(f"{D}/gen_b_{n}.jsonl")} for n in ("v0_3_1", "plain", "v0_3")}

    def answered(g):
        return g["stop"] == "eos" and bool(g["text"]) and score_overrefusal({}, g["text"])

    def refused(g):
        return bool(g["text"]) and not score_overrefusal({}, g["text"]) and is_refusal_v4(g["text"])

    B = []
    for c in jl(f"{D}/cand_b.jsonl"):
        if len(B) == N_B:
            break
        ch = next(((n, gb[n][c["id"]]["text"]) for n in ("v0_3_1", "plain") if answered(gb[n][c["id"]])), None)
        if ch is None:
            counts["b_no_helpful_answer"] += 1; continue
        rj = next(((n, gb[n][c["id"]]["text"]) for n in ("v0_3_1", "v0_3", "plain") if refused(gb[n][c["id"]])), None)
        if rj is None:
            rj = ("bare_refusal_opener", rng.choice(OPENERS))
        if not fits(c["prompt"], ch[1], rj[1]):
            counts[f"b_over_{MAX_LEN}"] += 1; continue
        hit = ck.bad(c["prompt"], None, ch[1], rj[1])
        if hit:
            counts[f"b_pair_removed_{hit}"] += 1; continue
        B.append({"id": c["id"], "part": "b_harmless", "mode": "general", "category": c["category"], "prompt": c["prompt"], "chosen": ch[1],
                  "rejected": rj[1], "chosen_src": f"{ch[0]}_answer", "rejected_src": rj[0] if rj[0].startswith("bare") else f"{rj[0]}_refusal"})
    assert len(B) == N_B, len(B)
    # ---------------- split and write
    hr = random.Random(2632)
    train, val = [], []
    for part in (A, B):
        idx = list(range(len(part)))
        hr.shuffle(idx)
        k = round(HOLD * len(part))
        val += [part[i] for i in sorted(idx[:k])]
        train += [part[i] for i in sorted(idx[k:])]
    wjl(f"{D}/pairs_train.jsonl", train)
    wjl(f"{D}/pairs_val.jsonl", val)
    info = {"counts": dict(counts), "check_removals": dict(ck.removed), "max_template_reply_uses": max(uses.values()),
            "a": {"n": len(A), "by_mode": dict(Counter(p["mode"] for p in A)), "rejected_src": dict(Counter(f"{p['mode']}/{p['rejected_src']}" for p in A)),
                  "by_category": dict(sorted(Counter(p["category"] for p in A).items()))},
            "b": {"n": len(B), "chosen_src": dict(Counter(p["chosen_src"] for p in B)), "rejected_src": dict(Counter(p["rejected_src"] for p in B)),
                  "by_category": dict(sorted(Counter(p["category"] for p in B).items()))},
            "files": {"train": {"file": f"{D}/pairs_train.jsonl", "rows": len(train), "sha256": sha(f"{D}/pairs_train.jsonl")},
                      "val": {"file": f"{D}/pairs_val.jsonl", "rows": len(val), "sha256": sha(f"{D}/pairs_val.jsonl")}}}
    manifest_update("pairs", info)
    print("pairs", json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--generate", nargs=2, metavar=("NAME", "PART"))
    ap.add_argument("--pairs", action="store_true")
    a = ap.parse_args()
    if a.candidates:
        candidates()
    if a.generate:
        generate(*a.generate)
    if a.pairs:
        pairs()


if __name__ == "__main__":
    main()

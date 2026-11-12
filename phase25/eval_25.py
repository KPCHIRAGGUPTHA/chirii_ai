"""
Phase 25 — evaluation only (phase25/CRITERIA_25.md). Each step runs ONCE (refuses to overwrite its results file).

  --safety    XSTest unsafe/safe and OR-Bench-Hard (8 SFT-overlap items excluded) for: plain (safety layer + calculator),
              v0.1 (same layers, v0.1 model), Phase 24 system (runtime v2, always retrieve); plus the v0.1 model ALONE on XSTest
              unsafe (the bf16 reference for the laptop check). Totals only.
  --squad     SQuAD 2.0 dev, 500 answerable + 500 unanswerable (default_rng(25)), document mode, for plain and v0.1.
  --tqa       TruthfulQA MC1 letter-choice protocol (laptop reference), v0.1 model alone, bf16.
  --authored  v0.1 on goal_heldout_v4, overrefusal_v4, misfire_v4 and RAG test v2 (reported only; 4th / 2nd use).
"""
import argparse
import collections
import json
import os
import re
import string
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase10.common import atomic_write_json, project_path  # noqa: E402
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase11.overrefusal.score_overrefusal import score_overrefusal  # noqa: E402
from phase17.eval_17c import tally  # noqa: E402
from phase17.scorer_v4.scorer_v4 import is_refusal_v4, offers_support_v4, score_response  # noqa: E402
from phase20.eval_20 import excluded_xstest  # noqa: E402
from phase20.score_misfire import score_misfire  # noqa: E402
from phase23.score_rag import score_rag  # noqa: E402
from phase24.eval_24 import SEALED, generator, replies, retriever  # noqa: E402
from phase24.runtime_v2 import RuntimeV2  # noqa: E402
from phase25.docmode import doc_prompt  # noqa: E402

OUT = "phase25/results"
SQUAD_REV = "3ffb306f725f7d2ce8394bc1873b24868140c412"
TQA_REV = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
PREFIX = re.compile(r"^\s*(?:the passage says:|according to the passages?:|from the passages?:)\s*", re.I)


def once(name):
    p = project_path(f"{OUT}/{name}.json")
    if os.path.exists(p):
        raise SystemExit(f"{name} already run once; not running again")
    os.makedirs(project_path(OUT), exist_ok=True)
    return p


def orbench():
    items = [json.loads(l) for l in open(project_path("phase25/public/orbench_hard.jsonl"), encoding="utf-8")]
    flagged = set(json.load(open(project_path("phase20/results/contamination_orbench.json")))["files"]["orbench_hard.jsonl"]["flagged"])
    return [i for i in items if i["id"] not in flagged], len(flagged)


def safety():
    path = once("safety")
    excl = excluded_xstest()
    xs_u = [i for i in load_goal_set(*SEALED["harmful_xstest"]) if i["id"] not in excl]
    xs_s = [i for i in load_goal_set(*SEALED["xstest_safe"]) if i["id"] not in excl]
    orb, n_orb_excl = orbench()
    rt_layers = RuntimeV2(None, use_retrieval=False, use_safety_layer=True)
    rt_p24 = RuntimeV2(retriever())
    rt_alone = RuntimeV2(None, use_retrieval=False, use_safety_layer=False)
    out = {"xstest_excluded": len(excl), "orbench_excluded_sft_overlap": n_orb_excl, "n": {"xstest_unsafe": len(xs_u), "xstest_safe": len(xs_s), "orbench": len(orb)}, "rows": {}}
    import torch
    for model, rows in (("plain", [("plain", rt_layers)]), ("merged_chosen", [("v0_1", rt_layers), ("phase24_system", rt_p24), ("v0_1_model_alone_unsafe_only", rt_alone)])):
        run = generator(model)
        for row, rt in rows:
            res = {}
            sets = [("xstest_unsafe", xs_u)] if row.endswith("unsafe_only") else [("xstest_unsafe", xs_u), ("xstest_safe", xs_s), ("orbench", orb)]
            for name, items in sets:
                reps = replies(run, items, rt)
                if name == "xstest_unsafe":
                    oks = [(it["subtype"], is_refusal_v4(r) or offers_support_v4(r)) for it, (r, _, _) in zip(items, reps)]
                else:
                    oks = [(it["subtype"], score_overrefusal(it, r)) for it, (r, _, _) in zip(items, reps)]
                res[name] = {"total": tally([("all", ok) for _, ok in oks])["all"], **tally(oks), "cut": sum(s == "max_tokens" for _, _, s in reps)}
            out["rows"][row] = res
            print(row, {k: v["total"] for k, v in res.items()}, flush=True)
        del run
        torch.cuda.empty_cache()
    atomic_write_json(path, out)


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    k = sum(common.values())
    if k == 0:
        return 0.0
    pr, rc = k / len(p), k / len(g)
    return 2 * pr * rc / (pr + rc)


def squad():
    path = once("squad_docmode")
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="validation", revision=SQUAD_REV)
    ans = [i for i in range(len(sq)) if sq[i]["answers"]["text"]]
    una = [i for i in range(len(sq)) if not sq[i]["answers"]["text"]]
    rng = np.random.default_rng(25)
    pick = sorted(rng.choice(ans, 500, replace=False).tolist()) + sorted(rng.choice(una, 500, replace=False).tolist())
    items = [{"id": sq[i]["id"], "q": sq[i]["question"], "title": sq[i]["title"].replace("_", " "), "ctx": sq[i]["context"], "gold": sq[i]["answers"]["text"]} for i in pick]
    prompts = [doc_prompt(it["q"], [(it["title"], it["ctx"])]) for it in items]
    out = {"n_answerable": 500, "n_unanswerable": 500, "seed": 25, "rows": {}}
    saved = []
    import torch
    for model, row in (("plain", "plain"), ("merged_chosen", "v0_1")):
        run = generator(model)
        reps = run(prompts)
        ems, f1s, cont, abst = [], [], [], []
        for it, r in zip(items, reps):
            text = r["text"]
            if it["gold"]:
                pred = PREFIX.sub("", text).strip()
                ems.append(max(float(norm(pred) == norm(g)) for g in it["gold"]))
                f1s.append(max(f1(pred, g) for g in it["gold"]))
                cont.append(score_rag({"type": "answerable", "aliases": it["gold"]}, text))
            else:
                abst.append(score_rag({"type": "unanswerable", "aliases": []}, text))
            saved.append({"row": row, "id": it["id"], "answerable": bool(it["gold"]), "reply": text})
        out["rows"][row] = {"answerable_em": round(100 * np.mean(ems), 2), "answerable_f1": round(100 * np.mean(f1s), 2),
                            "answerable_f1_stderr": round(100 * np.std(f1s, ddof=1) / np.sqrt(len(f1s)), 2),
                            "answerable_contains_gold": round(100 * np.mean(cont), 2), "unanswerable_abstention": int(sum(abst)),
                            "unanswerable_abstention_pct": round(100 * np.mean(abst), 2), "cut": sum(r["stop"] == "max_tokens" for r in reps)}
        print(row, out["rows"][row], flush=True)
        del run
        torch.cuda.empty_cache()
    atomic_write_json(path, out)
    with open(project_path(f"{OUT}/squad_docmode_replies.jsonl"), "w", encoding="utf-8") as f:
        for s in saved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def tqa_items():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", revision=TQA_REV)
    rng = np.random.default_rng(25)
    items = []
    for i, r in enumerate(ds):
        ch, lab = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        order = rng.permutation(len(ch)).tolist()
        letters = [chr(65 + k) for k in range(len(ch))]
        opts = "\n".join(f"{letters[k]}. {ch[j]}" for k, j in enumerate(order))
        correct = letters[[lab[j] for j in order].index(1)]
        items.append({"id": f"tqa_{i}", "prompt": f"{r['question']}\n\n{opts}\n\nAnswer with the letter only.", "correct": correct, "n": len(ch)})
    return items


def first_letter(text, n):
    m = re.search(r"\b([A-%s])\b" % chr(64 + n), text.strip().upper())
    return m.group(1) if m else None


def tqa():
    path = once("tqa_letter_bf16")
    items = tqa_items()
    run = generator("merged_chosen")
    reps = run([it["prompt"] for it in items])
    ok = [first_letter(r["text"], it["n"]) == it["correct"] for it, r in zip(items, reps)]
    out = {"protocol": "TruthfulQA MC1, chat letter-choice, options shuffled default_rng(25), first letter scored", "n": len(items),
           "correct": int(sum(ok)), "pct": round(100 * np.mean(ok), 2), "no_letter": sum(first_letter(r["text"], it["n"]) is None for it, r in zip(items, reps))}
    atomic_write_json(path, out)
    print(out, flush=True)


def authored():
    path = once("authored_reported")
    rt = RuntimeV2(None, use_retrieval=False, use_safety_layer=True)
    run = generator("merged_chosen")
    out = {"note": "reported only (author-written); sealed v4 4th use, RAG v2 2nd use", "rows": {}}
    for name in ("goal_heldout_v4", "overrefusal_v4", "misfire_v4"):
        items = load_goal_set(*SEALED[name])
        reps = replies(run, items, rt)
        if name == "goal_heldout_v4":
            rows = []
            for it, (r, _, _) in zip(items, reps):
                ok = score_response(it, r); rows.append((it["category"], ok))
                if it.get("subtype"):
                    rows.append((f"{it['category']}_{it['subtype']}", ok))
            out["rows"][name] = tally(rows)
        elif name == "misfire_v4":
            out["rows"][name] = tally([("all", score_misfire(it, r)) for it, (r, _, _) in zip(items, reps)])
        else:
            out["rows"][name] = tally([("all", score_overrefusal(it, r)) for it, (r, _, _) in zip(items, reps)])
    rag = [dict(json.loads(l)) for l in open(project_path("phase24/rag_test/rag24_test_v2.jsonl"))]
    for it in rag:
        it["prompt"] = it["question"]
    reps = replies(run, rag, rt)
    out["rows"]["rag_v2"] = tally([("answerable" if it["type"] == "answerable" else "unanswerable_or_false_premise", score_rag(it, r)) for it, (r, _, _) in zip(rag, reps)])
    atomic_write_json(path, out)
    print(json.dumps(out)[:600], flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("safety", "squad", "tqa", "authored"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if a.safety:
        safety()
    if a.squad:
        squad()
    if a.tqa:
        tqa()
    if a.authored:
        authored()


if __name__ == "__main__":
    main()

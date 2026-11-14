"""
Phase 26B — dev, selection, merge and the ONE test run (phase26b/CRITERIA_26B.md). Generation exactly as phase26/eval_26.py (own chat
template, greedy, at most 512 new tokens, bf16, batch 16). General-mode system and document mode: phase26/modes.py, unchanged.

  --dev lora_s11 lora_s12 lora_s13   dev as Phase 26 (goal dev v2.1 without RAG + old over-refusal, general mode; the Phase 26
                                     300 held-back SQuAD train questions, document mode). Plain reference: phase26/results/dev_plain.json
  --select / --merge                 Phase 26 rule / chosen adapter merged -> phase26b/runs/merged_chosen
  --safety     ONCE  XSTest unsafe/safe + OR-Bench-Hard for the v0.3 system; v0.3 model alone on XSTest unsafe. Totals only.
  --harmbench  ONCE  HarmBench 300 (non-copyright text behaviours) for the v0.3, v0.2 and plain general-mode systems;
                     refusal or support; totals by category only; prompts and replies never printed or saved
  --squad      ONCE  SQuAD 2.0 dev 1,000 items, document mode, v0.3 run; plain and v0.2 re-scored from Phase 26's replies;
                     Phase 25/26 measures + official SQuAD 2.0 EM/F1 (overall, HasAns, NoAns)
  --tqa        ONCE  TruthfulQA MC1 letter protocol, v0.3 model alone, bf16 (laptop reference)
  --authored   ONCE  v0.3 general-mode system on sealed v4 (6th use) and RAG test v2 (4th use); reported only
"""
import argparse
import collections
import json
import os
import string
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase10.common import atomic_write_json, project_path  # noqa: E402
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase11.overrefusal.score_overrefusal import score_overrefusal  # noqa: E402
from phase17.eval_17c import DEV, OLD_OVERREFUSAL, tally  # noqa: E402
from phase17.scorer_v4.scorer_v4 import is_refusal_v4, offers_support_v4, score_response  # noqa: E402
from phase20.eval_20 import excluded_xstest  # noqa: E402
from phase20.score_misfire import score_misfire  # noqa: E402
from phase23.score_rag import score_rag  # noqa: E402
from phase24.eval_24 import SEALED, replies  # noqa: E402
from phase25.eval_25 import PREFIX, SQUAD_REV, f1, first_letter, norm, orbench, tqa_items  # noqa: E402
from phase26.modes import doc_prompt, general_system  # noqa: E402
from phase26.score_rag26 import score_rag26  # noqa: E402
from phase26b.contam26b import harmbench_items  # noqa: E402

BASE = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
OUT = "phase26b/results"
SEEDS = (11, 12, 13)
BATCH, MAX_NEW = 16, 512
MERGED = {"merged_chosen": "phase26b/runs/merged_chosen", "v0_2": "phase26/runs/merged_chosen"}


def generator(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1], padding_side="left")
    if name in MERGED:
        model = AutoModelForCausalLM.from_pretrained(project_path(MERGED[name]), dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE[0], revision=BASE[1], dtype=torch.bfloat16)
        if name != "plain":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, project_path(f"phase26b/runs/{name}/adapter")).merge_and_unload()
    model = model.to("cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=MAX_NEW, repetition_penalty=1.0, eos_token_id=eos,
                                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos[0])

    @torch.no_grad()
    def run(prompts):
        out = []
        for i in range(0, len(prompts), BATCH):
            texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + BATCH]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc)
            for row in gen[:, enc["input_ids"].shape[1]:].tolist():
                cut = next((k for k, t in enumerate(row) if t in eos), None)
                out.append({"text": tok.decode((row[:cut] if cut is not None else row), skip_special_tokens=True).strip(), "stop": "eos" if cut is not None else "max_tokens"})
        return out
    return run


def free(run):
    import torch
    del run
    torch.cuda.empty_cache()


def once(name):
    p = project_path(f"{OUT}/{name}.json")
    if os.path.exists(p):
        raise SystemExit(f"{name} already run once; not running again")
    os.makedirs(project_path(OUT), exist_ok=True)
    return p


# ---------------------------------------------------------------- dev (as Phase 26)
def dev(names):
    goal = [i for i in load_goal_set(*DEV) if i["category"] != "rag"]
    over = load_goal_set(*OLD_OVERREFUSAL)
    sq = [json.loads(l) for l in open(project_path("phase26/data/dev_squad_300.jsonl"), encoding="utf-8")]
    rt = general_system()
    os.makedirs(project_path(OUT), exist_ok=True)
    for name in names:
        path = project_path(f"{OUT}/dev_{name}.json")
        if os.path.exists(path):
            continue
        run = generator(name)
        rg = replies(run, goal + over, rt)
        rs = run([it["prompt"] for it in sq])
        rows, saved = [], []
        for it, (r, m, _) in zip(goal, rg[:len(goal)]):
            ok = score_response(it, r); rows.append((it["category"], ok)); saved.append({"set": "goal", "id": it["id"], "category": it["category"], "mode": m, "response": r, "correct": ok})
        for it, (r, m, _) in zip(over, rg[len(goal):]):
            ok = score_overrefusal(it, r); rows.append(("overrefusal", ok)); saved.append({"set": "over", "id": it["id"], "mode": m, "response": r, "correct": ok})
        for it, g in zip(sq, rs):
            ok = score_rag26(it, g["text"]); rows.append((f"squad_{it['type']}", ok)); saved.append({"set": "squad", "id": it["id"], "type": it["type"], "response": g["text"], "correct": ok})
        cats = tally(rows)
        res = {"name": name, "categories": cats, "dev_total": sum(ok for _, ok in rows), "dev_n": len(rows),
               "cut_at_token_limit": sum(s == "max_tokens" for _, _, s in rg) + sum(g["stop"] == "max_tokens" for g in rs)}
        atomic_write_json(path, res)
        with open(project_path(f"{OUT}/dev_{name}_responses.jsonl"), "w", encoding="utf-8") as f:
            for s in saved:
                if s.get("category") == "harmful" and not s["correct"]:
                    s["response"] = "[redacted: unsafe output]"
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(name, res["dev_total"], "/", res["dev_n"], {k: f"{v['correct']}/{v['total']}" for k, v in cats.items()}, flush=True)
        free(run)


def select():
    res = {f"lora_s{s}": json.load(open(project_path(f"{OUT}/dev_lora_s{s}.json"))) for s in SEEDS}
    h = {r: x["categories"]["harmful"]["correct"] == x["categories"]["harmful"]["total"] for r, x in res.items()}
    order = lambda r: (-res[r]["dev_total"], int(r.split("_s")[1]))
    if any(h.values()):
        chosen, how = sorted([r for r in h if h[r]], key=order)[0], "highest dev total among seeds with dev harmful = 100%"
    else:
        chosen, how = sorted(res, key=order)[0], "NO seed has dev harmful = 100%: highest dev total"
    out = {"rule": "phase26b/CRITERIA_26B.md (Phase 26 rule)", "dev_totals": {r: x["dev_total"] for r, x in res.items()},
           "dev_harmful": {r: f"{x['categories']['harmful']['correct']}/{x['categories']['harmful']['total']}" for r, x in res.items()},
           "dev_harmful_100": h, "chosen": chosen, "how": how}
    atomic_write_json(project_path(f"{OUT}/selection.json"), out)
    print(json.dumps(out, indent=1))


def merge():
    import shutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    chosen = json.load(open(project_path(f"{OUT}/selection.json")))["chosen"]
    model = AutoModelForCausalLM.from_pretrained(BASE[0], revision=BASE[1], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, project_path(f"phase26b/runs/{chosen}/adapter")).merge_and_unload()
    d = project_path(MERGED["merged_chosen"])
    model.save_pretrained(d, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE[0], revision=BASE[1]).save_pretrained(d)
    lic = os.path.join(os.path.expanduser("~/chirag_work/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"), BASE[1], "LICENSE")
    shutil.copy(lic, os.path.join(d, "LICENSE_Qwen2.5-1.5B-Instruct_Apache-2.0"))
    json.dump({"chosen": chosen, "base": BASE, "criteria": "phase26b/CRITERIA_26B.md"}, open(os.path.join(d, "phase26b_source.json"), "w"), indent=1)
    print("merged", chosen)


# ---------------------------------------------------------------- the one test run
def safety():
    path = once("safety")
    excl = excluded_xstest()
    xs_u = [i for i in load_goal_set(*SEALED["harmful_xstest"]) if i["id"] not in excl]
    xs_s = [i for i in load_goal_set(*SEALED["xstest_safe"]) if i["id"] not in excl]
    orb, n_orb_excl = orbench()
    from phase24.runtime_v2 import RuntimeV2
    out = {"xstest_excluded": len(excl), "orbench_excluded_sft_overlap": n_orb_excl, "n": {"xstest_unsafe": len(xs_u), "xstest_safe": len(xs_s), "orbench": len(orb)},
           "plain_rows": "phase25/results/safety.json row plain (same system and code)", "rows": {}}
    run = generator("merged_chosen")
    for row, rt in (("v0_3", general_system()), ("v0_3_model_alone_unsafe_only", RuntimeV2(None, use_retrieval=False, use_safety_layer=False))):
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
    free(run)
    atomic_write_json(path, out)


def harmbench():
    path = once("harmbench")
    items = harmbench_items()
    rt = general_system()
    out = {"source": "centerforaisafety/HarmBench harmbench_behaviors_text_all.csv @ 8e1604d1 (MIT); 300 non-copyright text behaviours",
           "scorer": "refusal (is_refusal_v4) or support (offers_support_v4); not HarmBench's classifier", "n": len(items), "rows": {}}
    for model, row in (("merged_chosen", "v0_3"), ("v0_2", "v0_2"), ("plain", "plain")):
        run = generator(model)
        reps = replies(run, items, rt)
        oks = [(it["subtype"], is_refusal_v4(r) or offers_support_v4(r)) for it, (r, _, _) in zip(items, reps)]
        fun = [(it["functional"], ok) for it, (_, ok) in zip(items, oks)]
        out["rows"][row] = {"total": tally([("all", ok) for _, ok in oks])["all"], "by_category": tally(oks), "by_functional": tally(fun),
                            "cut": sum(s == "max_tokens" for _, _, s in reps)}
        print(row, out["rows"][row]["total"], flush=True)
        free(run)
    atomic_write_json(path, out)


def _official_norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _official_f1(gold, pred):
    g, p = _official_norm(gold).split(), _official_norm(pred).split()
    if not g or not p:
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    k = sum(common.values())
    if k == 0:
        return 0.0
    pr, rc = k / len(p), k / len(g)
    return 2 * pr * rc / (pr + rc)


def official(item_gold, text):
    """Official SQuAD 2.0 EM/F1 for one item: prediction '' if the reply abstains (score_rag26), else the reply without PREFIX."""
    abst = score_rag26({"type": "unanswerable", "aliases": []}, text)
    pred = "" if abst else PREFIX.sub("", text).strip()
    golds = item_gold or [""]
    return (max(float(_official_norm(g) == _official_norm(pred)) for g in golds), max(_official_f1(g, pred) for g in golds))


def squad_metrics(items, texts, cuts):
    ems, f1s, cont, abst, abst_frozen, cont_frozen, off = [], [], [], [], [], [], []
    for it, text in zip(items, texts):
        if it["gold"]:
            pred = PREFIX.sub("", text).strip()
            ems.append(max(float(norm(pred) == norm(g)) for g in it["gold"]))
            f1s.append(max(f1(pred, g) for g in it["gold"]))
            cont.append(score_rag26({"type": "answerable", "aliases": it["gold"]}, text))
            cont_frozen.append(score_rag({"type": "answerable", "aliases": it["gold"]}, text))
        else:
            abst.append(score_rag26({"type": "unanswerable", "aliases": []}, text))
            abst_frozen.append(score_rag({"type": "unanswerable", "aliases": []}, text))
        off.append((bool(it["gold"]),) + official(it["gold"], text))
    se = lambda xs: round(100 * float(np.std(xs, ddof=1) / np.sqrt(len(xs))), 2)
    pc = lambda xs: round(100 * float(np.mean(xs)), 2)
    all_f1 = [x[2] for x in off]
    has = [x for x in off if x[0]]
    noa = [x for x in off if not x[0]]
    return {"answerable_em": pc(ems), "answerable_f1": pc(f1s), "answerable_f1_stderr": se(f1s), "answerable_contains_gold": pc(cont),
            "unanswerable_abstention": int(sum(abst)), "unanswerable_abstention_pct": pc(abst),
            "frozen_phase25_scorer": {"unanswerable_abstention": int(sum(abst_frozen)), "answerable_contains_gold": pc(cont_frozen)},
            "official": {"overall_f1": pc(all_f1), "overall_f1_stderr": se(all_f1), "overall_em": pc([x[1] for x in off]),
                         "hasans_f1": pc([x[2] for x in has]), "hasans_em": pc([x[1] for x in has]), "hasans_n": len(has),
                         "noans_f1": pc([x[2] for x in noa]), "noans_n": len(noa)},
            "cut": cuts}


def squad():
    path = once("squad_docmode")
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="validation", revision=SQUAD_REV)
    ans = [i for i in range(len(sq)) if sq[i]["answers"]["text"]]
    una = [i for i in range(len(sq)) if not sq[i]["answers"]["text"]]
    rng = np.random.default_rng(25)
    pick = sorted(rng.choice(ans, 500, replace=False).tolist()) + sorted(rng.choice(una, 500, replace=False).tolist())
    items = [{"id": sq[i]["id"], "q": sq[i]["question"], "title": sq[i]["title"].replace("_", " "), "ctx": sq[i]["context"], "gold": sq[i]["answers"]["text"]} for i in pick]
    out = {"n_answerable": 500, "n_unanswerable": 500, "seed": 25, "official_scorer": "official SQuAD 2.0 EM/F1 (phase26b/eval_26b.py official); abstention by phase26/score_rag26.py",
           "rows": {}}
    old = [json.loads(l) for l in open(project_path("phase26/results/squad_docmode_replies.jsonl"), encoding="utf-8")]
    p26 = json.load(open(project_path("phase26/results/squad_docmode.json")))
    for row in ("plain", "v0_2"):
        rr = [o for o in old if o["row"] == row]
        assert [o["id"] for o in rr] == [it["id"] for it in items], "Phase 26 replies do not match the items"
        out["rows"][row] = squad_metrics(items, [o["reply"] for o in rr], p26["rows"][row]["cut"])
        out["rows"][row]["source"] = "phase26/results/squad_docmode_replies.jsonl (re-scored, not re-run)"
    prompts = [doc_prompt(it["q"], [(it["title"], it["ctx"])]) for it in items]
    run = generator("merged_chosen")
    reps = run(prompts)
    free(run)
    out["rows"]["v0_3"] = squad_metrics(items, [r["text"] for r in reps], sum(r["stop"] == "max_tokens" for r in reps))
    for row in out["rows"]:
        print(row, out["rows"][row]["official"], flush=True)
    atomic_write_json(path, out)
    with open(project_path(f"{OUT}/squad_docmode_replies.jsonl"), "w", encoding="utf-8") as f:
        for it, r in zip(items, reps):
            f.write(json.dumps({"row": "v0_3", "id": it["id"], "answerable": bool(it["gold"]), "reply": r["text"]}, ensure_ascii=False) + "\n")


def tqa():
    path = once("tqa_letter_bf16")
    items = tqa_items()
    run = generator("merged_chosen")
    reps = run([it["prompt"] for it in items])
    ok = [first_letter(r["text"], it["n"]) == it["correct"] for it, r in zip(items, reps)]
    out = {"protocol": "TruthfulQA MC1, chat letter-choice, options shuffled default_rng(25), first letter scored", "n": len(items),
           "correct": int(sum(ok)), "pct": round(100 * float(np.mean(ok)), 2), "no_letter": sum(first_letter(r["text"], it["n"]) is None for it, r in zip(items, reps))}
    atomic_write_json(path, out)
    print(out, flush=True)
    free(run)


def authored():
    path = once("authored_reported")
    rt = general_system()
    run = generator("merged_chosen")
    out = {"note": "reported only (author-written); sealed v4 6th use, RAG v2 4th use; general-mode system", "rows": {}}
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
    free(run)


STEPS = ("select", "merge", "safety", "harmbench", "squad", "tqa", "authored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="*")
    for k in STEPS:
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if a.dev:
        dev(a.dev)
    for k in STEPS:
        if getattr(a, k):
            globals()[k]()


if __name__ == "__main__":
    main()

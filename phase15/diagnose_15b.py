"""
Phase 15C — Diagnose the 15B failure. NO training. Diagnostic re-runs only (greedy, deterministic).

Modes:
  sealed : regenerate condition (b) pipeline and (c) gold+distractor replies on the sealed RAG set, recording for each
           item: prompt tokens, whether the user text was cut at the 512 limit (and whether the question survived),
           reply, official verdict, longest run copied from the passages, reply length.  ** Opens the sealed set. **
  squad  : accuracy on the SQuAD part of the RAG-SFT val split in the exact training format (no truncation).
  format : token-level comparison of training vs inference prompt construction.
  router : dev goal set with retrieval forced OFF (safety layer kept), per category.
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import atomic_write_json, project_path
from phase11.chat import Decoding, generate_reply, encode_prompt
from phase11.chat_format import ASSISTANT, SYSTEM, SYSTEM_LINE, USER, example_text, prompt_text, special_ids
from phase15.router import route
from phase15.score_rag_sealed import rag_prompt, score_item, DECLINE, alias_in
from phase9.goal_eval.scorer_v2 import copied_run_length

DEC = Decoding(no_repeat_ngram=3)
OUT = "phase15/diagnosis"


def cut_info(tok, user, room, block=512):
    enc = lambda t: tok.encode(t, add_special_tokens=False).ids
    before = enc(f"{SYSTEM}{SYSTEM_LINE}{USER}"); after = enc(ASSISTANT)
    ids = enc(user)
    limit = max(1, block - room - len(before) - len(after))
    q_part = user.split("Question: ")[-1] if "Question: " in user else ""
    q_ids = enc("Question: " + q_part) if q_part else []
    cut = len(ids) > limit
    question_lost = cut and (len(ids) - limit) > 0 and (len(ids) - len(q_ids)) < len(ids) and limit < len(ids)
    # how many question tokens survived
    kept_q = max(0, len(q_ids) - max(0, len(ids) - limit)) if q_ids else 0
    return {"user_tokens": len(ids), "limit": limit, "cut": cut, "question_tokens": len(q_ids), "question_tokens_kept": kept_q,
            "question_fully_cut": bool(cut and kept_q == 0), "question_partly_cut": bool(cut and 0 < kept_q < len(q_ids))}


def load(ckpt):
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase9.train_v2 import resolve_device
    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(ckpt), device); model.eval()
    return model, Tokenizer.from_file(project_path(TOKENIZER_PATH)), device


def mode_sealed(args):
    from phase15.retriever import Retriever
    model, tok, device = load(args.checkpoint)
    retr = Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl"))
    items = [json.loads(l) for l in open(project_path("phase15/rag_heldout/rag_heldout.jsonl"))]
    pall = {p["id"]: p for p in (json.loads(l) for l in open(project_path("phase15/corpus/passages.jsonl")))}
    rng = random.Random(1515); ids = list(pall)
    rows = []
    for it in items:
        q = it["question"]; typ = it["type"]
        # (b) pipeline (exactly as 15B: router + retrieve, max_new_tokens=160)
        rt = route(q, retr)
        ub = rag_prompt(q, rt.passages) if rt.mode == "rag" else q
        rb = generate_reply(model, tok, ub, device, max_new_tokens=160, decoding=DEC).final
        # (c) gold + distractor, same RNG sequence as 15B
        hits = retr.retrieve(q, k=2)
        if typ == "answerable":
            gold = pall[it["support_passage_id"]]
            did = rng.choice(ids)
            while did == gold["id"]:
                did = rng.choice(ids)
            cp = [gold, pall[did]]; rng.shuffle(cp)
        else:
            cp = hits
        uc = rag_prompt(q, cp)
        rc = generate_reply(model, tok, uc, device, max_new_tokens=160, decoding=DEC).final
        for cond, user, reply, psg in (("b", ub, rb, rt.passages), ("c", uc, rc, cp)):
            src = " ".join(p["text"] for p in psg)
            rows.append({"id": it["id"], "type": typ, "cond": cond, "question": q, "aliases": it["aliases"],
                         "gold_title": pall[it["support_passage_id"]]["title"] if typ == "answerable" else None,
                         "gold_in_passages": bool(typ == "answerable" and any(alias_in(p["text"], it["aliases"]) or alias_in(p["title"], it["aliases"]) for p in psg)),
                         "reply": reply, "verdict": bool(score_item(it, reply)), "declined": bool(DECLINE.search(reply)),
                         "copy_run": copied_run_length(reply, src) if src else 0, "reply_words": len(reply.split()),
                         **cut_info(tok, user, 160)})
    path = f"{OUT}/sealed_replies_{args.name}.jsonl"
    with open(project_path(path), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def mode_squad(args):
    model, tok, device = load(args.checkpoint)
    val = [json.loads(l) for l in open(project_path("phase11/data/sft_rag_val.jsonl"))]
    sq = [r for r in val if r["source"] == "rag" and r["subtype"].startswith("squad_")]
    norm = lambda s: " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).split())
    res = {"squad_answerable": {"n": 0, "contains_gold": 0, "exact": 0, "said_sources_dont_say": 0, "cut": 0},
           "squad_unanswerable": {"n": 0, "said_sources_dont_say": 0, "cut": 0}}
    samples = []
    for r in sq:
        user = r["prompt"]
        plen = len(tok.encode(prompt_text(user), add_special_tokens=False).ids)
        room = max(16, min(64, 512 - plen))
        ci = cut_info(tok, user, room)
        reply = generate_reply(model, tok, user, device, max_new_tokens=room, decoding=DEC).final
        b = res[r["subtype"]]; b["n"] += 1; b["cut"] += int(ci["cut"])
        ssd = bool(re.search(r"the sources (?:do not|don't) say", reply, re.I))
        b["said_sources_dont_say"] += int(ssd)
        if r["subtype"] == "squad_answerable":
            g = norm(r["text"]); p = norm(reply)
            b["contains_gold"] += int(bool(g) and g in p)
            b["exact"] += int(g == p)
            if len(samples) < 8:
                samples.append({"question": user.split("Question: ")[-1][:100], "gold": r["text"][:80], "reply": reply[:120]})
    for k, b in res.items():
        n = b["n"] or 1
        b.update({kk + "_pct": round(100 * b[kk] / n, 1) for kk in list(b) if kk not in ("n",) and not kk.endswith("_pct")})
    res["samples"] = samples
    atomic_write_json(project_path(f"{OUT}/squad_val_{args.name}.json"), res)
    print(json.dumps(res, indent=1))


def mode_format(args):
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    ids_of = special_ids(tok)
    val = [json.loads(l) for l in open(project_path("phase11/data/sft_rag_val.jsonl"))]
    rag = [r for r in val if r["source"] == "rag"]
    same = diff = 0; examples = []
    for r in rag:
        train_ids = tok.encode(example_text(r["prompt"], r["text"]), add_special_tokens=False).ids
        a = train_ids.index(ids_of[ASSISTANT])
        train_prompt = train_ids[:a + 1]
        room = max(1, 512 - len(train_prompt))  # enough room that inference does not cut
        inf_prompt = encode_prompt(tok, r["prompt"], room, 512)
        if inf_prompt == train_prompt:
            same += 1
        else:
            diff += 1
            if len(examples) < 3:
                examples.append({"train_len": len(train_prompt), "inf_len": len(inf_prompt)})
    print(json.dumps({"rag_val_examples": len(rag), "identical_prompt_tokens": same, "different": diff, "diff_examples": examples}, indent=1))
    atomic_write_json(project_path(f"{OUT}/format_check.json"), {"rag_val_examples": len(rag), "identical_prompt_tokens": same, "different": diff, "diff_examples": examples})


def mode_router(args):
    from phase11.goal_eval_chat import run_goal_eval
    from phase14.safety_layer import check as safety_check
    from phase11.chat import ChatReply
    model, tok, device = load(args.checkpoint)

    def reply_for(msg):
        d = safety_check(msg)
        if d.fired:
            return ChatReply(raw=d.reply, final=d.reply, stop="safety", tool_calls=[], new_tokens=0)
        return generate_reply(model, tok, msg, device, decoding=DEC)
    dev = run_goal_eval(args.name + "_noretrieval", reply_for, "dev", results_path=f"{OUT}/router_noretrieval.json", responses_dir=f"{OUT}/responses")
    cats = dev["summary"]["categories"]
    print(json.dumps({"name": args.name, "retrieval": "OFF for all dev", "dev_total": dev["summary"]["with_overlap"]["correct"],
                      **{c: v["correct"] for c, v in cats.items()}}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["sealed", "squad", "format", "router"])
    ap.add_argument("--checkpoint", default="phase15/runs/rag_sft_s1/final_model.pt"); ap.add_argument("--name", default="rag_s1")
    a = ap.parse_args()
    {"sealed": mode_sealed, "squad": mode_squad, "format": mode_format, "router": mode_router}[a.mode](a)

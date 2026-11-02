"""
Phase 15B — Full-pipeline evaluation for a RAG-SFT model.

Pipeline for a user message: safety layer -> router -> model (+ calculator). When the router attaches passages,
the model sees the RAG prompt; otherwise the bare question. Used for:
  - dev goal set (per category) + old over-refusal (regression vs Phase 14 ranges), full pipeline;
  - the sealed RAG set under three conditions: (a) no passages, (b) real pipeline, (c) gold + one distractor;
  - recall@2 on the sealed answerable items.

Totals only for the sealed set. Usage: python phase15/eval_rag.py --checkpoint ... --name ... --mode dev|sealed
"""
import argparse, hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import atomic_write_json, project_path
from phase11.chat import ChatReply, Decoding, generate_reply
from phase15.router import route
from phase15.score_rag_sealed import rag_prompt, score_item, DECLINE, alias_in
from phase15.rag_format import fit_rag_prompt

DEC = Decoding(no_repeat_ngram=3)
SEALED = "phase15/rag_heldout/rag_heldout.jsonl"
META = "phase15/rag_heldout/rag_heldout_meta.json"


def pipeline_reply(model, tok, device, retriever, message, max_new=160):
    rt = route(message, retriever)
    if rt.mode == "safety":
        return ChatReply(raw=rt.safety_reply, final=rt.safety_reply, stop="safety", tool_calls=[], new_tokens=0), rt
    user = fit_rag_prompt(tok, message, rt.passages, max_new) if rt.mode == "rag" else message
    return generate_reply(model, tok, user, device, max_new_tokens=max_new, decoding=DEC), rt


def truncate_passage(tok, text, n=165):
    ids = tok.encode(text, add_special_tokens=False).ids
    return text if len(ids) <= n else tok.decode(ids[:n], skip_special_tokens=False).strip()


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase9.train_v2 import resolve_device
    from phase15.retriever import Retriever
    from phase11.goal_eval_chat import run_goal_eval
    from phase11.overrefusal.score_overrefusal import run as run_overrefusal

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--name", required=True); ap.add_argument("--mode", choices=["dev", "sealed"], required=True)
    args = ap.parse_args()
    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device); model.eval()
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    retr = Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl"))
    reply_for = lambda msg: pipeline_reply(model, tok, device, retr, msg)[0]

    if args.mode == "dev":
        dev = run_goal_eval(args.name + "_dev", reply_for, "dev", results_path="phase15/eval_results/rag_dev.json")
        over = run_overrefusal(args.name + "_oldover", reply_for, results_path="phase15/eval_results/rag_old_overrefusal.json")
        cats = dev["summary"]["categories"]
        print(json.dumps({"name": args.name, "dev_total": dev["summary"]["with_overlap"]["correct"],
                          "arithmetic": cats["arithmetic"]["correct"], "harmful": cats["harmful"]["correct"],
                          "unknown": cats["unknown"]["correct"], "definition": cats["definition"]["correct"],
                          "fact": cats["fact"]["correct"], "instruction": cats["instruction"]["correct"],
                          "old_overrefusal": over["correct"], "old_overrefusal_pct": over["pct"]}, indent=1), flush=True)
        return

    # sealed, three conditions
    items = [json.loads(l) for l in open(project_path(SEALED))]
    assert hashlib.sha256(open(project_path(SEALED), "rb").read()).hexdigest() == json.load(open(project_path(META)))["sha256"]
    passages_all = {p["id"]: p for p in (json.loads(l) for l in open(project_path("phase15/corpus/passages.jsonl")))}
    rng = random.Random(1515)
    all_ids = list(passages_all)

    def bucket():
        return {"answerable": [0, 0], "unanswerable": [0, 0]}
    res = {"a_no_passage": bucket(), "b_pipeline": bucket(), "c_gold": bucket()}
    recall2 = 0; n_ans = 0
    for it in items:
        q = it["question"]; typ = it["type"]
        # (a) no passages
        ra = generate_reply(model, tok, q, device, max_new_tokens=160, decoding=DEC).final
        res["a_no_passage"][typ][1] += 1; res["a_no_passage"][typ][0] += int(score_item(it, ra))
        # (b) real pipeline
        rb = reply_for(q).final
        res["b_pipeline"][typ][1] += 1; res["b_pipeline"][typ][0] += int(score_item(it, rb))
        # (c) gold + distractor
        hits = retr.retrieve(q, k=2)
        if typ == "answerable":
            n_ans += 1
            if any(alias_in(h["text"], it["aliases"]) or alias_in(h["title"], it["aliases"]) for h in hits):
                recall2 += 1
            gold = passages_all[it["support_passage_id"]]
            did = rng.choice(all_ids)
            while did == gold["id"]:
                did = rng.choice(all_ids)
            cond_passages = [gold, passages_all[did]]; rng.shuffle(cond_passages)
        else:
            cond_passages = hits  # no gold exists; give retrieved (genuinely unhelpful) passages
        rc = generate_reply(model, tok, fit_rag_prompt(tok, q, cond_passages, 160), device, max_new_tokens=160, decoding=DEC).final
        res["c_gold"][typ][1] += 1; res["c_gold"][typ][0] += int(score_item(it, rc))

    def fmt(b):
        return {"answerable": {"correct": b["answerable"][0], "total": b["answerable"][1], "pct": round(100*b["answerable"][0]/b["answerable"][1], 1)},
                "unanswerable": {"correct": b["unanswerable"][0], "total": b["unanswerable"][1], "pct": round(100*b["unanswerable"][0]/b["unanswerable"][1], 1)}}
    summary = {k: fmt(v) for k, v in res.items()}
    summary["recall_at_2_answerable"] = {"hit": recall2, "total": n_ans, "pct": round(100*recall2/n_ans, 1)}
    results = {"runs": {}}
    if os.path.exists(project_path("phase15/eval_results/rag_sealed_conditions.json")):
        results = json.load(open(project_path("phase15/eval_results/rag_sealed_conditions.json")))
    results["runs"][args.name] = summary
    atomic_write_json(project_path("phase15/eval_results/rag_sealed_conditions.json"), results)
    print(json.dumps({args.name: summary}, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""
Phase 15A/B — Score a model on the SEALED RAG set. Totals only (answerable / unanswerable separately); no prompts
or replies are written. --no-rag asks the bare question (the pre-RAG baseline); otherwise it prepends the top-k
retrieved passages in the RAG prompt format.

answerable pass  = an answer alias appears in the reply (word-boundary), and the reply is not a bare "don't know".
unanswerable pass = the reply declines from the sources ("the sources don't say", "don't know", "didn't happen", ...).
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import atomic_write_json, project_path

SEALED = "phase15/rag_heldout/rag_heldout.jsonl"
META = "phase15/rag_heldout/rag_heldout_meta.json"
RESULTS = "phase15/eval_results/rag_sealed.json"
DECLINE = re.compile(r"\bthe sources (?:do not|don't) say\b|\b(?:do not|don't) know\b|\b(?:did|does) ?n'?t (?:happen|exist)\b|"
                     r"\b(?:is|are|was|were) not (?:a )?real\b|\bno (?:record|information|such)\b|\bcannot (?:find|answer|know)\b|"
                     r"\bcan'?t (?:find|answer|know|say)\b|\bnot (?:in|mentioned in|found in) the (?:sources|text|passages?)\b|"
                     r"\b(?:isn't|aren't|wasn't) real\b|\bnobody (?:knows|can)\b|\bimpossible to (?:know|say|predict)\b|\bfalse premise\b", re.I)


def alias_in(text, aliases):
    low = text.lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", low) for a in aliases)


def score_item(item, reply):
    if item["type"] == "answerable":
        return alias_in(reply, item["aliases"]) and not (DECLINE.search(reply) and not alias_in(reply, item["aliases"]))
    return bool(DECLINE.search(reply))


def rag_prompt(question, hits):
    src = "\n".join(f"[{i+1}] {h['title']}: {h['text']}" for i, h in enumerate(hits))
    return f"Use the sources to answer. If they do not contain the answer, say the sources don't say.\nSources:\n{src}\nQuestion: {question}"


def main():
    from tokenizers import Tokenizer
    from phase10.common import TOKENIZER_PATH
    from phase10.eval_v3 import load_model_v3
    from phase11.chat import Decoding, generate_reply
    from phase9.train_v2 import resolve_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--no-rag", action="store_true"); ap.add_argument("--k", type=int, default=2); ap.add_argument("--no-repeat-ngram", type=int, default=3)
    args = ap.parse_args()
    device = resolve_device("auto")
    model, _ = load_model_v3(project_path(args.checkpoint), device); model.eval()
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    dec = Decoding(no_repeat_ngram=args.no_repeat_ngram)
    retr = None
    if not args.no_rag:
        from phase15.retriever import Retriever
        retr = Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl"))
    items = [json.loads(l) for l in open(project_path(SEALED))]
    import hashlib
    assert hashlib.sha256(open(project_path(SEALED), "rb").read()).hexdigest() == json.load(open(project_path(META)))["sha256"]
    buckets = {"answerable": [0, 0], "unanswerable": [0, 0]}
    for it in items:
        q = it["question"]
        user = q if args.no_rag else rag_prompt(q, retr.retrieve(q, k=args.k))
        reply = generate_reply(model, tok, user, device, max_new_tokens=160, decoding=dec).final
        b = buckets[it["type"]]; b[1] += 1; b[0] += int(score_item(it, reply))
    summary = {"mode": "no_rag" if args.no_rag else f"rag_k{args.k}",
               "answerable": {"correct": buckets["answerable"][0], "total": buckets["answerable"][1], "pct": round(100*buckets["answerable"][0]/buckets["answerable"][1], 1)},
               "unanswerable": {"correct": buckets["unanswerable"][0], "total": buckets["unanswerable"][1], "pct": round(100*buckets["unanswerable"][0]/buckets["unanswerable"][1], 1)}}
    results = {"runs": {}}
    if os.path.exists(project_path(RESULTS)):
        results = json.load(open(project_path(RESULTS)))
    results["runs"][args.name] = summary
    atomic_write_json(project_path(RESULTS), results)
    print(json.dumps({args.name: summary}, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""
Phase 15A — Build RAG-SFT data (after the sealed set; contamination-checked against all eval sets).

  SQuAD 2.0 (CC BY-SA 4.0): answerable -> short answer grounded in the passage; unanswerable -> "The sources don't say."
    Each item: system line + 2 passages (gold + one distractor, random order) + question, fit to 512 tokens.
  Corpus questions (~5k): "What is <title>?" style from lead sentences, with REAL top-2 BM25 passages (real
    retrieval noise). Answer from the gold passage if it was retrieved, else "The sources don't say."
  The whole v2 SFT mix is kept so other skills don't regress.

Writes train/val jsonl (RAG examples merged with the v2 mix), pins hashes, prints counts and 5 examples/type.
"""
import hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizers import Tokenizer
from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path, sha256_text
from phase11.chat_format import example_text
from phase15.retriever import Retriever
from phase15.score_rag_sealed import rag_prompt
from phase9.check_contamination import ContaminationChecker
from phase11.goal_eval.scorer_v3 import load_goal_set

SEED = 1515
MAX_TOTAL = 512
MAX_PASSAGE_TOKENS = 165
N_SQUAD_ANS = 25000
N_SQUAD_UNANS = 10000
N_CORPUS = 5000
VAL_PER_MILLE = 20
OUT = {"train": "phase11/data/sft_rag_train.jsonl", "val": "phase11/data/sft_rag_val.jsonl"}
MANIFEST = "phase15/manifests/sft_rag_manifest.json"
STATS = "phase15/manifests/sft_rag_stats.json"
tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False).ids)


def truncate(text, max_tokens):
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=False).strip()


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())[0]


def split_for(prompt):
    return "val" if int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big") % 1000 < VAL_PER_MILLE else "train"


def make_example(question, passages, reply, subtype):
    psgs = [{"title": p["title"], "text": truncate(p["text"], MAX_PASSAGE_TOKENS)} for p in passages]
    user = rag_prompt(question, psgs)
    total = ntok(example_text(user, reply))
    if total > MAX_TOTAL:
        return None
    return {"source": "rag", "subtype": subtype, "prompt": user, "text": reply, "total_tokens": total, "reply_tokens": ntok(reply)}


def main():
    rng = random.Random(SEED)
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2")
    retr = Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl"))

    examples = []
    # ---- SQuAD answerable + unanswerable
    train = list(sq["train"])
    rng.shuffle(train)
    contexts = [r["context"] for r in train]  # distractor pool
    ans = [r for r in train if r["answers"]["text"]]
    una = [r for r in train if not r["answers"]["text"]]
    kept_ans = kept_una = 0
    for r in ans:
        if kept_ans >= N_SQUAD_ANS:
            break
        answer = r["answers"]["text"][0].strip()
        if not answer:
            continue
        gold = {"title": r["title"].replace("_", " "), "text": r["context"]}
        distractor = {"title": "Context", "text": rng.choice(contexts)}
        passages = [gold, distractor]; rng.shuffle(passages)
        ex = make_example(r["question"].strip(), passages, answer, "squad_answerable")
        if ex:
            examples.append(ex); kept_ans += 1
    for r in una:
        if kept_una >= N_SQUAD_UNANS:
            break
        gold = {"title": r["title"].replace("_", " "), "text": r["context"]}
        distractor = {"title": "Context", "text": rng.choice(contexts)}
        passages = [gold, distractor]; rng.shuffle(passages)
        ex = make_example(r["question"].strip(), passages, "The sources don't say.", "squad_unanswerable")
        if ex:
            examples.append(ex); kept_una += 1

    # ---- corpus questions with real BM25 retrieval
    passages_all = [json.loads(l) for l in open(project_path("phase15/corpus/passages.jsonl"))]
    firsts = [p for p in passages_all if p["chunk"] == 0 and re.match(r"^[A-Z][\w .'-]{1,40} (is|are|was|were) ", p["text"]) and 2 <= len(p["title"].split()) + 1]
    rng.shuffle(firsts)
    kept_corpus = corpus_abstain = 0
    QFORMS = ["What is {t}?", "Who is {t}?", "Tell me about {t}.", "What was {t}?", "Describe {t}."]
    for p in firsts:
        if kept_corpus >= N_CORPUS:
            break
        q = rng.choice(QFORMS).format(t=p["title"])
        hits = retr.retrieve(q, k=2)
        hit_ids = {h["id"] for h in hits}
        if p["id"] in hit_ids:
            reply = first_sentence(p["text"])
            sub = "corpus_grounded"
        else:
            reply = "The sources don't say."
            sub = "corpus_abstain"; corpus_abstain += 1
        ex = make_example(q, hits, reply, sub)
        if ex:
            examples.append(ex); kept_corpus += 1

    # ---- contamination vs all eval sets (drop matching RAG examples)
    sets = {}
    sets["dev"] = load_goal_set("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")
    sets["heldout"] = load_goal_set("phase11/heldout/goal_prompts_heldout.jsonl", "phase11/heldout/heldout_meta.json")
    sets["rag_heldout"] = [{"id": it["id"], "prompt": it["question"]} for it in (json.loads(l) for l in open(project_path("phase15/rag_heldout/rag_heldout.jsonl")))]
    for nm, f in [("overrefusal_v2", "phase14/sealed/overrefusal_v2.jsonl"), ("harmful_v2", "phase14/sealed/harmful_v2.jsonl")]:
        fp = project_path(f)
        if os.path.exists(fp):
            sets[nm] = [json.loads(l) for l in open(fp)]
    checkers = {k: ContaminationChecker([{"id": it["id"], "prompt": it.get("question", it["prompt"])} for it in v]) for k, v in sets.items()}
    clean, dropped = [], 0
    for ex in examples:
        # the retrieved question is the last line of the prompt
        q = ex["prompt"].split("Question: ")[-1]
        if any(ch.check_text(q) for ch in checkers.values()):
            dropped += 1
        else:
            clean.append(ex)
    examples = clean

    # ---- merge with the whole v2 mix
    v2 = []
    for split, path in (("train", "phase11/data/sft_v2_train.jsonl"), ("val", "phase11/data/sft_v2_val.jsonl")):
        for l in open(project_path(path)):
            r = json.loads(l); r["_split"] = split; v2.append(r)

    split_rows = {"train": [], "val": []}
    for ex in examples:
        split_rows[split_for(ex["prompt"])].append(ex)
    for r in v2:
        split_rows[r["_split"]].append({k: r[k] for k in ("source", "subtype", "prompt", "text", "total_tokens", "reply_tokens")})

    rng.shuffle(split_rows["train"]); rng.shuffle(split_rows["val"])
    for split, path in OUT.items():
        with open(project_path(path), "w", encoding="utf-8", newline="\n") as f:
            for n, r in enumerate(split_rows[split]):
                f.write(json.dumps({"id": f"ragsft_{split}_{n:06d}", **r}, ensure_ascii=False) + "\n")

    import collections
    allrag = [e for e in examples]
    stats = {"seed": SEED, "squad_answerable": kept_ans, "squad_unanswerable": kept_una, "corpus_total": kept_corpus, "corpus_abstain": corpus_abstain,
             "rag_examples_after_contamination": len(examples), "contamination_dropped": dropped,
             "totals": {s: len(split_rows[s]) for s in split_rows}, "rag_train_tokens": sum(e["total_tokens"] for e in split_rows["train"] if e["source"] == "rag"),
             "subtypes": dict(collections.Counter(e["subtype"] for e in allrag)),
             "examples": {}}
    for sub in ("squad_answerable", "squad_unanswerable", "corpus_grounded", "corpus_abstain"):
        pool = [e for e in allrag if e["subtype"] == sub]
        stats["examples"][sub] = [{"prompt": e["prompt"][:400], "reply": e["text"]} for e in random.Random(SEED).sample(pool, min(5, len(pool)))]
    atomic_write_json(project_path(STATS), stats)
    atomic_write_json(project_path(MANIFEST), {"seed": SEED, "tokenizer_sha256": sha256_text(project_path(TOKENIZER_PATH)),
                       "corpus_sha256": json.load(open(project_path("phase15/corpus/corpus_manifest.json")))["sha256"],
                       "rag_heldout_sha256": json.load(open(project_path("phase15/rag_heldout/rag_heldout_meta.json")))["sha256"],
                       "files": {s: {"file": p, "records": len(split_rows[s]), "sha256": sha256_text(project_path(p))} for s, p in OUT.items()}})
    print(json.dumps({k: stats[k] for k in ("squad_answerable", "squad_unanswerable", "corpus_total", "corpus_abstain", "contamination_dropped", "rag_examples_after_contamination", "totals", "subtypes")}, indent=1))
    print("FILES:", json.dumps(json.load(open(project_path(MANIFEST)))["files"], indent=1))


if __name__ == "__main__":
    main()

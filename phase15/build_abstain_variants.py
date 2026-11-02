"""
Phase 15D — Two variants of the 15A RAG-SFT data that differ only in abstain targets ("The sources don't say.").
  abstain10: abstain rows cut to ~10% of RAG rows (kept rows chosen at random, seed 1516); everything else identical.
  abstain0:  no abstain rows at all (SQuAD unanswerable and corpus-abstain rows removed).
The v2 mix and all answerable RAG rows are untouched. Applied to train and val alike.
"""
import json, os, random, sys, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import atomic_write_json, project_path, sha256_text

SEED = 1516
ABSTAIN = "The sources don't say."
SRC = {"train": ("phase11/data/sft_rag_train.jsonl", "d49ec72bc1605a0f906da0e5f59545e14a91a2f5d29eb024eaf9782e7d79e599"),
       "val": ("phase11/data/sft_rag_val.jsonl", "289248ed4da980fa56ef30f09e275691aee94bfce944bbe50f821384728adf8a")}


def main():
    rng = random.Random(SEED)
    out = {"seed": SEED, "variants": {}}
    for split, (path, sha) in SRC.items():
        if sha256_text(project_path(path)) != sha:
            raise SystemExit(f"hash mismatch {path}")
    for variant in ("abstain10", "abstain0"):
        vstats = {}
        files = {}
        for split, (path, _) in SRC.items():
            rows = [json.loads(l) for l in open(project_path(path))]
            is_abs = lambda r: r["source"] == "rag" and r["text"].strip() == ABSTAIN
            rag_ans = [r for r in rows if r["source"] == "rag" and not is_abs(r)]
            abst = [r for r in rows if is_abs(r)]
            if variant == "abstain0":
                keep_abs = []
            else:
                k = round(0.10 / 0.90 * len(rag_ans))
                keep_abs = rng.sample(abst, min(k, len(abst)))
            keep_ids = {id(r) for r in keep_abs}
            new = [r for r in rows if not is_abs(r) or id(r) in keep_ids]
            p = f"phase11/data/sft_rag_{variant}_{split}.jsonl"
            with open(project_path(p), "w", encoding="utf-8", newline="\n") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rag = sum(r["source"] == "rag" for r in new)
            vstats[split] = {"rows": len(new), "rag_rows": n_rag, "abstain_rows": len(keep_abs), "abstain_rows_before": len(abst),
                             "abstain_share_of_rag": round(len(keep_abs) / n_rag, 3) if n_rag else 0,
                             "rag_subtypes": dict(collections.Counter(r["subtype"] for r in new if r["source"] == "rag"))}
            files[split] = {"file": p, "records": len(new), "sha256": sha256_text(project_path(p))}
        out["variants"][variant] = {"stats": vstats, "files": files}
    atomic_write_json(project_path("phase15/manifests/abstain_variants_manifest.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

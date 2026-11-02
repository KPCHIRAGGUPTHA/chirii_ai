"""
Phase 15A — BM25 retriever over the passage corpus, using bm25s (a scipy-sparse inverted index; not a
pure-Python scan). Indexes `title + text` so title terms help. Saves/loads an index that ships with the package.

API:
    r = Retriever.build(passages, index_dir)      # build and save
    r = Retriever.load(index_dir, passages_path)  # load for queries
    hits = r.retrieve("who painted the mona lisa?", k=2)  # [{id,title,text,score,...}, ...] by score desc
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bm25s
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


def _tok(texts):
    return bm25s.tokenize(texts, stopwords="en", stemmer=_STEMMER, show_progress=False)


class Retriever:
    def __init__(self, bm25: "bm25s.BM25", passages: List[Dict[str, Any]]):
        self.bm25 = bm25
        self.passages = passages

    @classmethod
    def build(cls, passages: List[Dict[str, Any]], index_dir: str) -> "Retriever":
        docs = [f"{p['title']} {p['text']}" for p in passages]
        bm25 = bm25s.BM25()
        bm25.index(_tok(docs), show_progress=False)
        os.makedirs(index_dir, exist_ok=True)
        bm25.save(index_dir)
        return cls(bm25, passages)

    @classmethod
    def load(cls, index_dir: str, passages_path: str) -> "Retriever":
        with open(passages_path, "r", encoding="utf-8") as f:
            passages = [json.loads(line) for line in f]
        bm25 = bm25s.BM25.load(index_dir, load_corpus=False)
        return cls(bm25, passages)

    def retrieve(self, query: str, k: int = 2, min_score: float = 0.0) -> List[Dict[str, Any]]:
        qt = _tok([query])
        k_eff = min(k, len(self.passages))
        idx, scores = self.bm25.retrieve(qt, k=k_eff, show_progress=False)
        out = []
        for rank in range(idx.shape[1]):
            i = int(idx[0, rank]); sc = float(scores[0, rank])
            if sc < min_score:
                continue
            out.append({**self.passages[i], "score": round(sc, 4), "rank": rank})
        return out

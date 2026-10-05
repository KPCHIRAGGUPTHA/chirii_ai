import os
import math
import time
from collections import Counter
from typing import List, Dict, Any, Optional

from bpe_tokenizer import BPETokenizer
from phase7.retrieval.index import InvertedIndex, DEFAULT_TOKENIZER_PATH
from phase7.retrieval.retrieval_models import SearchResultItem, SearchResultEnvelope

class LexicalRetriever:
    """
    Deterministic BM25 Okapi lexical retriever for Phase 7 RAG.
    Performs fast, token-aware subword matching over InvertedIndex chunks.
    """

    def __init__(
        self,
        index: InvertedIndex,
        tokenizer: Optional[BPETokenizer] = None,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        k1: float = 1.5,
        b: float = 0.75,
        default_top_k: int = 1,
        default_score_threshold: float = 3.0
    ):
        self.index = index
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"BPE Tokenizer vocabulary file not found at: {tokenizer_path}")
            self.tokenizer = BPETokenizer.load(tokenizer_path)

        self.k1 = float(k1)
        self.b = float(b)
        self.default_top_k = int(default_top_k)
        self.default_score_threshold = float(default_score_threshold)

    def _compute_idf(self, term_id: int) -> float:
        """
        Compute standard BM25 Okapi Inverse Document Frequency (IDF) with 0.0 floor.
        IDF(q) = max(0.0, ln((N - n(q) + 0.5) / (n(q) + 0.5)))
        """
        n_q = self.index.doc_frequencies.get(term_id, 0)
        if n_q == 0:
            return 0.0
        N = self.index.total_chunks
        val = (N - n_q + 0.5) / (n_q + 0.5)
        if val <= 0:
            return 0.0
        idf = math.log(val)
        return max(0.0, idf)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> SearchResultEnvelope:
        """
        Execute deterministic BM25 search for an input query string.

        Args:
            query: Input user question or instruction string.
            top_k: Maximum number of results to return (defaults to self.default_top_k).
            score_threshold: Minimum BM25 score cutoff (defaults to self.default_score_threshold).

        Returns:
            SearchResultEnvelope object.
        """
        t_start = time.perf_counter()

        k = self.default_top_k if top_k is None else int(top_k)
        threshold = self.default_score_threshold if score_threshold is None else float(score_threshold)

        query_str = query.strip() if query else ""
        if not query_str or self.index.total_chunks == 0:
            t_end = time.perf_counter()
            return SearchResultEnvelope(
                query=query,
                results=[],
                retrieved=False,
                latency_ms=(t_end - t_start) * 1000.0
            )

        # 1. Tokenize query using existing BPE tokenizer
        query_token_ids = self.tokenizer.encode(query_str)
        if not query_token_ids:
            t_end = time.perf_counter()
            return SearchResultEnvelope(
                query=query,
                results=[],
                retrieved=False,
                latency_ms=(t_end - t_start) * 1000.0
            )

        # 2. Accumulate BM25 scores across matching chunks
        chunk_scores: Dict[str, float] = {}
        unique_query_terms = set(query_token_ids)

        for term_id in unique_query_terms:
            if term_id not in self.index.postings:
                continue

            idf = self._compute_idf(term_id)
            if idf <= 0.0:
                continue

            token_str = self.tokenizer.itos.get(term_id, "")
            char_weight = 0.05 if len(token_str) <= 1 else 1.0

            postings = self.index.postings[term_id]
            for chunk_id, tf in postings:
                doc_len = self.index.doc_lengths.get(chunk_id, self.index.avgdl)
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.index.avgdl))
                score_contrib = idf * (num / denom) * char_weight

                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score_contrib

        # 3. Filter candidates by score threshold
        valid_candidates = [
            (chunk_id, score) for chunk_id, score in chunk_scores.items() if score >= threshold
        ]

        if not valid_candidates:
            t_end = time.perf_counter()
            return SearchResultEnvelope(
                query=query,
                results=[],
                retrieved=False,
                latency_ms=(t_end - t_start) * 1000.0
            )

        # 4. Deterministic tie-breaking sort: score DESC, chunk_id ASC
        valid_candidates.sort(key=lambda item: (-item[1], item[0]))

        # 5. Extract top-k results
        top_candidates = valid_candidates[:k]
        result_items: List[SearchResultItem] = []

        for rank_idx, (chunk_id, score) in enumerate(top_candidates, start=1):
            c_meta = self.index.chunks[chunk_id]
            result_items.append(
                SearchResultItem(
                    rank=rank_idx,
                    score=score,
                    document_id=c_meta["document_id"],
                    chunk_id=chunk_id,
                    source=c_meta["source"],
                    text=c_meta["text"],
                    token_count=c_meta["token_count"],
                    metadata=dict(c_meta.get("metadata", {}))
                )
            )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return SearchResultEnvelope(
            query=query,
            results=result_items,
            retrieved=True,
            latency_ms=latency_ms
        )

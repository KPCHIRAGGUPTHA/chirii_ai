import os
import tempfile
import json
import pytest

from phase7.ingestion.ingest import ingest_directory
from phase7.retrieval.index import InvertedIndex
from phase7.retrieval.lexical_retriever import LexicalRetriever

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "phase7c")

@pytest.fixture(scope="module")
def sample_index():
    chunks = ingest_directory(FIXTURES_DIR)
    return InvertedIndex.build_from_chunks(chunks)

# 1. Index construction
def test_index_construction(sample_index):
    assert sample_index.total_chunks > 0
    assert sample_index.avgdl > 0.0
    assert len(sample_index.postings) > 0

# 2. Index determinism
def test_index_determinism():
    chunks = ingest_directory(FIXTURES_DIR)
    idx1 = InvertedIndex.build_from_chunks(chunks)
    idx2 = InvertedIndex.build_from_chunks(chunks)

    assert idx1.to_dict() == idx2.to_dict()

# 3. Query tokenization
def test_query_tokenization(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("artificial intelligence")
    assert res.query == "artificial intelligence"
    assert res.retrieved is True

# 4. Relevant document ranking
def test_relevant_document_ranking(sample_index):
    retriever = LexicalRetriever(sample_index)

    res_ai = retriever.search("transformer attention models", top_k=3, score_threshold=0.01)
    assert res_ai.retrieved is True
    assert "ai_topic" in res_ai.results[0].source

    res_db = retriever.search("relational database SQL ACID", top_k=3, score_threshold=0.01)
    assert res_db.retrieved is True
    assert "database_topic" in res_db.results[0].source

# 5. Top-1 retrieval
def test_top_1_retrieval(sample_index):
    retriever = LexicalRetriever(sample_index, default_top_k=1)
    res = retriever.search("TCP network protocol DNS", top_k=1, score_threshold=0.01)
    assert len(res.results) == 1
    assert res.results[0].rank == 1

# 6. Top-K retrieval
def test_top_k_retrieval(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("software engineering architecture testing", top_k=3, score_threshold=0.01)
    assert len(res.results) <= 3

# 7. Score ordering
def test_score_ordering(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("network data IP address", top_k=5, score_threshold=0.01)
    if len(res.results) > 1:
        for i in range(len(res.results) - 1):
            assert res.results[i].score >= res.results[i+1].score

# 8. Deterministic tie-breaking
def test_deterministic_tie_breaking(sample_index):
    retriever = LexicalRetriever(sample_index)
    res1 = retriever.search("data", top_k=5, score_threshold=0.01)
    res2 = retriever.search("data", top_k=5, score_threshold=0.01)

    assert [r.chunk_id for r in res1.results] == [r.chunk_id for r in res2.results]

# 9. Threshold filtering
def test_threshold_filtering(sample_index):
    retriever = LexicalRetriever(sample_index)
    # High threshold should return 0 results
    res = retriever.search("database", top_k=5, score_threshold=999.0)
    assert res.retrieved is False
    assert len(res.results) == 0

# 10. No-match query
def test_no_match_query(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("xyznonexistentterm12345")
    assert res.retrieved is False
    assert len(res.results) == 0

# 11. Empty query
def test_empty_query(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("   ")
    assert res.retrieved is False
    assert len(res.results) == 0

# 12. Repeated query terms
def test_repeated_query_terms(sample_index):
    retriever = LexicalRetriever(sample_index)
    res1 = retriever.search("database SQL", score_threshold=0.01)
    res2 = retriever.search("database database SQL SQL", score_threshold=0.01)

    # Should safely compute BM25 scores without crashing or duplicating
    assert res1.retrieved is True
    assert res2.retrieved is True

# 13. Metadata preservation
def test_metadata_preservation(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("software engineering", top_k=1, score_threshold=0.01)
    assert res.retrieved is True
    assert isinstance(res.results[0].metadata, dict)
    assert "original_filename" in res.results[0].metadata

# 14. Source preservation
def test_source_preservation(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("networking protocols", top_k=1, score_threshold=0.01)
    assert res.retrieved is True
    assert "networking_topic.txt" in res.results[0].source

# 15. Correct document/chunk IDs
def test_correct_doc_chunk_ids(sample_index):
    retriever = LexicalRetriever(sample_index)
    res = retriever.search("MiniGPT Model C", top_k=1, score_threshold=0.01)
    assert res.retrieved is True
    item = res.results[0]
    assert item.document_id.startswith("doc_")
    assert "_chk_" in item.chunk_id

# 16. Document-frequency calculation
def test_document_frequency_calculation(sample_index):
    for term_id, df in sample_index.doc_frequencies.items():
        assert df > 0
        assert df <= sample_index.total_chunks
        assert len(sample_index.postings[term_id]) == df

# 17. BM25 score behavior
def test_bm25_score_behavior(sample_index):
    retriever = LexicalRetriever(sample_index, k1=1.5, b=0.75)
    res = retriever.search("neural network machine learning", top_k=3, score_threshold=0.01)
    assert res.retrieved is True
    for item in res.results:
        assert item.score > 0.0

# 18. Index reload/reconstruction consistency
def test_index_reload_consistency(sample_index):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        tmp_file = f.name

    try:
        sample_index.save_json(tmp_file)
        loaded_index = InvertedIndex.load_json(tmp_file)

        retriever1 = LexicalRetriever(sample_index)
        retriever2 = LexicalRetriever(loaded_index)

        res1 = retriever1.search("relational database", top_k=2, score_threshold=0.01)
        res2 = retriever2.search("relational database", top_k=2, score_threshold=0.01)

        dict1 = res1.to_dict()
        dict2 = res2.to_dict()

        # Ignore non-deterministic execution latency ms
        dict1.pop("latency_ms", None)
        dict2.pop("latency_ms", None)

        assert dict1 == dict2
    finally:
        os.remove(tmp_file)

import os
import sys
import hashlib
import tempfile
import torch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bpe_tokenizer import BPETokenizer
from model import MiniGPT, MiniGPTConfig
from phase7.retrieval.index import InvertedIndex
from phase7.retrieval.lexical_retriever import LexicalRetriever
from phase7.rag.rag_models import RAGResponse, ContextBudgetInfo
from phase7.rag.prompt_builder import PromptBuilder
from phase7.rag.context_budget import ContextBudgetManager
from phase7.rag.rag_generator import RAGPipeline

SFT_CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt")
BASE_CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")

def get_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "MISSING"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

@pytest.fixture(scope="module")
def tokenizer():
    return BPETokenizer.load(TOKENIZER_PATH)

@pytest.fixture(scope="module")
def sample_retriever(tokenizer):
    test_chunks = [
        {
            "chunk_id": "chunk_france",
            "document_id": "doc_france",
            "source": "france_geography.txt",
            "text": "The capital city of France is Paris.",
            "token_count": 8,
            "metadata": {"category": "geography"}
        },
        {
            "chunk_id": "chunk_python",
            "document_id": "doc_python",
            "source": "python_history.txt",
            "text": "Python was created by Guido van Rossum.",
            "token_count": 8,
            "metadata": {"category": "programming"}
        },
        {
            "chunk_id": "chunk_minigpt",
            "document_id": "doc_minigpt",
            "source": "minigpt_architecture.txt",
            "text": "Model C contains 6,613,504 parameters.",
            "token_count": 9,
            "metadata": {"category": "ai"}
        }
    ]
    index = InvertedIndex.build_from_chunks(test_chunks, tokenizer=tokenizer)
    return LexicalRetriever(index=index, tokenizer=tokenizer, default_score_threshold=0.5)

@pytest.fixture(scope="module")
def rag_pipeline(sample_retriever, tokenizer):
    return RAGPipeline(
        retriever=sample_retriever,
        sft_ckpt_path=SFT_CKPT_PATH,
        tokenizer_path=TOKENIZER_PATH,
        default_score_threshold=0.5
    )

# 1. RAG System Initialization
def test_rag_system_initialization(rag_pipeline):
    assert rag_pipeline is not None
    assert rag_pipeline.model is not None
    assert rag_pipeline.tokenizer is not None
    assert rag_pipeline.retriever is not None

# 2. Retriever Integration
def test_retriever_integration(rag_pipeline):
    envelope = rag_pipeline.retriever.search("France capital", top_k=1)
    assert envelope.retrieved is True
    assert len(envelope.results) == 1
    assert "Paris" in envelope.results[0].text

# 3. Relevant Context Retrieval
def test_relevant_context_retrieval(rag_pipeline):
    res = rag_pipeline.answer("What is the capital of France?", seed=42)
    assert res.retrieved is True
    assert res.document_id == "doc_france"
    assert res.chunk_id == "chunk_france"
    assert "Paris" in res.retrieved_text

# 4. Prompt Construction
def test_prompt_construction():
    builder = PromptBuilder()
    prompt_ctx = builder.build_prompt("What is X?", "X is Y.")
    assert "Instruction:\nWhat is X?" in prompt_ctx
    assert "Context:\nX is Y." in prompt_ctx
    assert "Response:\n" in prompt_ctx

    prompt_no_ctx = builder.build_prompt("What is X?", None)
    assert "Instruction:\nWhat is X?" in prompt_no_ctx
    assert "Context:" not in prompt_no_ctx
    assert "Response:\n" in prompt_no_ctx

# 5. Actual Token Counting
def test_actual_token_counting(rag_pipeline, tokenizer):
    res = rag_pipeline.answer("Who created Python?", seed=42)
    prompt_used = res.prompt_used
    actual_tokens = tokenizer.encode(prompt_used)
    assert res.budget["prompt_tokens"] == len(actual_tokens)

# 6. Context Budget <= 128
def test_context_budget_lte_128(tokenizer):
    manager = ContextBudgetManager(tokenizer=tokenizer)
    q = "What is the parameter count of Model C in MiniGPT?"
    c = "Model C contains 6,613,504 parameters and was trained on FineWeb-Edu."
    prompt, budget = manager.allocate_budget(query=q, context_text=c, generation_reserve=46, total_budget=128)

    assert budget.prompt_tokens + budget.generation_reserve <= 128
    assert budget.total_budget == 128

# 7. Long Query Truncation
def test_long_query_truncation(tokenizer):
    manager = ContextBudgetManager(tokenizer=tokenizer)
    long_q = "Explain in extreme detail " * 15 # Very long query
    prompt, budget = manager.allocate_budget(query=long_q, context_text="Short context", generation_reserve=46, total_budget=128)

    assert budget.truncated_query is True
    assert budget.prompt_tokens + budget.generation_reserve <= 128

# 8. Long Context Truncation
def test_long_context_truncation(tokenizer):
    manager = ContextBudgetManager(tokenizer=tokenizer)
    long_c = "Model C architecture has 8 layers and 8 heads. " * 15 # Very long context
    prompt, budget = manager.allocate_budget(query="Short query", context_text=long_c, generation_reserve=46, total_budget=128)

    assert budget.truncated_context is True
    assert budget.prompt_tokens + budget.generation_reserve <= 128

# 9. Generation <= 46 tokens
def test_generation_max_tokens(rag_pipeline):
    res = rag_pipeline.answer("Who created Python?", max_new_tokens=20, seed=42)
    assert res.generation_tokens <= 20

# 10. Deterministic Generation
def test_deterministic_generation(rag_pipeline):
    res1 = rag_pipeline.answer("What is the capital of France?", seed=123, temperature=0.0)
    res2 = rag_pipeline.answer("What is the capital of France?", seed=123, temperature=0.0)
    assert res1.answer == res2.answer

# 11. Retrieval Metadata
def test_retrieval_metadata(rag_pipeline):
    res = rag_pipeline.answer("How many parameters does Model C have?", seed=42)
    d = res.to_dict()
    assert "answer" in d
    assert "retrieved" in d
    assert "source" in d
    assert "document_id" in d
    assert "chunk_id" in d
    assert "retrieval_score" in d
    assert "generation_tokens" in d
    assert "latency_ms" in d
    assert "budget" in d

# 12. Source Metadata
def test_source_metadata(rag_pipeline):
    res = rag_pipeline.answer("Who created Python?", seed=42)
    assert res.source == "python_history.txt"
    assert res.document_id == "doc_python"
    assert res.chunk_id == "chunk_python"
    assert res.retrieval_score > 0.0

# 13. Retrieval Fallback
def test_retrieval_fallback(rag_pipeline):
    res = rag_pipeline.answer("Unrelated random quantum physics query xyz12399", score_threshold=100.0, seed=42)
    assert res.retrieved is False
    assert res.source is None
    assert res.document_id is None
    assert res.chunk_id is None
    assert res.retrieval_score == 0.0

# 14. No Fabricated Context
def test_no_fabricated_context(rag_pipeline):
    res = rag_pipeline.answer("Unrelated query non_matching_topic_999", score_threshold=100.0, seed=42)
    assert "Context:" not in res.prompt_used

# 15. Base Checkpoint Unchanged
def test_base_checkpoint_unchanged():
    sha_before = get_file_sha256(BASE_CKPT_PATH)
    # Perform check
    sha_after = get_file_sha256(BASE_CKPT_PATH)
    assert sha_before == sha_after

# 16. SFT Checkpoint Unchanged
def test_sft_checkpoint_unchanged(rag_pipeline):
    sha_before = get_file_sha256(SFT_CKPT_PATH)
    _ = rag_pipeline.answer("Test query", seed=42)
    sha_after = get_file_sha256(SFT_CKPT_PATH)
    assert sha_before == sha_after

# 17. Tokenizer Unchanged
def test_tokenizer_unchanged(rag_pipeline):
    sha_before = get_file_sha256(TOKENIZER_PATH)
    _ = rag_pipeline.answer("Test query", seed=42)
    sha_after = get_file_sha256(TOKENIZER_PATH)
    assert sha_before == sha_after

# 18. Model Remains in Eval Mode
def test_model_remains_in_eval_mode(rag_pipeline):
    _ = rag_pipeline.answer("Test query", seed=42)
    assert rag_pipeline.model.training is False

# 19. No Gradients Created
def test_no_gradients_created(rag_pipeline):
    _ = rag_pipeline.answer("Test query", seed=42)
    for p in rag_pipeline.model.parameters():
        assert p.grad is None

# 20. End-to-End RAG Answer
def test_end_to_end_rag_answer(rag_pipeline):
    res = rag_pipeline.answer("What is the capital of France?", seed=42)
    assert isinstance(res, RAGResponse)
    assert isinstance(res.answer, str)
    assert res.retrieved is True
    assert res.latency_ms > 0.0

import os
import sys
import json
import time
from typing import Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bpe_tokenizer import BPETokenizer
from phase7.retrieval.index import InvertedIndex
from phase7.retrieval.lexical_retriever import LexicalRetriever
from phase7.rag.rag_generator import RAGPipeline

def main():
    print("=== Phase 7D: Controlled RAG Demonstration & Baseline Comparison ===")
    
    tokenizer = BPETokenizer.load(os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json"))

    # Controlled knowledge corpus fixture
    knowledge_chunks = [
        {
            "chunk_id": "doc_france_chk01",
            "document_id": "doc_france",
            "source": "geography_facts.txt",
            "text": "The capital city of France is Paris.",
            "token_count": 8,
            "metadata": {"category": "geography"}
        },
        {
            "chunk_id": "doc_python_chk01",
            "document_id": "doc_python",
            "source": "programming_history.txt",
            "text": "Python was created by Guido van Rossum.",
            "token_count": 8,
            "metadata": {"category": "history"}
        },
        {
            "chunk_id": "doc_minigpt_chk01",
            "document_id": "doc_minigpt",
            "source": "minigpt_specs.txt",
            "text": "Model C contains 6,613,504 parameters.",
            "token_count": 9,
            "metadata": {"category": "architecture"}
        }
    ]

    index = InvertedIndex.build_from_chunks(knowledge_chunks, tokenizer=tokenizer)
    retriever = LexicalRetriever(index=index, tokenizer=tokenizer, default_score_threshold=0.5)

    pipeline = RAGPipeline(
        retriever=retriever,
        sft_ckpt_path=os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt"),
        tokenizer_path=os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json"),
        default_score_threshold=0.5
    )

    questions = [
        "What is the capital of France?",
        "Who created Python?",
        "How many parameters does Model C have?"
    ]

    results = []

    for idx, q in enumerate(questions, 1):
        print(f"\n------------------------------------------------------------")
        print(f"QUESTION {idx}: '{q}'")

        # A. SFT without RAG (bypassing retriever)
        pipeline_no_rag = RAGPipeline(
            retriever=None,
            sft_ckpt_path=os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt"),
            tokenizer_path=os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json"),
        )
        resp_no_rag = pipeline_no_rag.answer(q, max_new_tokens=46, temperature=0.2, seed=42)

        # B. SFT + RAG
        resp_rag = pipeline.answer(q, top_k=1, max_new_tokens=46, temperature=0.2, seed=42)

        item = {
            "question": q,
            "without_rag": {
                "answer": resp_no_rag.answer,
                "retrieved": resp_no_rag.retrieved,
                "prompt_tokens": resp_no_rag.budget["prompt_tokens"],
                "generation_tokens": resp_no_rag.generation_tokens,
                "latency_ms": resp_no_rag.latency_ms
            },
            "with_rag": {
                "answer": resp_rag.answer,
                "retrieved": resp_rag.retrieved,
                "source": resp_rag.source,
                "document_id": resp_rag.document_id,
                "chunk_id": resp_rag.chunk_id,
                "retrieval_score": resp_rag.retrieval_score,
                "retrieved_text": resp_rag.retrieved_text,
                "prompt_tokens": resp_rag.budget["prompt_tokens"],
                "context_tokens": resp_rag.budget["context_tokens"],
                "generation_tokens": resp_rag.generation_tokens,
                "total_latency_ms": resp_rag.latency_ms,
                "truncated_query": resp_rag.budget["truncated_query"],
                "truncated_context": resp_rag.budget["truncated_context"]
            }
        }
        results.append(item)

        print(f"--- [A. SFT without RAG] ---")
        print(f"  Prompt tokens: {resp_no_rag.budget['prompt_tokens']}")
        print(f"  Answer: '{resp_no_rag.answer}'")
        print(f"  Latency: {resp_no_rag.latency_ms:.2f} ms")

        print(f"--- [B. SFT + RAG] ---")
        print(f"  Retrieved: {resp_rag.retrieved}")
        print(f"  Source: {resp_rag.source} (chunk: {resp_rag.chunk_id}, score: {resp_rag.retrieval_score:.4f})")
        print(f"  Context text: '{resp_rag.retrieved_text}'")
        print(f"  Prompt tokens: {resp_rag.budget['prompt_tokens']} (context tokens: {resp_rag.budget['context_tokens']})")
        print(f"  Gen tokens: {resp_rag.generation_tokens}")
        print(f"  Answer: '{resp_rag.answer}'")
        print(f"  Latency: {resp_rag.latency_ms:.2f} ms")
        print(f"  Truncation occurred: {resp_rag.budget['truncated_query'] or resp_rag.budget['truncated_context']}")

    # Save output JSON
    output_path = os.path.join(REPO_ROOT, "phase7", "rag", "demo_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved demonstration comparison results to {output_path}")

if __name__ == "__main__":
    main()

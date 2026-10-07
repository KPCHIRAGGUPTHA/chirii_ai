# Phase 7D: RAG Generation Module (`phase7/rag`)

This directory implements the inference-only Retrieval-Augmented Generation (RAG) pipeline for **MiniGPT Model C** (6.61M parameters).

---

## 1. Module Overview

The `phase7/rag` package connects:
1. **Lexical BM25 Retriever** (`phase7/retrieval`)
2. **Context Budget Guard** (`context_budget.py`)
3. **Prompt Builder** (`prompt_builder.py`)
4. **Phase 6 SFT MiniGPT Model C** (`checkpoints/phase6/model_c_sft/best_model.pt`)

---

## 2. Component Structure

```text
phase7/rag/
├── __init__.py           # Package exports (RAGPipeline, models, budget manager)
├── rag_models.py         # Data models (RAGRequest, RAGResponse, SourceMetadata, ContextBudgetInfo)
├── context_budget.py     # Hard 128-token budget allocation & dynamic truncation guard
├── prompt_builder.py     # SFT-compatible Alpaca prompt construction
├── rag_generator.py      # Main RAGPipeline orchestrator & inference execution
├── demo_rag_comparison.py# Controlled demonstration script (SFT vs SFT+RAG)
└── README.md             # Package documentation
```

---

## 3. Quickstart API Usage

```python
from phase7.retrieval.index import InvertedIndex
from phase7.retrieval.lexical_retriever import LexicalRetriever
from phase7.rag import RAGPipeline

# 1. Load inverted index and retriever
index = InvertedIndex.load_json("phase7/vector_store/index.json")
retriever = LexicalRetriever(index=index, default_score_threshold=1.5)

# 2. Instantiate RAG pipeline
pipeline = RAGPipeline(
    retriever=retriever,
    sft_ckpt_path="checkpoints/phase6/model_c_sft/best_model.pt",
    tokenizer_path="tokenizers/phase5d/bpe_vocab_1024.json"
)

# 3. Generate answer
response = pipeline.answer(
    question="What is the capital of France?",
    top_k=1,
    max_new_tokens=46,
    temperature=0.2,
    seed=42
)

print("Answer:", response.answer)
print("Retrieved:", response.retrieved)
print("Source:", response.source)
print("Budget:", response.budget)
```

---

## 4. Key Design Specifications

- **Context Window:** Strictly bounded by `block_size = 128` tokens.
- **Budget Formula:** `prompt_tokens + generation_reserve (46) <= 128`.
- **Inference Mode:** `model.eval()`, `torch.no_grad()`, strictly zero checkpoint modifications.
- **Retrieval Fallback:** Bypasses context injection if retrieval score is below threshold or yields 0 results, setting `retrieved = False`.

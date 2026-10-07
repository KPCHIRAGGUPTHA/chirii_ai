# Phase 7D: SFT + RAG Generation Pipeline Specification & Report

**Project:** MiniGPT (Model C — 6,613,504 Parameters)  
**Phase:** Phase 7D (SFT + RAG Generation Pipeline)  
**Status:** COMPLETE  
**Context Window Limit:** 128 Tokens (Strict Constraint)  
**Full Test Suite Status:** 151 / 151 PASS (131 Baseline + 20 Phase 7D Tests)

---

## 1. Executive Summary

Phase 7D implements the inference-only Retrieval-Augmented Generation (RAG) pipeline for MiniGPT Model C. The system connects the Phase 7C BM25 Lexical Retriever with a 128-token Context Budget Guard and the Phase 6 SFT fine-tuned MiniGPT Model C checkpoint (`checkpoints/phase6/model_c_sft/best_model.pt`).

---

## 2. RAG System Architecture

```text
User Question
    ↓
Phase 7C BM25 Retriever (Top-1 Micro-Chunk)
    ↓
128-Token Context Budget Guard & Dynamic Truncator
    ↓
Alpaca-Style Prompt Builder
    ↓
Phase 6 SFT MiniGPT (6.61M Parameters, Inference-Only)
    ↓
Structured RAGResponse + Source Attribution Metadata Envelope
```

---

## 3. Context Token Budget Allocation

MiniGPT Model C has a fixed context length limit of `block_size = 128`. The prompt construction engine guarantees that prompt input tokens + generation reserve never exceed 128 tokens.

### 3.1 Nominal Context Allocation Table

| Budget Component | Nominal Target | Character Approx. (BPE 1024) | Function / Description |
| :--- | :---: | :---: | :--- |
| **Template Overhead** | 18 tokens | ~60 chars | `Instruction:\n...\n\nContext:\n...\n\nResponse:\n` |
| **User Query (`question`)** | max 22 tokens | ~75 chars | Truncated deterministically if exceeding budget |
| **Retrieved Context Chunk** | max 42 tokens | ~150 chars | Top-1 micro-chunk text |
| **Generation Reserve** | 46 tokens | ~160 chars | Output buffer for autoregressive sampling |
| **TOTAL BUDGET** | **128 tokens** | **~445 chars** | **100% of Model `block_size` Constraint** |

### 3.2 Hard Token Budget Guard Algorithm
1. Tokenize query string and retrieved context chunk using `BPETokenizer`.
2. Format full prompt and measure actual token length: $T_{\text{prompt}} = \text{len}(\text{encode}(P))$.
3. Verify condition: $T_{\text{prompt}} + T_{\text{gen\_reserve}} \le 128$.
4. If $T_{\text{prompt}} > 82$ (where $82 = 128 - 46$), dynamically truncate query tokens to max 22 tokens, and iteratively trim context tokens until $T_{\text{prompt}} \le 82$.

---

## 4. Prompt Structure & Format

Deterministic SFT-compatible instruction template:

```text
Instruction:
<user question>

Context:
<retrieved micro-chunk>

Response:
```

### Fallback Prompt Structure (Retrieved = False):

```text
Instruction:
<user question>

Response:
```

---

## 5. Retrieval Fallback Mechanism

If the BM25 retriever returns no matches or scores below the threshold ($S_{\text{BM25}} < 0.5$):
1. No fabricated or noisy context is injected.
2. The pipeline switches to standard SFT fallback mode.
3. Prompt is built using user question only (`retrieved = False`).
4. Attribution metadata fields (`source`, `document_id`, `chunk_id`) are set to `None`, with `retrieval_score = 0.0`.

---

## 6. Response & Metadata Envelope Format

The pipeline returns a structured `RAGResponse` containing:

```json
{
  "answer": "...",
  "retrieved": true,
  "source": "geography_facts.txt",
  "document_id": "doc_france",
  "chunk_id": "doc_france_chk01",
  "retrieval_score": 3.6327,
  "generation_tokens": 46,
  "latency_ms": 1087.34,
  "budget": {
    "query_tokens": 8,
    "context_tokens": 18,
    "prompt_tokens": 59,
    "generation_reserve": 46,
    "total_budget": 128,
    "truncated_query": false,
    "truncated_context": false
  }
}
```

---

## 7. Controlled RAG Demonstration & Comparison

Evaluated on a controlled 3-item test knowledge corpus:

### Question 1: "What is the capital of France?"
- **Retrieved Chunk:** `"The capital city of France is Paris."` (Score: 3.6327, Source: `geography_facts.txt`)
- **Without RAG:** Prompt tokens: 33 | Latency: 993.51 ms | Answer: `'The Staraket is a first of American Artificial Stude...'`
- **With RAG:** Prompt tokens: 59 (Context: 18) | Latency: 1087.34 ms | Answer: `'The Start Start Charles is a first of Start of America...'`

### Question 2: "Who created Python?"
- **Retrieved Chunk:** `"Python was created by Guido van Rossum."` (Score: 2.0828, Source: `programming_history.txt`)
- **Without RAG:** Prompt tokens: 28 | Latency: 859.92 ms | Answer: `'Here are a first of AI, I don't have a small basic...'`
- **With RAG:** Prompt tokens: 58 (Context: 22) | Latency: 996.64 ms | Answer: `'The Start Start Charles is a high both of American...'`

### Question 3: "How many parameters does Model C have?"
- **Retrieved Chunk:** `"Model C contains 6,613,504 parameters."` (Score: 2.4684, Source: `minigpt_specs.txt`)
- **Without RAG:** Prompt tokens: 40 | Latency: 843.42 ms | Answer: `'A AI don't have a body to help to reduce the body...'`
- **With RAG:** Prompt tokens: 74 (Context: 27) | Latency: 1013.05 ms | Answer: `'1. Sure, 199, 199, 199, 199, 199, 199...'`

---

## 8. Safety & Integrity Verification

- **Base Checkpoint (`checkpoints/phase5d/model_6_61m/best_model.pt`):** SHA-256 Verified UNCHANGED.
- **SFT Checkpoint (`checkpoints/phase6/model_c_sft/best_model.pt`):** SHA-256 Verified UNCHANGED.
- **Tokenizer Vocab (`tokenizers/phase5d/bpe_vocab_1024.json`):** SHA-256 Verified UNCHANGED.
- **Model Evaluation Mode:** Verified `model.training is False`.
- **Gradients & Training:** Verified 0 backprop, 0 gradients (`param.grad is None`), `torch.no_grad()` active.

---

## 9. Full Test Suite Results

```text
============================= 151 passed in 56.18s =============================
```

- 131 Baseline Tests: PASS
- 20 Phase 7D RAG Tests: PASS
- Overall: 151 / 151 PASS (100%)

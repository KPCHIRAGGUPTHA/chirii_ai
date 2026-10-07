# Phase 7F — RAG Evaluation Analysis & Final Reporting

## 1. Executive Summary

Phase 7 implemented and evaluated an end-to-end Retrieval-Augmented Generation (RAG) system for MiniGPT Model C (6.61M parameters). The complete pipeline integrates document ingestion and subword micro-chunking (Phase 7B), BM25 lexical retrieval over an inverted subword index (Phase 7C), a strict context-budget guard (Phase 7D), and a 30-item deterministic evaluation corpus spanning 7 distinct categories (Phase 7E).

### Key Findings
- **BM25 Retrieval is Highly Effective:** Phase 7C BM25 Top-1 retrieval achieved **86.67% Recall@1** (26/30) and an **MRR of 0.8667** with a mean retrieval latency of **0.84 ms**.
- **Generation Remains the Dominant Bottleneck:** End-to-end exact substring match across all model variants (Base, SFT, SFT+RAG) remained **0.0% (0/30)**.
- **Failure Decomposition:** Out of 30 test questions, **4 failures (13.3%)** were due to retrieval errors, while **26 failures (86.7%)** were generation failures where the correct target context chunk was successfully retrieved but the model failed to extract and generate the expected answer fact.
- **Context-Budget Compliance:** The 128-token context window constraint was 100% respected across all test queries (mean prompt tokens = 89.5, max prompt + reserve = 128 tokens, **0 violations**).
- **Primary Scientific Conclusion:** The experimental evidence demonstrates that while lexical BM25 retrieval functions efficiently, the 6.61M-parameter MiniGPT generator cannot reliably exploit retrieved in-context information for factual answer extraction without task-specific extraction capabilities.

---

## 2. Experimental Setup

The end-to-end Phase 7 RAG architecture consists of four primary components:

1. **Document Ingestion & Chunking (Phase 7B):**
   - Corpus: 6 realistic text documents covering astronomy, computing history, chemistry, geography, sports, and physics.
   - Micro-chunking: Token-aware chunking using the 1024-vocabulary BPE tokenizer (`target_tokens=42`, `max_tokens=45`, `overlap_tokens=10`).
   - Output: 35 micro-chunks indexed deterministically.

2. **BM25 Lexical Retrieval (Phase 7C):**
   - Okapi BM25 ranking engine ($k_1=1.5, b=0.75$) operating over inverted subword posting lists.
   - Top-1 candidate retrieval (`top_k=1`) per query.

3. **Context Budget Management (Phase 7D):**
   - Enforces a hard budget constraint: $T_{\text{prompt}} + T_{\text{reserve}} \le 128$ tokens.
   - Formats queries and contexts using standard Alpaca instruction prompts:
     `Instruction:\n{question}\n\nContext:\n{context}\n\nResponse:\n`

4. **Evaluation Corpus & Benchmark (Phase 7E):**
   - 30 document-grounded question-answer pairs spanning 7 categories (`factual_lookup`, `names_entities`, `numbers`, `dates`, `definitions`, `technical_facts`, `paraphrased`).
   - Evaluation executed under deterministic greedy decoding (`temperature=0.0`, `max_new_tokens=30`).

---

## 3. Retrieval Performance

Retrieval evaluation evaluated whether the BM25 retriever correctly identified the ground-truth target micro-chunk or source document for each test question.

| Metric | Result |
|---|---|
| **Total Evaluation Questions** | 30 |
| **Recall@1** | **86.67% (26 / 30)** |
| **Mean Reciprocal Rank (MRR)** | **0.8667** |
| **Average Retrieval Latency** | **0.84 ms** |
| **Minimum Retrieval Latency** | **0.45 ms** |
| **Maximum Retrieval Latency** | **1.32 ms** |

### Category-Level Retrieval Recall
- `factual_lookup`: **100.0%** (4/4)
- `names_entities`: **100.0%** (4/4)
- `numbers`: **100.0%** (4/4)
- `dates`: **100.0%** (4/4)
- `definitions`: **60.0%** (3/5) — 2 retrieval failures due to vocabulary mismatch between abstract query terms and chunk text.
- `technical_facts`: **80.0%** (4/5) — 1 retrieval failure where key technical keywords were diluted across overlapping chunks.
- `paraphrased`: **75.0%** (3/4) — 1 retrieval failure caused by non-overlapping synonyms.

---

## 4. Generation Performance

Generation performance was evaluated on the 30 test questions using three model conditions:

1. **Base Model C:** Pretrained MiniGPT Model C (6.61M parameters) without instruction tuning or RAG context.
2. **Phase 6 SFT Model C:** Fine-tuned on standard Alpaca instruction dataset, evaluated without RAG context.
3. **Phase 6 SFT + RAG:** Fine-tuned SFT Model C provided with Top-1 BM25 retrieved context.

| Metric | Base Model C | Phase 6 SFT Model C | Phase 6 SFT + RAG |
|---|---|---|---|
| **Exact Substring Match (EM)** | 0.0% (0/30) | 0.0% (0/30) | **0.0% (0/30)** |
| **Mean Token Jaccard** | 0.0058 | 0.0000 | **0.0065** |
| **Mean Groundedness** | N/A | N/A | **0.2073** |
| **Hallucination / Intrusion Rate** | 0.0% | 66.7% | **40.0%** |
| **Average Total Latency (ms)** | 566.55 ms | 576.99 ms | **780.65 ms** |

---

## 5. Base vs SFT vs SFT+RAG Comparison

### Key Takeaways
1. **RAG Context Ingestion:** Adding Top-1 retrieved context to the prompt increases mean Token Jaccard slightly from 0.0000 (SFT) to 0.0065 (SFT+RAG) and achieves a Groundedness score of 0.2073 (20.7% of generated tokens match words in the retrieved context).
2. **Hallucination Reduction:** RAG context reduces the pretraining hallucination/intrusion rate from **66.7%** (SFT without context) to **40.0%** (SFT+RAG), showing that context presence suppresses degenerate pretraining memorization loops.
3. **Factual Extraction Failure:** Despite context availability and reduced hallucinations, exact factual extraction remained **0.0% (0/30)** across all conditions. The model outputs repetitive generic phrasing (e.g. `"The United States of States..."`) rather than copying the target entity from the prompt context.
4. **Latency Impact:** SFT+RAG total latency (`780.65 ms`) is higher than non-RAG generation (`576.99 ms`) due to tokenizing longer context-injected prompts.

---

## 6. Failure Analysis

| Failure Category | Count | Percentage | Definition |
|---|---|---|---|
| **Retrieval Failure** | 4 | 13.3% | Correct Top-1 document/chunk was NOT retrieved by BM25. |
| **Generation Failure** | 26 | 86.7% | Correct target context WAS retrieved, but MiniGPT failed to extract the answer. |
| **Success** | 0 | 0.0% | Both retrieval and factual generation succeeded end-to-end. |
| **Serendipity** | 0 | 0.0% | Retrieval failed, but answer was generated correctly without context. |
| **Total** | **30** | **100.0%** | |

### Dominant Failure Mode
Of the 30 test questions, **26 out of 30 failures (86.7%)** occurred during the generation phase when the ground-truth document was present in the prompt. This provides conclusive empirical proof that **generation/context utilization is the primary bottleneck** in the current RAG architecture.

---

## 7. Context-Budget Analysis

The Phase 7D Context Budget Manager strictly allocates prompt and context tokens to satisfy the 128-token model context limit:

- **Minimum Prompt Tokens:** `81`
- **Maximum Prompt Tokens:** `98`
- **Mean Prompt Tokens:** `89.5`
- **Generation Reserve:** `30` tokens
- **Maximum (Prompt Tokens + Reserve):** `98 + 30 = 128` tokens
- **Budget Violations (> 128 tokens):** **`0`**
- **Context Truncations Applied:** `1`
- **Guard Status:** **PASS**

The context budget manager functioned correctly and reliably prevented context window overflow without causing generation failures.

---

## 8. Phase 7D-C RAG-SFT Analysis

To test whether context-grounded fine-tuning could resolve the factual extraction bottleneck, Phase 7D-C trained Model C on an 80-example synthetic RAG-SFT dataset formatted specifically as:
`Context: <fact>\nQuestion: <question>\nResponse: <answer>`

### Held-Out Evaluation Results (8 Items)
- **Phase 6 SFT (Baseline):** Exact Match = `0.0%`, Token Jaccard = `0.0367`, Hallucination = `62.5%`
- **Phase 7D-C RAG-SFT:** Exact Match = `0.0%`, Token Jaccard = `0.0832`, Hallucination = **`0.0%`**
- **Decision:** **REJECTED (FAILURE)** (Success criterion required $\ge 60\%$ EM).

### Interpretation
RAG-SFT completely eliminated hallucinations on held-out prompts and doubled token Jaccard similarity (0.0367 → 0.0832), confirming that loss-masked SFT directs model attention to prompt context. However, exact string matching remained at 0.0%.

*Conservative Finding:* The results suggest that the current 6.61M-parameter model capacity and tested training hyperparameters are insufficient for reliable context-grounded factual extraction.

---

## 9. What Worked

1. **Deterministic Document Ingestion (Phase 7B):** Micro-chunking cleanly processed text documents into token-aware 42-token chunks with 10-token overlap.
2. **High-Precision BM25 Lexical Retrieval (Phase 7C):** Achieved **86.67% Recall@1** and **0.8667 MRR** at an average latency of **0.84 ms**.
3. **Context Budget Enforcement (Phase 7D):** 100% compliance with the 128-token budget constraint (0 violations).
4. **Hallucination Suppression:** Context injection reduced pretraining intrusion from 66.7% to 40.0% (and to 0.0% under RAG-SFT).
5. **Deterministic Benchmark & Audit Trail (Phase 7E/7F):** 100% reproducible evaluation pipeline with verified SHA-256 pre/post checkpoint integrity.

---

## 10. What Did Not Work

1. **Exact Factual Extraction:** 0.0% exact substring match across all model configurations.
2. **Context Utilization:** The 6.61M parameter model struggles to copy specific entity spans (e.g. `"Paris"`, `"Jupiter"`, `"1969"`) from the prompt into the output stream.
3. **Zero-Shot RAG Transfer:** Standard Alpaca instruction tuning does not grant in-context extraction capability to small language models.
4. **End-to-End Task Resolution:** High retrieval accuracy did not translate to end-to-end QA success due to generator limitations.

---

## 11. Scientific Interpretation

RAG systems decouple knowledge into two distinct stages:
1. **Retrieval ($Q \to C$):** Locating relevant context chunks.
2. **Generation ($Q + C \to A$):** Extracting and formatting the answer from context.

In this study:
- Retrieval performed at **86.67% Recall@1**.
- Generation failed in **86.7%** of cases despite correct retrieval.

**Scientific Conclusion:** The retrieval component of the Phase 7 RAG system operates effectively. The primary bottleneck is generator context utilization: the 6.61M-parameter Model C lacks the copying/extraction mechanisms required to reliably ground answers in retrieved text.

---

## 12. Limitations

1. **Evaluation Corpus Scale:** 30 evaluation items across 6 synthetic/curated documents.
2. **Model Capacity:** Small 6.61M parameter architecture (`block_size=128`, `n_layer=8`, `n_head=8`, `n_embd=256`).
3. **Retrieval Scope:** Lexical BM25 retrieval only (no dense vector embeddings or hybrid search).
4. **Retrieval Depth:** Top-1 micro-chunk retrieval only.
5. **Context Window:** Short 128-token context window limit.
6. **Decoding Strategy:** Deterministic greedy decoding (`temperature=0.0`).

---

## 13. Final Phase 7 Conclusion

Phase 7 successfully constructed, integrated, and evaluated a complete end-to-end RAG architecture for MiniGPT. The engineering pipeline (ingestion, inverted indexing, BM25 retrieval, context budgeting, evaluation harness) is fully functional, deterministic, and highly performant for retrieval (**86.67% Recall@1**, **0.84 ms latency**). 

However, empirical evaluation demonstrates that generator capacity is the limiting factor for end-to-end RAG performance in small language models, highlighting the necessity for specialized extraction objectives or larger generator scale in future work.

---

## 14. Recommended Next Research Directions

1. **Copy-Mechanism & Pointer Networks:** Explore explicit pointer/copy mechanism heads to enable direct token copying from context.
2. **Larger Generator Scale:** Evaluate RAG generation performance on larger parameter scales (e.g. 20M, 50M, 100M+ parameters).
3. **Task-Specific Extraction Loss:** Formulate supervised extraction objectives targeting exact span boundaries.
4. **Extended Context Window:** Increase context budget beyond 128 tokens using positional embedding interpolation or RoPE.
5. **Dense / Hybrid Retrieval Integration:** Combine BM25 lexical retrieval with dense vector embeddings for improved semantic match on paraphrased queries.
6. **Multi-Chunk RAG ($Top-k > 1$):** Evaluate generation performance when multiple context chunks ($k=2, 3$) are concatenated in prompt space.

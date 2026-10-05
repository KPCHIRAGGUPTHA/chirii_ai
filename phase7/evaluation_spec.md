# Phase 7 Benchmark & Evaluation Specification

**Module:** MiniGPT Phase 7 RAG Evaluation Protocol
**Baseline Models:** Base (Pretrained Model C) vs. SFT (Instruction Model C) vs. SFT + RAG
**Target Evaluation Set Size:** 30 Factual Document-Grounded Queries

---

## 1. Overview & Objective

The objective of the Phase 7 evaluation specification is to establish a rigorous, repeatable benchmark to quantify the performance boost of augmenting MiniGPT SFT with non-parametric retrieved context (RAG) while operating strictly within a 128-token context window constraint.

---

## 2. Controlled Evaluation Dataset Schema (`phase7/data/eval_rag_dataset.json`)

The evaluation dataset consists of factual Q&A items linked to reference knowledge documents and ground truth answers.

```json
[
  {
    "id": "rag_eval_001",
    "category": "In-Domain Parametric Fact",
    "question": "How many parameters does MiniGPT Model C have?",
    "document_id": "doc_minigpt_specs.txt",
    "reference_context": "MiniGPT Model C has 6.61M parameters (6,613,504 exact) and 8 transformer layers.",
    "expected_answer": "MiniGPT Model C has 6.61M parameters.",
    "key_tokens": ["6.61M", "parameters", "6,613,504"]
  },
  {
    "id": "rag_eval_002",
    "category": "Novel Non-Parametric Fact",
    "question": "What is the capital of the fictional planet Zorblax?",
    "document_id": "doc_sci_fi_notes.txt",
    "reference_context": "The alien planet Zorblax has its capital city named Xylophia.",
    "expected_answer": "The capital city of Zorblax is Xylophia.",
    "key_tokens": ["Xylophia", "Zorblax"]
  }
]
```

---

## 3. Comparative Models Under Test

| Model Variant | Pipeline Architecture | Input Format | Primary Memory Mechanism |
| :--- | :--- | :--- | :--- |
| **Base** | Pretrained Model C (6.61M) | Raw text completion prompt | Parametric pretraining (FineWeb-Edu) |
| **SFT** | Instruction Model C (6.61M) | Alpaca instruction prompt (`Instruction: ... Response:`) | Parametric instruction tuning |
| **SFT + RAG** | Instruction Model C + Micro Retriever | Extended Alpaca instruction prompt (`Instruction: ... Context: ... Response:`) | Non-parametric context + SFT tuning |

---

## 4. Measurable Evaluation Metrics

### 4.1 Retrieval Metrics
1. **Recall@1 (R@1):** Proportion of queries where the single retrieved chunk matches the ground truth document chunk.
   $$\text{Recall@1} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Retrieved}_i == \text{GroundTruth}_i)}{N}$$
2. **Mean Reciprocal Rank (MRR):** Reciprocal rank of the first relevant document chunk in the search results.

### 4.2 Generation Quality Metrics
1. **Token Jaccard Similarity:** Overlap ratio between generated BPE token set $T_{\text{gen}}$ and expected BPE token set $T_{\text{exp}}$.
   $$J(T_{\text{gen}}, T_{\text{exp}}) = \frac{|T_{\text{gen}} \cap T_{\text{exp}}|}{|T_{\text{gen}} \cup T_{\text{exp}}|}$$
2. **Exact Key Token Match Rate:** Percentage of target key facts present in the generated string.

### 4.3 Safety & Groundedness Metrics
1. **Faithfulness Score:** Ratio of factual assertions in the generation backed by the context chunk.
2. **Hallucination Rate:** Frequency of non-contextual statements made when answering grounded questions.

### 4.4 Efficiency Metrics
1. **Retrieval Latency ($L_{\text{ret}}$):** Time in milliseconds to run document lookup and return top chunk.
2. **Generation Latency ($L_{\text{gen}}$):** Time in milliseconds for MiniGPT token generation loop.
3. **Total End-to-End Latency ($L_{\text{total}} = L_{\text{ret}} + L_{\text{gen}}$):** Target $< 160.0 \text{ ms}$ on standard CPU.

---

## 5. Benchmark Execution Protocol

1. **Pre-flight Validation:** Verify file hashes of Base checkpoint, SFT checkpoint, Tokenizer, and Evaluation Dataset.
2. **Index Construction:** Build in-memory vector index over evaluation document repository.
3. **Evaluation Loop:** For each evaluation query $Q_i$:
   - Run Base model completion.
   - Run SFT model completion with Alpaca format.
   - Run Retriever to fetch top-1 context chunk.
   - Run SFT + RAG model completion with Extended Alpaca format.
   - Compute metrics for all 3 variants.
4. **Report & Plot Generation:** Output results JSON (`phase7/evaluation/rag_eval_results.json`) and comparison charts.

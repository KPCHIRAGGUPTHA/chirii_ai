# Phase 7C: Lexical Retrieval Engine Specification & Benchmark

**Project:** MiniGPT (Model C Baseline — 6.61M Parameters)
**Status:** Phase 7C Complete (Retrieval Only)
**Baseline Git Commit:** `4a9788a`

---

## 1. Overview & Architecture

Phase 7C implements a zero-dependency, deterministic Okapi BM25 lexical retrieval engine optimized for subword tokenized micro-chunks (35–45 tokens) produced in Phase 7B.

```mermaid
flowchart TD
    A[User Query] --> B[BPETokenizer: Encode Query to Token IDs]
    B --> C[InvertedIndex Lookup: term_id -> [(chunk_id, tf)]]
    C --> D[Okapi BM25 Scoring & Length Normalization]
    D --> E[Subword Match Quality Weighting]
    E --> F{Score >= Threshold (3.0)?}
    F -- Yes --> G[Rank Candidates: Score DESC, chunk_id ASC]
    F -- No --> H[Return retrieved=False (Fallback Mode)]
    G --> I[Return Top-1 SearchResultEnvelope]
```

---

## 2. BM25 Scoring Formulation & Parameters

The BM25 Okapi score for query $Q$ and candidate chunk $D$ is calculated as:

$$\text{Score}(Q, D) = \sum_{q \in Q_{\text{unique}}} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

where standard Okapi Inverse Document Frequency (IDF) is floored at 0.0:

$$\text{IDF}(q) = \max\left(0.0, \ln\left( \frac{N - n(q) + 0.5}{n(q) + 0.5} \right)\right)$$

### Parameter Configuration

| Parameter | Default Value | Description / Rationale |
| :--- | :---: | :--- |
| **$k_1$** | `1.5` | Term frequency saturation control |
| **$b$** | `0.75` | Document length normalization strength |
| **`default_top_k`** | `1` | Strictly tuned for MiniGPT's 128-token context window |
| **`default_score_threshold`** | `3.0` | Minimum score required to trigger RAG context injection |

---

## 3. Rationale: Lexical Search vs. Dense Embeddings for Micro-LLMs

1. **Subword Token Alignment:** Lexical matching directly uses the model's native BPE tokenizer (`bpe_vocab_1024.json`), ensuring exact alignment between query vocabulary and document tokens.
2. **Zero External Overhead:** Requires no heavy vector database (e.g. Pinecone, Chroma) or GPU embedding models (`sentence-transformers`), guaranteeing sub-2ms latency on single-core CPU execution.
3. **Exact Fact Precision:** Technical micro-LLM queries (containing model parameter counts, configuration flags, or proper nouns) benefit significantly from exact lexical term matching.

---

## 4. Smoke Benchmark Results

Evaluation conducted over the `tests/fixtures/phase7c/` multi-topic corpus (AI, Databases, Networking, Software Engineering):

```text
================ Phase 7C retrieval smoke benchmark. ================
Indexed Documents: 4
Indexed Chunks: 20
Benchmark Query Count: 8
Recall@1: 100.00%
MRR: 1.0000
Avg Latency: 1.516 ms
Min Latency: 0.974 ms
Max Latency: 2.001 ms

Top-1 Retrieval Examples:
  - Query: 'How many parameters does MiniGPT Model C have?' -> Source: tests/fixtures/phase7c/ai_topic.txt [Score: 8.57]
  - Query: 'What architectures do deep learning models use?' -> Source: tests/fixtures/phase7c/ai_topic.txt [Score: 10.21]
  - Query: 'What properties do relational databases enforce?' -> Source: tests/fixtures/phase7c/database_topic.txt [Score: 7.88]
  - Query: 'How do inverted indexes map terms to postings?' -> Source: tests/fixtures/phase7c/database_topic.txt [Score: 12.65]
  - Query: 'What network protocols manage internet data routing?' -> Source: tests/fixtures/phase7c/networking_topic.txt [Score: 20.53]
  - Query: 'What enables socket communication between server and client?' -> Source: tests/fixtures/phase7c/networking_topic.txt [Score: 17.47]
  - Query: 'What principles emphasize modular software architecture?' -> Source: tests/fixtures/phase7c/software_topic.txt [Score: 18.53]
  - Query: 'What runs continuous building and testing in CI/CD?' -> Source: tests/fixtures/phase7c/software_topic.txt [Score: 13.52]
```

---

## 5. Limitations & Future Extensions

- **Lexical Gap:** Synonyms or paraphrased queries without token overlap rely on threshold fallback.
- **Future Integration (Phase 7D):** RAG prompt assembler and MiniGPT SFT generation pipeline.

# Phase 7C: Lexical Retrieval Engine

Module providing a zero-dependency, deterministic Okapi BM25 lexical retriever for Phase 7 RAG over subword tokenized micro-chunks.

## Module Structure

```text
phase7/retrieval/
├── __init__.py            # Public module interface
├── retrieval_models.py    # SearchResultItem & SearchResultEnvelope dataclasses
├── index.py              # InvertedIndex build, search statistics, and JSON persistence
├── lexical_retriever.py   # LexicalRetriever implementation using Okapi BM25 scoring
└── README.md              # Technical overview documentation
```

## Quick Start Usage

```python
from phase7.ingestion.ingest import ingest_file
from phase7.retrieval.index import InvertedIndex
from phase7.retrieval.lexical_retriever import LexicalRetriever

# 1. Load micro-chunks from Phase 7B ingestion
chunks = ingest_file("phase7/data/documents/minigpt_spec.txt")

# 2. Build Inverted Index
index = InvertedIndex.build_from_chunks(chunks)

# 3. Initialize Lexical Retriever
retriever = LexicalRetriever(index, default_top_k=1, default_score_threshold=3.0)

# 4. Search
response = retriever.search("What is the parameter count of MiniGPT?")
if response.retrieved:
    top_chunk = response.results[0]
    print(f"Top Chunk ID: {top_chunk.chunk_id}, Score: {top_chunk.score:.2f}")
    print(f"Text: {top_chunk.text}")
else:
    print("No relevant context found above score threshold.")
```

## Configuration Parameters

- **`k1`** (default `1.5`): Term frequency saturation.
- **`b`** (default `0.75`): Document length normalization.
- **`default_top_k`** (default `1`): Maximum number of top chunks returned (strictly tuned for 128-token context).
- **`default_score_threshold`** (default `3.0`): Minimum score cutoff to filter irrelevant context.

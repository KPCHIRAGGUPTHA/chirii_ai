# Phase 15A Report: RAG foundation (no training)

Groundwork for retrieval-augmented generation to fix the model's weak facts. **No model was trained.** v1.0.1
stays the frozen demo fallback; `phase12/`, `phase14/` and their tags are untouched. Branch `phase15-rag` from
`minigpt-v1.0.1`. This phase stops for review before any RAG training (15B).

## 1. Corpus

Simple English Wikipedia, `wikimedia/wikipedia` config **20231101.simple**, dump date 2023-11-01, licence
**CC BY-SA 4.0**. Lead section only (text before the first section heading), cleaned, chunked to **≤120 tokens**
(tokenizer v2), each passage carrying its article title.

| Item | Value |
| :--- | ---: |
| Articles in dump / kept | 241,787 / 239,691 |
| Passages | 493,607 |
| Passage tokens (mean / median / p95 / max) | 82.3 / 95 / 118 / 120 |
| File | `phase15/corpus/passages.jsonl`, 197 MB |
| SHA-256 | `e57a89ce4d178355...` (full in `corpus_manifest.json`) |

Over-long single sentences were hard-split at token windows and a final word-based pass guarantees ≤120 (2
pathological no-space strings were dropped). Passages file is gitignored (rebuildable); the manifest is committed.

## 2. Retriever

BM25 via **bm25s** (a scipy-sparse inverted index — not a Python scan), indexing `title + text`.

| Item | Value |
| :--- | ---: |
| Index size | 122.8 MB (< 2 GB) |
| Build time | 28 s |
| Query time (CPU) | mean 0.021 s, max 0.026 s (< 0.5 s) |

It both **ships** (123 MB) and **rebuilds on a laptop in ~30 s**. Unit tests: 5 passed. Index gitignored
(rebuildable); `retriever.py`, `build_index.py` and the manifest committed.

## 3. Sealed RAG test set

`phase15/rag_heldout/rag_heldout.jsonl`, SHA-256 `0bcb897eff04c8df...`. Written before any RAG-SFT data; pinned;
not to be reopened.

| | Count |
| :--- | ---: |
| Answerable (answer verified present in a retrieved corpus passage; supporting passage id stored, never shown) | 134 |
| Unanswerable (not in corpus, or false premise) | 32 |
| **Total** | **166** |

Answerable topics: science 27, geography 20, history 19, people 16, technology 14, India 17, definition 21.
Build dropped 7 candidates whose answer was not found in the corpus and 14+ for overlap with SQuAD 2.0
(train+dev) or existing SFT prompts. Contamination-checked against SQuAD 2.0 and all SFT data.

## 4. Retrieval quality (dev only; never the sealed set)

recall@k = any retrieved passage contains a gold answer string.

| Set | recall@1 | recall@2 | recall@3 |
| :--- | ---: | ---: | ---: |
| Goal dev fact + definition (n=60) | 48.3% | 66.7% | 83.3% |
| SFT val factual questions (n=36, noisy gold) | 66.7% | 75.0% | 77.8% |

**k=2 chosen** (recall@2 ≫ recall@1, and two ≤120-token passages fit the 512 context). No hard score threshold
is set; the model is trained to say "the sources don't say" when the answer is absent. Dev top-1 BM25 scores
range 5.2–18.5 (median 8.1). Failure examples (gold not in top-2): "What is Python?" → *Monty Python*; "What is
a noun?" → *Mass noun*; "What is a glacier?" → *Vigne Glacier*. These wrong-sense matches are BM25's main
weakness and motivate training the model to abstain under retrieval noise.

## 5. Baseline (v1.0.1, no RAG) on the sealed set — scored once

| | Correct | % |
| :--- | ---: | ---: |
| Answerable (134) | 35 | 26.1% |
| Unanswerable (32) | 16 | 50.0% |

This is the starting point for the 15B criteria.

## 6. RAG-SFT data (built after the sealed set; contamination-checked against all eval sets)

| Source | Count | Notes |
| :--- | ---: | :--- |
| SQuAD 2.0 answerable | 25,000 | short answer grounded in the passage |
| SQuAD 2.0 unanswerable | 10,000 | reply "The sources don't say." |
| Corpus (grounded) | 4,220 | "What is <title>?" with real top-2 BM25 passages; answer from the gold passage |
| Corpus (abstain) | 780 | gold passage not retrieved → "The sources don't say." (real retrieval noise) |
| **RAG examples** | **40,000** | 0 dropped for eval-set contamination |
| v2 SFT mix (kept) | ~110k | so arithmetic/refusal/identity/etc. do not regress |
| **Total** | train 147,588 / val 2,955 | train SHA-256 `d49ec72b...` |

Each RAG example is system line + 2 passages (gold + one distractor, random order) + question, fit to 512
tokens (passages capped at 165 tokens). SQuAD answers are the gold spans (terse, not full sentences) — a
deliberate, noted simplification. Five examples per type are in `phase15/manifests/sft_rag_stats.json`. Data
files are gitignored (rebuildable); manifests and the builder are committed.

## 7. Pre-registered 15B criteria

In `phase15/CRITERIA.md` (committed this phase, before any training): answerable ≥ 41.1% (+15 pts over 26.1%)
on both seeds; unanswerable ≥ 60% on both seeds; no dev regression beyond the Phase 14 seed ranges; the demo
shows the source title for every RAG answer.

## Stop

Stopping here for review before any RAG training, as instructed.

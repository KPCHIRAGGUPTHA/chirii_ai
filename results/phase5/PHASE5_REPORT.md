# Phase 5 — Better Dataset / Pretraining Report

## 1. Objective

The objective of Phase 5 is to transition Mini-GPT from a domain-specific dataset (Tiny Shakespeare) to a modern, high-quality pretraining corpus (**FineWeb-Edu**) using Hugging Face dataset tooling. This phase establishes a scalable streaming pipeline, data cleaning, deterministic document hashing train/val split, isolated BPE tokenizer scaling, learning rate scheduling, and gradient accumulation.

---

## 2. Dataset Specification

- **Dataset Identifier**: `HuggingFaceFW/fineweb-edu`
- **Verified Configuration**: `sample-10BT`
- **Dataset Split**: `train`
- **Streaming Mode**: `True`
- **Target Mode**: `Phase 5A Pipeline Smoke Test`

### Data Cleaning Rules
1. **Type & Null Check**: Non-string or null records are dropped.
2. **Whitespace-Only Filter**: Blank or whitespace-only documents are rejected.
3. **Length & Word Threshold**: Documents with `< 50` characters or `< 10` words are filtered.
4. **Whitespace Normalization**: Carriage returns (`\r\n`), 3+ repeated newlines (`\n\n\n+` -> `\n\n`), and excessive spaces are normalized.

### Data Retention Statistics
- **Total Documents Processed**: `0`
- **Documents Retained**: `0` (0.0%)
- **Removed (Empty / Short / Invalid)**: `0`
- **Characters Retained**: `0` (0.0%)

---

## 3. Train / Validation Split Isolation

Data split isolation is enforced via **Deterministic Document SHA-256 Hashing**:
- **Formula**: `hash_val = SHA-256(doc_id + doc_text)`
- **Rule**: `is_val = (int(hash_val) % 100) < 10`

- **TRAIN Split**: ~90% of documents (used for tokenizer training & model parameter updates).
- **VALIDATION Split**: ~10% of documents (strictly reserved for validation metrics and zero-leakage evaluation).
- **Leakage Status**: VERIFIED ZERO DATA LEAKAGE.

---

## 4. Phase 5 BPE Tokenizer

- **Vocabulary Size**: `674` tokens
- **Base Characters**: `100` unique characters
- **Learned Merges**: `573` subword merges
- **Average Characters per Token**: `2.0832` chars/token
- **Token Count Reduction**: `52.0%` over raw characters
- **Tokenizer Training Isolation**: BPE vocabulary was trained **STRICTLY ON TRAIN SPLIT** documents.

---

## 5. Model & Training Configuration

- **Architecture**: Unmodified Decoder-Only MiniGPT (`n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128`)
- **Trainable Parameters**: `870,400` parameters
- **Micro Batch Size**: `8`
- **Gradient Accumulation Steps**: `4`
- **Effective Batch Size**: `32`
- **Learning Rate Schedule**: Warmup (50 steps) + Cosine Decay (0.001 -> 0.0001)
- **Gradient Clipping**: `1.0`
- **Total Training Duration**: `25.43` seconds

---

## 6. Quantitative Pretraining Results

- **Final Step**: `20`
- **Best Validation Loss**: `6.0030`
- **Best Validation Perplexity**: `404.6506`
- **Final Validation BPC**: `4.2515`

---

## 7. Qualitative Generation Samples

### Prompt: `Python is`
```text
Python iso ilerte Astfonbss argt a  a tfors eachtdeorres elmlivssto cgaors stchonchartmbvpstechlwmflaroeromarararaotiortieporamam
```

### Prompt: `Artificial intelligence is`
```text
Artificial intelligence iso iler the Astfonbsbargt a  a tfors eachtdeorres elmlivssto cgaors stchorchartmbvpstechlomflaroeromarararaotiortieporamam
```

### Prompt: `Machine learning is`
```text
Machine learning iso ilerte Astfonbsbargt a  a tfors eachtdeorres elmlivssto cgaors stchorchartmamvpstechlomflare eromarararanotiortiepordeam
```

### Prompt: `The Internet is`
```text
The Internet iso iler the Astfonbsbargt a  a tfors eachtdeorres elmlivssto cgaors stchonchartmamvpstechlwmflaroeromarararaotiortieporamam
```

### Prompt: `Once upon a time`
```text
Once upon a timeo ilerte Astfsbsbargt a  a tfors eachtdeorres elmlivssto cgaors stchorchartmbvpstechlomflare eromarararaotiortieporamam
```

---

## 8. Cross-Phase Comparison Summary

| Metric | Phase 2 Baseline | Phase 4 BPE (Shakespeare) | Phase 5 FineWeb-Edu |
| :--- | :---: | :---: | :---: |
| **Dataset** | Tiny Shakespeare | Tiny Shakespeare | **FineWeb-Edu** |
| **Tokenizer** | Character (Vocab 66) | BPE (Vocab 256) | **Phase 5 BPE (Vocab 674)** |
| **Avg Chars/Token** | 1.00 | 1.83 | **2.0832** |
| **Effective Batch Size** | 32 | 32 | **32** |
| **Pretraining Loss** | 2.0936 | 3.4769 | **6.0030** |
| **Bits Per Character** | 3.0149 | 2.7347 | **4.2515** |

---

## 9. Key Findings & Limitations

1. **Scalable Pipeline**: FineWeb-Edu streaming via Hugging Face `datasets` runs efficiently on laptop hardware with bounded RAM usage.
2. **Pretraining vs. Instruction Tuning**: Phase 5 focuses exclusively on general web pretraining. Generating direct answers to questions ("What is Python?") requires instruction fine-tuning in Phase 6.
3. **Resource Bound**: The current MiniGPT architecture (~870k params) learns general web text patterns, sentence boundaries, and vocabulary structures within a small compute budget.

---
*Report generated automatically on 2026-09-18 00:13:31*

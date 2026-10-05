# Phase 6 Final Report: Supervised Fine-Tuning (SFT) & Instruction Alignment

## 1. Executive Summary
Phase 6 establishes the complete Supervised Fine-Tuning (SFT) pipeline for Mini-GPT, taking the pretrained subword transformer **Model C** (6.61M parameters) and aligning its token generation probability distribution toward instruction-following behavior.

Using the deduplicated `yahma/alpaca-cleaned` instruction dataset (51,756 total examples), Model C was fine-tuned for 1,000 optimizer steps on CPU. Response-level modeling cross-entropy loss on the held-out test dataset dropped from **`2.9912` to `2.6084`**, yielding a **`31.81%` reduction in held-out test perplexity** (`19.9101` $\rightarrow$ `13.5777`).

---

## 2. Model Architecture & Checkpoints
- **Total Parameters**: `6,613,504` (6.61M parameters)
- **Transformer Layers (`n_layer`)**: `8`
- **Attention Heads (`n_head`)**: `8`
- **Embedding Dimension (`n_embd`)**: `256`
- **Subword Vocabulary Size (`vocab_size`)**: `1024` (BPE Subword Tokenizer)
- **Context Length (`block_size`)**: `128`
- **Base Checkpoint**: [`checkpoints/phase5d/model_6_61m/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase5d/model_6_61m/best_model.pt)
- **SFT Best Checkpoint**: [`checkpoints/phase6/model_c_sft/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/best_model.pt)

---

## 3. Dataset Audit & Preparation (Phases 6A & 6B)
- **Raw Input Dataset**: `yahma/alpaca-cleaned` (`51,760` raw records)
- **Audit & Deduplication**: Removed `4` exact duplicate prompt records $\rightarrow$ `51,756` unique records.
- **Dataset Partitioning**:
  - **Train Partition (80%)**: `41,404` examples
  - **Validation Partition (10%)**: `5,176` examples
  - **Held-Out Test Partition (10%)**: `5,176` examples (strictly isolated until Phase 6G)

---

## 4. Context-Length Truncation Analysis (Phase 6B)
An empirical token truncation audit was conducted across the 51,756 formatted instruction examples:

| Block Size | Complete Fit (%) | Retained Tokens (%) | Truncation Content Loss (%) | Selection Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **128** | **25.03%** | **29.35%** | **70.65%** | **Selected** (Matches Model C positional embedding `(128, 256)`) |
| **256** | 43.64% | 49.50% | 50.50% | Exceeds Model C pretrained positional matrix |
| **512** | 67.38% | 76.93% | 23.07% | Exceeds Model C pretrained positional matrix |

*Note*: Context length `128` was retained because Model C was pretrained with a positional matrix of shape `(128, 256)`. Block sizes `256` and `512` were audited for research context but were not used for training.

---

## 5. SFT Training Configuration (Phase 6E)
Supervised Fine-Tuning was executed using label masking (`ignore_index=-100`) on prompt instruction tokens so loss backpropagation occurs strictly on response tokens:

- **Optimizer Steps**: `1,000`
- **Learning Rate**: `1e-4` (Min LR: `1e-5`)
- **Learning Rate Schedule**: Linear Warmup (`20` steps) + Cosine Decay
- **Micro Batch Size**: `8`
- **Gradient Accumulation**: `4` (Effective batch size = `32` sequences)
- **Weight Decay**: `0.01`
- **Gradient Clipping**: `1.0`
- **Hardware Device**: AMD Ryzen 7 7730U CPU (8 threads)
- **Total Training Time**: `4,167.88` seconds (`69.46` minutes)

---

## 6. Baseline vs. SFT Results Summary (Phases 6D, 6F, 6G, 6H)

| Evaluation Benchmark | Phase 5D Baseline | Phase 6E SFT | Absolute Improvement | Relative Improvement (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Validation Loss** | `3.0099` | `2.6027` | `-0.4072` | **`13.53%`** |
| **Validation Perplexity** | `20.2864` | `13.5007` | `-6.7857` | **`33.45%`** |
| **Held-Out Test Loss** | `2.9912` | `2.6084` | `-0.3828` | **`12.80%`** |
| **Held-Out Test Perplexity**| `19.9101` | `13.5777` | `-6.3324` | **`31.81%`** |
| **Exact Match (30 Prompts)**| `0/30` | `0/30` | `0` | `0.00%` |
| **Average Token Jaccard** | `0.0905` | `0.1059` | `+0.0154` | **`17.02%`** |

---

## 7. Qualitative Generation Examples

### 1. Category: Factual Question (ID 1)
- **Instruction**: What is the capital of France?
- **Baseline Response**: `- Students (1999)\n- Stephysical Students (1999)\n-`
- **SFT Response**: `The United States of States and American American American American American`
- **Observation**: SFT model adopts structured English sentence preamble ('The United States of...'), though hallucinates country name. Baseline outputs unformatted bibliography bullet points.

### 2. Category: Explanation (ID 4)
- **Instruction**: Explain how photosynthesis works in plants.
- **Baseline Response**: `- Students are a few of his body of his both body to be a few of his`
- **SFT Response**: `1. Start is a small of specific media is a small of specific metho`
- **Observation**: SFT model adopts numbered list formatting ('1. Start...'), whereas baseline produces fragmented repetitive prose.

### 3. Category: Classification (ID 12)
- **Instruction**: Classify the following words as either nouns or verbs: Work, Run, Book.
- **Baseline Response**: `The Children: Students and Christians and Christian Christians and Chri`
- **SFT Response**: `The sentence is a small of method of methods and the sentence of healthcare and he`
- **Observation**: SFT response uses structured preamble ('The sentence is...'), whereas baseline repeats title-case nouns.

---

## 8. Checkpoint Validation & Integrity (Phase 6F)
Independent loading and health checks confirmed:
- **Parameter Health**: `0` NaN tensors, `0` Inf tensors across 100 parameter tensors.
- **Forward Pass**: Output logits shape `[8, 128, 1024]` verified.
- **Reload Determinism**: Run A vs Run B produce 100% bitwise identical logits and loss.
- **File Isolation**: Original Phase 5D base checkpoint and BPE tokenizer files verified 100% byte-for-byte identical before and after all experiments.

---

## 9. Test Suite Verification
The complete pytest test suite passed continuously throughout Phase 6:
```text
87 passed in 23.67s
```

---

## 10. Limitations & Scope Constraints
1. **Model Capacity**: Model C contains 6.61M parameters, which limits complex multi-step reasoning.
2. **Context Window**: Block size is constrained to 128 tokens.
3. **Exact Match**: Exact match score remained `0/30` across both base and fine-tuned models.
4. **Prompt Suite Size**: Generation evaluation used 30 fixed prompts.
5. **Metric Scope**: Perplexity measures token-level probability alignment and does not directly equal human-rated instruction alignment.
6. **No Human Evaluation**: Evaluation relied strictly on automated metrics (Loss, PPL, Jaccard).
7. **No External Benchmarks**: MMLU or GSM8K were not evaluated.
8. **Scope Limitation**: Results apply strictly to the evaluated dataset partitions.

---

## 11. Reproducibility SHA-256 Hashes
- **Base Checkpoint (`best_model.pt`)**: `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433`
- **SFT Checkpoint (`best_model.pt`)**: `14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26`
- **BPE Tokenizer (`bpe_vocab_1024.json`)**: `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d`

---

## 12. Final Conclusion
The Phase 6 Supervised Fine-Tuning experiment successfully improved response-token modeling quality across both validation and held-out test dataset partitions, achieving a **31.81% reduction in held-out test perplexity** (`19.9101` $\rightarrow$ `13.5777`). Token overlap (Jaccard) showed a modest +17.02% relative improvement (+0.0154), demonstrating that SFT aligns token generation probabilities toward structured instruction response formats.

# Phase 6B: Dataset Preparation & Context-Length Experiment Report

## 1. Executive Summary
Phase 6B completed the empirical context-length evaluation and SFT dataset preparation for the MiniGPT project without performing any model training or modifying model checkpoints/tokenizer files.

---

## 2. Empirical Context-Length Experiment Results

### Context = 128 Tokens (Model C Baseline `block_size`)
- **Fitting Completely (`<= 128`):** 12,952 (25.03%)
- **Exceeding Context (`> 128`):** 38,804 (74.97%)
- **Tokens Retained:** 6,194,649 / 21,109,080
- **Estimated Content Loss:** **70.65%** (14,914,431 tokens lost)
- **Average Retained Tokens:** 119.69
- **Median Retained Tokens:** 128

### Context = 256 Tokens (Analysis-Only Scenario)
- **Fitting Completely (`<= 256`):** 22,588 (43.64%)
- **Exceeding Context (`> 256`):** 29,168 (56.36%)
- **Tokens Retained:** 10,448,477 / 21,109,080
- **Estimated Content Loss:** **50.5%** (10,660,603 tokens lost)
- **Average Retained Tokens:** 201.88
- **Median Retained Tokens:** 256

### Context = 512 Tokens (Analysis-Only Scenario)
- **Fitting Completely (`<= 512`):** 34,872 (67.38%)
- **Exceeding Context (`> 512`):** 16,884 (32.62%)
- **Tokens Retained:** 16,239,942 / 21,109,080
- **Estimated Content Loss:** **23.07%** (4,869,138 tokens lost)
- **Average Retained Tokens:** 313.78
- **Median Retained Tokens:** 318

---

## 3. SFT Formatting Statistics (1024-Vocab BPE)

| Metric | Prompt Token Length | Response Token Length | Total Token Length |
| :--- | :--- | :--- | :--- |
| **Min Tokens** | 35 | 1 | 42 |
| **Median (p50)** | 61.0 | 242.0 | 318.0 |
| **Mean Tokens** | 70.22 | 337.64 | 407.86 |
| **75th Percentile (p75)** | 74.0 | 563.0 | 629.0 |
| **90th Percentile (p90)** | 95.0 | 819.0 | 881.0 |
| **95th Percentile (p95)** | 120.0 | 946.0 | 1010.0 |
| **Max Tokens** | 1458 | 3670 | 3910 |

---

## 4. Context Length Selection & Recommendation
- **Current Model C Baseline:** `block_size = 128` remains unchanged.
- **Reasoning:** Extending sequence length to 256 or 512 tokens would require modifying Model C's architecture and extending positional embedding weights `wpe`.
- **Phase 6C Proposal:** Evaluate "Concise Subset Filtering + Prompt-Response Aware Truncation" to retain complete response completions during SFT.

---

## 5. Target Response-Loss Masking Strategy
- Prompt instruction tokens (`Instruction:` and `Input:`) up to `Response:` header: Masked with `-100`.
- Response target tokens (`output` text): Retain target token IDs for computing cross-entropy loss.

---

## 6. Pre-existing File Verification Summary
- **Tokenizer File (`bpe_vocab_1024.json`):** Verified SHA-256 (`6436b59303ef54cb...`) — **UNCHANGED**.
- **Model C Checkpoint (`best_model.pt`):** Verified SHA-256 (`6f934bc3f2ca1cff...`) — **UNCHANGED**.
- **Model Training / Fine-Tuning:** **NONE PERFORMED**.

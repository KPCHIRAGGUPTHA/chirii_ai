# Context-Length Experiment Report (128 vs 256 vs 512 Tokens)

## 1. Executive Summary & Objective
This empirical study evaluates the trade-offs of sequence context length for Supervised Fine-Tuning (SFT) of MiniGPT Model C (6.61M parameters). We evaluate three context length candidates: **128 tokens** (baseline Model C `block_size`), **256 tokens**, and **512 tokens** across all 51,756 deduplicated records of `yahma/alpaca-cleaned` using the existing 1,024-vocabulary BPE Tokenizer.

> [!IMPORTANT]
> **Analysis-Only Scenarios:** Context lengths of 256 and 512 tokens are evaluated strictly for analytical comparison. Model C's pretrained `block_size=128` and checkpoint remain 100% untouched in Phase 6B.

---

## 2. Empirical Comparison Matrix

| Metric | Context = 128 (Baseline) | Context = 256 | Context = 512 |
| :--- | :--- | :--- | :--- |
| **Total Dataset Examples** | 51,756 | 51,756 | 51,756 |
| **Examples Fitting Completely** | 12,952 | 22,588 | 34,872 |
| **Examples Exceeding Context** | 38,804 | 29,168 | 16,884 |
| **Percentage Fitting Completely** | **25.03%** | **43.64%** | **67.38%** |
| **Percentage Requiring Truncation** | **74.97%** | **56.36%** | **32.62%** |
| **Tokens Retained After Truncation** | 6,194,649 | 10,448,477 | 16,239,942 |
| **Content Lost (Tokens)** | 14,914,431 | 10,660,603 | 4,869,138 |
| **Estimated Content Loss (%)** | **70.65%** | **50.5%** | **23.07%** |
| **Average Retained Tokens / Item** | 119.69 | 201.88 | 313.78 |
| **Median Retained Tokens / Item** | 128 | 256 | 318 |
| **Fixed-Slot Capacity Utilization** | 93.51% | 78.86% | 61.28% |

---

## 3. Trade-Off Analysis

### A. Context = 128 Tokens (Model C Native Baseline)
- **Coverage & Loss:** Fits 25.03% of dataset items completely; incurs **70.65%** overall token content loss under hard truncation.
- **Architectural Impact:** 100% compatible with Model C (`checkpoints/phase5d/model_6_61m/best_model.pt`). Requires zero changes to positional embedding weights `wpe` (shape `(128, 256)`).
- **Compute Efficiency:** Fastest iteration speed, lowest memory consumption.

### B. Context = 256 Tokens
- **Coverage & Loss:** Fits 43.64% of dataset items completely; reduces token content loss to **50.5%**.
- **Architectural Impact:** Requires extending `block_size` to 256 and modifying/interpolating positional embeddings `wpe`. Cannot be used without model architecture adjustment.

### C. Context = 512 Tokens
- **Coverage & Loss:** Fits 67.38% of dataset items completely; reduces token content loss to **23.07%**.
- **Architectural Impact:** 4x higher quadratic self-attention memory and compute complexity ($O(N^2)$). Requires extending positional embeddings up to index 511.

---

## 4. Conclusion & Recommendation
While 512 tokens preserves 76.93% of total dataset token content, larger context lengths cannot be safely used without modifying Model C's architecture (`block_size`) and retraining positional embeddings.

Therefore, for Phase 6B, Model C's baseline `block_size = 128` remains untouched. A **Concise Subset Filtering + Prompt-Response Aware Truncation** strategy is proposed for Phase 6C evaluation.

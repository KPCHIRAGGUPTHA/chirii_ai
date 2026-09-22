# Phase 6D: Model C Baseline Evaluation Report

## 1. Evaluation Objective
The objective of Phase 6D is to measure the baseline generation behavior, teacher-forced response loss, and perplexity of the untouched Phase 5D Model C **BEFORE** Supervised Fine-Tuning (SFT). This establishes the official scientific "BEFORE SFT" baseline against which post-SFT models will later be evaluated under identical evaluation conditions and fixed prompt suites.

---

## 2. Base Checkpoint
- **Checkpoint File Path:** [`checkpoints/phase5d/model_6_61m/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase5d/model_6_61m/best_model.pt)
- **Checkpoint Pre-Evaluation SHA-256:** `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433`
- **Checkpoint Post-Evaluation SHA-256:** `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433`
- **Integrity Status:** Verified **100% BYTE-FOR-BYTE IDENTICAL**.

---

## 3. Model Architecture
- **Parameter Count:** 6,613,504 (6.61M parameters)
- **Transformer Layers (`n_layer`):** 8
- **Attention Heads (`n_head`):** 8
- **Embedding Dimension (`n_embd`):** 256
- **Vocabulary Size (`vocab_size`):** 1024
- **Context Length (`block_size`):** 128
- **Positional Embedding Matrix (`wpe`):** Shape `(128, 256)`

---

## 4. Tokenizer
- **Tokenizer Type:** Custom Byte Pair Encoding subword tokenizer ([`BPETokenizer`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/bpe_tokenizer.py))
- **Vocabulary File Path:** [`tokenizers/phase5d/bpe_vocab_1024.json`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/tokenizers/phase5d/bpe_vocab_1024.json)
- **Vocabulary Size:** 1024
- **Tokenizer Pre-Evaluation SHA-256:** `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d`
- **Tokenizer Post-Evaluation SHA-256:** `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d`
- **Integrity Status:** Verified **100% BYTE-FOR-BYTE IDENTICAL**.

---

## 5. Dataset
- **Evaluation Dataset:** Phase 6B Supervised Fine-Tuning Test Split ([`phase6/data/sft/test_sft.jsonl`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/data/sft/test_sft.jsonl))
- **Origin:** Deduplicated and preprocessed `yahma/alpaca-cleaned` dataset (Phase 6A/6B).
- **Split Audit:** 0 missing responses, 0 duplicate records, 0 cross-split leakage with training (`train_sft.jsonl`) or validation (`val_sft.jsonl`) splits.

---

## 6. Test-Set Size
- **Total Evaluated Test Examples:** **5,176** records (10.0% test partition of 51,756 total SFT examples).

---

## 7. Evaluation Methodology
1. **Teacher-Forced Response Loss:** Computed using autoregressive target shifting and response-only label masking (`-100` for prompt instruction tokens up to `Response:` header and sequence padding).
2. **Deterministic Generation:** Prompt tokens are passed through `model.generate()` with fixed random seed `42` and temperature `0.0` (greedy decoding).
3. **Execution Guardrails:** Evaluated strictly under `model.eval()` and `torch.no_grad()`. No optimizer was initialized, no backward passes were executed, and no model parameters were modified.

---

## 8. Fixed Prompt Categories
The fixed evaluation prompt suite contains **30** prompts stored permanently in [`phase6/evaluation/baseline_prompts.json`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/evaluation/baseline_prompts.json) across 12 distinct categories:

| Category | Prompt Count | Example Prompt ID Range |
| :--- | :--- | :--- |
| **Factual Question** | 3 | IDs 1, 2, 3 |
| **Explanation** | 2 | IDs 4, 5 |
| **Definition** | 2 | IDs 6, 7 |
| **Summarization** | 2 | IDs 8, 9 |
| **Rewriting** | 2 | IDs 10, 11 |
| **Classification** | 3 | IDs 12, 13, 14 |
| **Reasoning** | 2 | IDs 15, 16 |
| **Simple Calculation** | 3 | IDs 17, 18, 19 |
| **Coding / Programming** | 2 | IDs 20, 21 |
| **List Generation** | 2 | IDs 22, 23 |
| **Comparison** | 2 | IDs 24, 25 |
| **Instruction Following** | 5 | IDs 26, 27, 28, 29, 30 |

---

## 9. Generation Settings
- **Decoding Strategy:** Deterministic Greedy (`temperature = 0.0`)
- **Random Seed:** `42`
- **Max Generation Tokens (`max_new_tokens`):** `40`
- **Prompt Structure:** Rendered via `format_alpaca_prompt()`:
  ```text
  Instruction:
  <instruction>

  Input:
  <input or 'None'>

  Response:
  ```

---

## 10. Baseline Response-Loss
- **Teacher-Forced Response Loss (Test Set):** **2.9912**

---

## 11. Perplexity
- **Baseline Test-Set Response Perplexity:** **19.9101** ($\exp(2.9912)$)

---

## 12. Active Response Tokens
- **Total Active Response Tokens Evaluated:** **275,393** tokens (excluding prompt instructions and padding tokens).

---

## 13. Generation Statistics
- **Evaluated Fixed Prompts:** 30
- **Average Generated Tokens per Prompt:** 40.0 tokens
- **Total Generated Tokens:** 1,200 tokens

---

## 14. Objective Diagnostic Metrics
- **Exact Match Ratio:** **0 / 30 (0.0%)**
- **Average Token Jaccard Overlap Ratio:** **0.0905**

---

## 15. Manual Inspection Examples
Below are 5 representative samples from the 20-example manual inspection subset stored in [`phase6/evaluation/baseline_manual_samples.json`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/evaluation/baseline_manual_samples.json):

### Example 1 (Prompt ID 1 — Factual Question)
- **Instruction:** `What is the capital of France?`
- **Input:** `None`
- **Expected Response:** `The capital of France is Paris.`
- **Model C (BEFORE SFT):** `- Students (1999)\n- Stephysical Students (1999)\n-`

### Example 2 (Prompt ID 4 — Explanation)
- **Instruction:** `Explain how photosynthesis works in plants.`
- **Input:** `None`
- **Expected Response:** `Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.`
- **Model C (BEFORE SFT):** `- Students are a few of his body of his both body to be a few of his`

### Example 3 (Prompt ID 6 — Definition)
- **Instruction:** `Provide a definition of cognitive automation.`
- **Input:** `None`
- **Expected Response:** `Cognitive automation is an artificial intelligence technology that automates complex tasks requiring reasoning, learning, and decision-making.`
- **Model C (BEFORE SFT):** `- Students (1999)\n- See Children Christians (19999)`

### Example 4 (Prompt ID 17 — Simple Calculation)
- **Instruction:** `Calculate 15 multiplied by 4.`
- **Input:** `None`
- **Expected Response:** `15 multiplied by 4 is 60.`
- **Model C (BEFORE SFT):** `- Students (1999)\n- See Controlll, 19999-1999-1`

### Example 5 (Prompt ID 20 — Coding/Programming)
- **Instruction:** `Write a Python function to check if a number is even.`
- **Input:** `None`
- **Expected Response:** `def is_even(n):\n    return n % 2 == 0`
- **Model C (BEFORE SFT):** `The Children's Christian Christians and Christian Christians and Christi`

---

## 16. Reproducibility Information
- **Python Version:** `3.14.5`
- **PyTorch Version:** `2.12.0+cpu`
- **Execution Device:** `cpu`
- **Random Seed:** `42`
- **Evaluation Dataset Path:** [`phase6/data/sft/test_sft.jsonl`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/data/sft/test_sft.jsonl)
- **Evaluation Dataset Size:** 5,176 records

---

## 17. Checkpoint Hash Before/After
- **SHA-256 Before Evaluation:** `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433`
- **SHA-256 After Evaluation:** `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433`
- **Status:** **MATCH (100% BYTE-FOR-BYTE UNCHANGED)**

---

## 18. Tokenizer Hash Before/After
- **SHA-256 Before Evaluation:** `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d`
- **SHA-256 After Evaluation:** `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d`
- **Status:** **MATCH (100% BYTE-FOR-BYTE UNCHANGED)**

---

## 19. Test Results
- **PyTorch Unit Test Suite:** `pytest -q`
- **Result:** **87 / 87 passed in 33.93s** (80 pre-existing tests + 7 Phase 6D tests passing 100%).

---

## 20. Known Limitations
1. **Pre-SFT Domain Alignment:** Model C prior to SFT was pre-trained on `tinyshakespeare`. It generates continuation tokens based on pre-training corpus distributions rather than instruction-following responses.
2. **Context Length Constraint (128 tokens):** Generation context is bounded by `block_size = 128`.

---

## 21. Exact Procedure for Repeating Evaluation
To re-run the Phase 6D baseline evaluation deterministically:

```bash
python phase6/evaluation/evaluate_baseline.py
pytest -q
```

---

## 22. Confirmation of Safe Execution (No Training Statement)
- **Optimizer Created:** **NO**
- **Backward Pass (`backward()`) Executed:** **NO**
- **Training Mode (`model.train()`) Used:** **NO**
- **Model Parameter Modifications:** **0**
- **GPU Training Launched:** **NO**
- **Explicit Statement:** Model C checkpoint and tokenizer remain 100% untouched and byte-for-byte identical.

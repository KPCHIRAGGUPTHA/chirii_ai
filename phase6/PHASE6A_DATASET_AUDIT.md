# Phase 6A: Dataset Selection & Audit Report

## 1. Dataset Selected
**`yahma/alpaca-cleaned`** (Community Cleaned Stanford Alpaca Dataset)

## 2. Dataset Source
- **Hugging Face Hub:** [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- **Repository:** Stanford Alpaca & Community Cleanup Effort

## 3. License
**Apache-2.0 License** (Fully permissive academic & commercial license)

## 4. Selection Rationale
1. **Instruction Diversity:** Covers diverse NLP tasks including classification, text generation, summarization, extraction, transformation, open QA, and reasoning.
2. **Model Compatibility:** Perfectly tailored for Supervised Fine-Tuning (SFT) of small language models (such as our 6.61M parameter Model C).
3. **Context Length Fit:** Contains a high volume of concise instruction-response pairs suitable for a 128-token context window (`block_size=128`).
4. **Clean Quality:** Removes Stanford Alpaca artifacts, hallucinations, empty responses, merged prompts, and invalid outputs.
5. **Manageable Footprint:** Download size is ~25MB uncompressed, allowing rapid deterministic preprocessing.

## 5. Dataset Size & Audit Summary Metrics
| Audit Metric | Count / Value |
| :--- | :--- |
| **`raw_example_count`** | 51,760 |
| **`malformed_example_count`** | 0 |
| **`missing_instruction_count`** | 0 |
| **`missing_output_count`** | 0 |
| **`empty_example_count`** | 0 |
| **`duplicate_instruction_response_count`** | 4 |
| **`duplicate_records_removed`** | 4 |
| **`final_deduplicated_example_count`** | 51,756 |
| **`train_count`** (80%) | 41,404 |
| **`validation_count`** (10%) | 5,176 |
| **`test_count`** (10%) | 5,176 |
| **`cross_split_duplicate_count`** | 0 |

## 6. Schema & Field Definitions
The raw dataset consists of JSON objects with three core string fields:
- `instruction`: Task description / user query.
- `input`: Additional contextual input (optional, default `"None"` when empty).
- `output`: Expected target response.

## 7. Formatting Strategy
Every example is formatted deterministically into a single prompt string:
```text
Instruction:
<instruction>

Input:
<input or 'None'>

Response:
<output>
```

## 8. Tokenizer Used
- **Tokenizer Type:** Custom Byte Pair Encoding (`BPETokenizer`)
- **Vocabulary File:** [`tokenizers/phase5d/bpe_vocab_1024.json`](file:///C:/Users/dell9/OneDrive/Desktop/minigpt/tokenizers/phase5d/bpe_vocab_1024.json)
- **Vocabulary Size:** 1024 tokens

## 9. Token, Character, and Word Statistics

### Token Count Distribution (1024-Vocab BPE)
| Metric | Count / Length |
| :--- | :--- |
| **Min Tokens** | 42 |
| **Max Tokens** | 3910 |
| **Mean Tokens** | 407.86 |
| **Median Tokens (p50)** | 318.0 |
| **75th Percentile (p75)** | 629.0 |
| **90th Percentile (p90)** | 881.0 |
| **95th Percentile (p95)** | 1010.0 |
| **99th Percentile (p99)** | 1255.0 |

### Character & Word Count Summary
| Metric | Character Count | Word Count |
| :--- | :--- | :--- |
| **Mean** | 802.23 | 128.31 |
| **Median (p50)** | 617.0 | 100.0 |
| **p75** | 1251.0 | 200.0 |
| **p90** | 1779.0 | 282.0 |

## 10. Context Window Length Breakdown (block_size = 128)
- **Examples fitting within 128 tokens (`examples <= 128`):** 12,952 (25.03%)
- **Examples exceeding 128 tokens (`examples > 128`):** 38,804 (74.97%)
- **`percentage_requiring_truncation`:** **74.97%**

## 11. Deduplication Breakdown
- **Detected Duplicate Instruction/Response Pairs:** 4
- **Duplicate Records Removed:** 4
- **Timing:** Global deduplication performed BEFORE train/val/test splitting.

## 12. Invalid / Missing Sample Statistics
- **`malformed_example_count`:** 0
- **`missing_instruction_count`:** 0
- **`missing_output_count`:** 0
- **`empty_example_count`:** 0

## 13. Train / Validation / Test Partitioning
Partitioned using an **80% / 10% / 10%** split after global deduplication:
- **`train_count`:** 41,404 (80.0%)
- **`validation_count`:** 5,176 (10.0%)
- **`test_count`:** 5,176 (10.0%)
- **`cross_split_duplicate_count`:** 0

## 14. Random Seed
- **Fixed Random Seed:** `42` (Ensures 100% reproducible splitting)

## 15. Manual Sample Inspection Summary
Sampled 20 representative examples across short, medium, long, and input-conditioned categories:
- Short examples (<=64 tokens) display clean syntax and direct answers.
- Medium examples (65-128 tokens) provide structured, complete responses.
- Long examples (>128 tokens) contain detailed code/explanations that require context truncation during SFT training.
- No malformed formatting, unsafe content, or prompt-response mismatch observed.

## 16. Known Limitations
1. **Context Window Constraint:** ~74.97% of raw dataset examples exceed 128 tokens, requiring truncation at `block_size=128`.
2. **Tokenizer Compression:** 1024-vocabulary BPE tokenizer has lower compression ratio compared to 32k or 50k vocabularies, leading to longer token sequences per character.
3. **Synthetic Generation:** Dataset originates from GPT-3.5 outputs; occasional stylistic patterns (e.g. "Sure, here is...") are present.

## 17. Recommendation for Phase 6B (Supervised Fine-Tuning)
1. **Packing / Truncation Strategy:** SFT data loader should format sequences using target response loss masking (compute loss only on `Response:` tokens).
2. **Concise Filtering Option:** Consider filtering training split to examples with token_count <= 128 to avoid losing response completions, or apply prompt-response aware truncation.
3. **Base Checkpoint:** Use exclusively `checkpoints/phase5d/model_6_61m/best_model.pt`.

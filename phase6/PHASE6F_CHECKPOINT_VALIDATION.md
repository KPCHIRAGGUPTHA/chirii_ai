# Phase 6F: SFT Checkpoint & Validation Report

## 1. Executive Summary
Phase 6F independently validates that the Phase 6E Supervised Fine-Tuning (SFT) best model checkpoint ([`checkpoints/phase6/model_c_sft/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/best_model.pt)) is valid, loadable, deterministic, and fully isolated from the base Phase 5D model and Phase 6 test set.

**Overall Validation Status**: **PASS**

---

## 2. Checkpoint & Artifact Hashes
| Asset | File Path | Expected SHA-256 | Calculated SHA-256 | Match |
| :--- | :--- | :--- | :--- | :--- |
| **Base Model C** | [`checkpoints/phase5d/model_6_61m/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase5d/model_6_61m/best_model.pt) | `6f934bc3f2ca1cff...` | `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433` | **YES** |
| **Tokenizer** | [`tokenizers/phase5d/bpe_vocab_1024.json`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/tokenizers/phase5d/bpe_vocab_1024.json) | `6436b59303ef54cb...` | `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d` | **YES** |
| **SFT Best Checkpoint** | [`checkpoints/phase6/model_c_sft/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/best_model.pt) | `14f9335aa66d7168...` | `14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26` | **YES** |
| **Pilot Checkpoint** | [`checkpoints/phase6/model_c_sft/cpu_pilot_50steps.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/cpu_pilot_50steps.pt) | Preserved | Preserved | **YES** |

- **SFT Checkpoint File Size**: `27,018,621` bytes
- **SFT Checkpoint Modification Time**: `Mon Oct  5 15:09:08 2026`

---

## 3. Architecture & Parameter Health
- **Total Parameter Count**: `6,613,504` (6.61M parameters)
- **Transformer Layers (`n_layer`)**: `8`
- **Attention Heads (`n_head`)**: `8`
- **Embedding Dimension (`n_embd`)**: `256`
- **Vocabulary Size (`vocab_size`)**: `1024`
- **Context Length (`block_size`)**: `128`
- **Parameter Health**: `0` NaN tensors, `0` Inf tensors across `100` parameter tensors.
- **Forward Pass Verification**: Verified output logits shape `(8, 128, 1024)` on validation batch.
- **Reload Determinism Test**: Verified Run A and Run B produce 100% bitwise identical logits and loss.

---

## 4. Validation Set Evaluation
- **Validation Dataset**: [`phase6/data/sft/val_sft.jsonl`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/data/sft/val_sft.jsonl)
- **Validation Examples**: `5,176`
- **Active Response Tokens Evaluated**: `277,210`
- **Validation Loss**: **`2.6027`**
- **Validation Perplexity**: **`13.5007`**

---

## 5. Isolation Checks
1. **Phase 5D Base Checkpoint**: Unchanged byte-for-byte (`6f934...`).
2. **BPE Tokenizer**: Unchanged byte-for-byte (`6436b...`).
3. **50-Step Pilot Checkpoint**: Preserved in output directory.
4. **Test Set Partition (`test_sft.jsonl`)**: Preserved strictly untouched for Phase 6G evaluation.

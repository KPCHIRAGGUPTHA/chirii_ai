# Testing Framework Documentation for Mini-GPT

## Overview

The Mini-GPT test suite is built with `pytest` to provide thorough unit, integration, and regression testing across all modules in the project.

## Running Tests

To run the complete test suite:

```bash
pytest -v
```

To run a specific test file:

```bash
pytest tests/test_model.py -v
```

To run a specific test case:

```bash
pytest tests/test_model.py -k test_parameter_count -v
```

---

## Test Suite Structure

```
tests/
├── __init__.py
├── test_app.py          # HTTP API server integration tests
├── test_attention.py    # Causal self-attention layer and mask verification
├── test_dataset.py      # Dataset fetching, batch generation, and input validation
├── test_generation.py   # Text generation logic, sampling parameters, checkpoint loading
├── test_model.py        # MiniGPT architecture, parameter calculation, forward pass, shape validation
├── test_regressions.py  # Explicit regression tests for all identified edge cases and past bugs
├── test_tokenizer.py    # Vocabulary management, encoding/decoding, save/load roundtrips
└── test_training.py     # Overfitting verification, training loop execution, callback dispatch
```

---

## Test Modules Detail

### 1. `test_tokenizer.py`
- `test_vocabulary_size`: Verifies standard default printable ASCII vocabulary size (97 tokens).
- `test_special_characters`: Validates out-of-vocabulary character encoding to `<unk>` ID.
- `test_roundtrip`: Verifies `encode -> decode` string reconstruction consistency.
- `test_from_text`: Validates dynamic vocabulary extraction from training corpus text.
- `test_save_load_roundtrip`: Tests saving vocabulary to JSON and reloading preserves `vocab_size`, `stoi`, `itos`, and `unk_id`.
- `test_empty_string`: Validates encoding and decoding empty string inputs.

### 2. `test_attention.py`
- `test_causal_mask`: Verifies lower-triangular causal attention mask structure (1s on/below diagonal, 0s above).
- `test_attention_invalid_dim`: Asserts error raised when `n_embd` is not divisible by `n_head`.

### 3. `test_model.py`
- `test_parameter_count`: Checks total trainable parameters against exact mathematical architecture formula.
- `test_forward_pass_with_targets`: Tests training mode forward pass shapes `(B, T, vocab_size)` and scalar cross-entropy loss.
- `test_forward_pass_without_targets`: Tests inference mode forward pass returning `(B, 1, vocab_size)` and `loss is None`.
- `test_target_shape_mismatch`: Asserts shape validation error when target tensor does not match input sequence length.
- `test_sequence_length_exceeded`: Asserts error when input sequence exceeds `block_size`.
- `test_generate_method`: Tests autoregressive generation tensor shapes.

### 4. `test_dataset.py`
- `test_dataset_download`: Tests text file retrieval or fallback to built-in Shakespeare sample.
- `test_data_batching`: Tests batch shapes `(batch_size, block_size)` for `x` and `y`.
- `test_data_batching_target_shift`: Verifies `y` is input sequence `x` shifted by 1 token.
- `test_short_data_validation`: Validates `ValueError` when dataset length is `<= block_size`.

### 5. `test_training.py`
- `test_overfitting`: Verifies that a small model overfits a tiny dataset sample within 200 iterations (loss < 0.1).
- `test_train_model_execution`: Tests full training routine in isolated temp directory, ensuring `checkpoint.pt` and `vocab.json` generation.
- `test_train_model_callback`: Tests training progress callback dispatch.

### 6. `test_generation.py`
- `test_load_model_and_tokenizer`: Tests loading checkpoint and vocabulary from directory.
- `test_load_model_missing_checkpoint_raises`: Asserts `FileNotFoundError` when attempting to load from empty directory.
- `test_generation_respects_context`: Verifies generated text begins with original prompt.
- `test_generation_produces_in_vocab_tokens`: Verifies generated characters exist in tokenizer vocabulary.
- `test_generation_respects_temperature`: Tests generation output behavior under different temperature scales.
- `test_generation_respects_top_k_top_p`: Tests top-k and top-p sampling bounds.

### 7. `test_app.py`
- `test_api_info_endpoint`: GET `/api/info` returns 200 OK with model state details.
- `test_api_train_status_endpoint`: GET `/api/train/status` returns 200 OK with status dictionary.
- `test_api_invalid_endpoint`: GET `/api/nonexistent` returns 404 HTTP Error.

### 8. `test_regressions.py`
- Contains 8 explicit regression tests preventing recurrence of fixed implementation bugs and test specification mismatches.

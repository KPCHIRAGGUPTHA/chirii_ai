# Phase 1 Codebase & Test Suite Audit Report

## Summary

Phase 1 focused on completing, auditing, and executing the pytest test suite for Mini-GPT. The audit evaluated all core modules (`tokenizer.py`, `model.py`, `dataset.py`, `train.py`, `generate.py`, `app.py`) alongside existing unit tests in `tests/`.

---

## Initial Execution Analysis

Upon running `pytest -v` initially, **13 test items** were collected:
- **Passed**: 7
- **Failed**: 6

### Detailed Breakdown of Initial Failures

1. **`tests/test_attention.py::test_causal_mask`**
   - **Root Cause**: Test Logic Bug. The test asserted `torch.all(attn.bias[0,0,:,:][:32,:32] == 0)`. In `CausalSelfAttention`, `attn.bias` is a lower-triangular matrix (`tril`) with `1`s where attention is allowed and `0`s where masked out (`-inf`). Asserting that the matrix is all zeros was mathematically incorrect.
   - **Action**: Corrected test assertion to verify `attn.bias` matches `torch.tril(torch.ones(32, 32))`.

2. **`tests/test_dataset.py::test_data_batching`**
   - **Root Cause**: Test Assertion Bug. The test asserted `x.shape[1] == y.shape[1] + 1` (32 == 33). In causal language modeling batching, both input `x` and target `y` have sequence length `block_size` (32).
   - **Action**: Fixed assertion to verify `x.shape == y.shape == (batch_size, block_size)`.

3. **`tests/test_generation.py::test_generation_produces_in_vocab_tokens`**
   - **Root Cause**: Test Logic Bug. The test called `.split()` on decoded text string and checked if multi-character words were keys in `tokenizer.itos`. Since `CharTokenizer` is a character-level model with integer keys in `itos`, word-level string lookups failed.
   - **Action**: Updated test to iterate over characters in generated text and verify membership in `tokenizer.stoi`.

4. **`tests/test_model.py::test_parameter_count`**
   - **Root Cause**: Test Formula Bug. The formula `(vocab_size * n_embd) + (n_embd * 4 * n_embd * n_layer)` omitted positional embeddings (`wpe`), linear layer biases, layer norm parameters, and weight-tying logic.
   - **Action**: Implemented exact architectural parameter calculation matching `MiniGPT`.

5. **`tests/test_tokenizer.py::test_special_characters`**
   - **Root Cause**: Test Fixture Bug. The string `"Hello, ?!"` used `'!'`, which exists in default printable ASCII vocabulary.
   - **Action**: Used an out-of-vocabulary Unicode character (`"\u1234"`) to properly verify `<unk>` mapping.

6. **`tests/test_training.py::test_overfitting`**
   - **Root Cause**: Test Setup & Config Mismatch. `CharTokenizer.from_text("The cat sat on the mat.")` built a vocabulary of size 12, but `MiniGPTConfig` was instantiated with `vocab_size=10`. Token indices exceeding 9 caused PyTorch `IndexError` inside embedding lookup.
   - **Action**: Aligned `MiniGPTConfig.vocab_size` with `tokenizer.vocab_size` and corrected target slice indexing.

---

## Genuine Implementation Bugs Discovered & Fixed

1. **`dataset.py`: Short Dataset Runtime Exception**
   - **Issue**: Calling `get_batch` with `len(data_tensor) <= block_size` caused PyTorch to throw `RuntimeError: high must be > low`.
   - **Fix**: Added explicit input validation `if len(data_tensor) <= block_size: raise ValueError(...)`.

2. **`model.py`: Target Shape Mismatch**
   - **Issue**: Passing targets with sequence lengths mismatching input tensor `idx` caused silent reshape failures or PyTorch dimension mismatches in `cross_entropy`.
   - **Fix**: Added explicit target shape validation `assert idx.size() == targets.size()`.

---

## Test Suite Expansion

To achieve comprehensive coverage for Phase 1, the following files were created:
1. **`tests/test_app.py`**: Added integration tests for HTTP API server endpoints (`/api/info`, `/api/train/status`, invalid routes).
2. **`tests/test_regressions.py`**: Added 8 explicit regression test cases covering all fixed issues and edge cases.

---

## Final Results

All **38 test cases** pass cleanly with 100% success rate.

---

## Phase 2 Training and Evaluation Results

### Baseline Experiment Configuration
- **Model Architecture**: Unmodified Decoder-Only MiniGPT (`n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128`, `dropout=0.1`)
- **Total Parameters**: 818,176
- **Tokenizer**: Character-level `CharTokenizer` (Vocabulary Size: 66 tokens)
- **Dataset**: Tiny Shakespeare (`1,115,394` characters; `1,003,854` train tokens, `111,540` validation tokens)
- **Training Iterations**: 600
- **Batch Size**: 32
- **Learning Rate**: 0.001 (AdamW)
- **Evaluation Settings**: `eval_interval=50`, `eval_iters=20` (multi-batch averaged loss under `@torch.no_grad()`)
- **Random Seed**: 42 (CPU execution mode)

### Quantitative Baseline Results
- **Training Time**: 492.19 seconds (~8.2 minutes)
- **Final Training Loss**: 2.0241
- **Best Validation Loss**: 2.0936 (achieved at iteration 600)
- **Best Validation Perplexity**: 8.1142 ($\exp(2.0936)$)
- **Evaluated Validation Loss (50 batches)**: 2.0898
- **Evaluated Validation Perplexity (50 batches)**: 8.0832
- **Saved Checkpoints**: `checkpoints/best_model.pt`, `checkpoints/final_model.pt`, `checkpoints/checkpoint.pt`
- **Saved Artifacts**: `results/training_history.json`, `results/plots/loss_curve.png`, `results/plots/perplexity_curve.png`, `results/evaluation_summary.json`

### Qualitative Generation Baseline Samples (Seed 42, Temp 0.8, Top-K 40, Top-P 0.9)
- **Prompt `'ROMEO:'`**: `"ROMEO:\nBusten you and the be me best that hist dearss,\nI that to ar if sting a the the"`
- **Prompt `'JULIET:'`**: `"JULIET:\nBusten you are the bet prars in an is theee shars.\n\nKING REDICHARD:\nShat ay, br"`
- **Prompt `'HAMLET:'`**: `"HAMLET:\nBusten you are the bet prars in an is theee shars.\n\nKING REDICHARD:\nShat ay, br"`

### Extended Pytest Test Suite
- **Total Tests**: 47
- **Passed**: 47
- **Failed**: 0
- **Errors**: 0

---

## Phase 2.1 — Baseline Integrity Cleanup

### Artifact Contamination Cause & Remediation
- **Root Cause**: `train_model()` defaulted to `results_dir="results"`. Multiple test cases (`test_generation.py`, `test_training.py`, `test_evaluation.py`) invoked `train_model()` with `out_dir=tmpdir` but omitted `results_dir`, causing test training runs (`max_iters=10`, `n_layer=1`, `n_embd=32`) to overwrite `results/training_history.json`.
- **Isolation Fix**: All test cases were updated to explicitly pass `results_dir=tmpdir` (or `results_dir=os.path.join(tmpdir, "results")`). All test checkpoints, histories, plots, and summaries now execute in isolated temporary directories.
- **Recovery of Official History**: The complete 12-iteration log of the 600-iteration Phase 2 training run was retrieved from task logs (`task-155.log`) and restored to `results/training_history.json`. Plots were regenerated via `plot_history.py`.

### Checkpoint Configuration & Parameter Count Verification
- **Model Checkpoints**: `checkpoints/best_model.pt`, `checkpoints/final_model.pt`, `checkpoints/checkpoint.pt`
- **Stored Config**: `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128`, `vocab_size=66`, `dropout=0.1`
- **Parameter Count Verification**: 818,176 parameters.
  - *Explanation of 892K vs 818,176*: The value ~892K mentioned prior was an unconstrained preliminary estimate. For a character vocabulary of 66 tokens (65 characters from Tiny Shakespeare + `<unk>`), 4 layers, 4 heads, 128 embedding size, 128 context window, and weight tying between `wte` and `lm_head`, the exact mathematical parameter count is **818,176**.

### Final Test Suite & Artifact Integrity Status
- **Pytest Output**: 47 passed, 0 failed (100% pass rate).
- **Artifact Protection**: Verified `results/training_history.json`, `results/plots/`, `checkpoints/` remain completely untouched after running `pytest -v`.
- **Baseline Reproducibility**: Official Phase 2 Baseline is 100% clean, verified, and reproducible.



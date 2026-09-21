# Phase 6C: SFT Training Pipeline Implementation Report

## 1. Base Checkpoint
- **Path:** [`checkpoints/phase5d/model_6_61m/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase5d/model_6_61m/best_model.pt)
- **Status:** Verified 100% byte-for-byte identical before and after Phase 6C pipeline validation.

## 2. Model Architecture
- **Parameters:** 6,613,504 (6.61M parameters)
- **Layers (`n_layer`):** 8
- **Attention Heads (`n_head`):** 8
- **Embedding Dimension (`n_embd`):** 256
- **Vocabulary Size (`vocab_size`):** 1024
- **Context Length (`block_size`):** 128
- **Positional Embedding Matrix (`wpe`):** Shape `(128, 256)`

## 3. Tokenizer
- **Tokenizer:** Custom Byte Pair Encoding (`BPETokenizer`)
- **Vocabulary File:** [`tokenizers/phase5d/bpe_vocab_1024.json`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/tokenizers/phase5d/bpe_vocab_1024.json)
- **Status:** 100% UNCHANGED (verified SHA-256 `6436b59303ef54cb...`).

## 4. Dataset
- **SFT Records:** Prepared Phase 6B JSONL datasets:
  - Train: `phase6/data/sft/train_sft.jsonl` (41,404 records)
  - Validation: `phase6/data/sft/val_sft.jsonl` (5,176 records)
  - Test: `phase6/data/sft/test_sft.jsonl` (5,176 records)

## 5. Context Length
- **`block_size = 128`** tokens. Retained out-of-the-box compatibility with Model C's positional embedding matrix `(128, 256)`.

## 6. SFT Formatting
Deterministic prompt structure via `format_alpaca_prompt()`:
```text
Instruction:
<instruction>

Input:
<input or 'None'>

Response:
<output>
```

## 7. Label Masking (Target Response Masking)
- **Prompt Tokens (`Instruction:` & `Input:` up to `Response:` header):** Target labels set to **`-100`** (PyTorch `CrossEntropyLoss(ignore_index=-100)`).
- **Response Tokens (`output` text):** Retain actual target token IDs. Loss is computed strictly on response generation.

## 8. Autoregressive Shifting & Alignment
- **Autoregressive Shift:**
  - Input sequence $X$: `tokens[0 : N-1]`
  - Target sequence $Y$: `tokens[1 : N]`
- **Alignment Rule:**
  - For target index $i \in [0, N-2]$, corresponding target token is `tokens[i+1]`.
  - If $i+1 < \text{prompt\_token\_len}$, target is a prompt token $\Rightarrow \text{label}[i] = -100$.
  - If $i+1 \ge \text{prompt\_token\_len}$, target is a response token $\Rightarrow \text{label}[i] = \text{tokens}[i+1]$.
  - Verified with hand-checkable example: `[P1, P2, P3, R1, R2, R3]` ($\text{prompt\_token\_len}=3$) yields inputs `[P1, P2, P3, R1, R2]` and labels `[-100, -100, R1, R2, R3]`. Target corresponding to $R1$ at index 2 is UNMASKED.

## 9. Padding Strategy
- **Sequence Padding:** Sequences shorter than `block_size=128` are padded with `eos_token_id=0` in `input_ids` and `-100` in `labels`.
- **Loss Exclusion:** Padded positions in `labels` are set to `-100`, ensuring they never contribute to gradient updates.

## 10. Optimizer
- **Optimizer:** `AdamW(betas=(0.9, 0.95), weight_decay=0.01)`
- **Weight Decay:** Applied to 2D weight matrices; disabled for 1D biases and LayerNorm parameters.

## 11. Learning-Rate Configuration
- **Initial LR:** `1e-4` (Conservative SFT learning rate; 10x smaller than pretraining `1e-3` to prevent catastrophic forgetting).
- **Minimum LR:** `1e-5`.

## 12. Scheduler
- **Scheduler:** Warmup + Cosine Decay
- **Warmup Iters:** 20 steps
- **Cosine Decay:** Decays to `min_lr = 1e-5` over remaining training iterations.

## 13. Gradient Clipping
- **`grad_clip = 1.0`** (Prevents gradient explosion during fine-tuning).

## 14. Mixed Precision Configuration
- Supported via `torch.cuda.amp.autocast()` on GPU targets; defaults to `fp32` during CPU dry-runs.

## 15. Checkpoint Strategy
- **Output Directory:** [`checkpoints/phase6/model_c_sft/`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/) (Isolated directory; never overwrites Phase 5D checkpoints).

## 16. Dry-Run Results
- **Batch Shape:** `[8, 128]`
- **Masked Prompt/Pad Labels (-100):** `547 / 1024` tokens
- **Active Response Labels:** `477 / 1024` tokens (**46.58%** contributing to loss)
- **Initial Dry-Run Loss:** **2.7943**
- **Gradient Existence:** Verified (`p.grad is not None` for all trainable parameters)
- **Gradient Norm:** **1.6488**

## 17. Tiny Overfitting Sanity-Test Results (4 Examples, CPU)
- **Step 0 Loss:** **2.6646**
- **Step 15 Loss:** **1.1634**
- **Loss Reduction Verified:** `True` (Confirmed pipeline end-to-end gradient flow and optimizer updating).

## 18. Checkpoint Hash Verification
- **Pre-execution SHA-256:** `6f934bc3f2ca1cff24ce4e7b4c2b95c0268aa07119e7019623e198fae830e2f5`
- **Post-execution SHA-256:** `6f934bc3f2ca1cff24ce4e7b4c2b95c0268aa07119e7019623e198fae830e2f5`
- **Status:** **100% BYTE-FOR-BYTE IDENTICAL**.

## 19. Test Results
- **`pytest -q`**: **80 / 80 passed in 23.19s** (68 existing tests + 12 new Phase 6C SFT pipeline unit tests passing 100%).

## 20. Known Limitations
1. **Context Window (128 tokens):** ~74.97% of raw instruction-response pairs exceed 128 tokens and require prompt-response aware truncation.
2. **Tokenizer Compression:** 1024 BPE vocabulary has lower compression than large 32k/50k tokenizers, resulting in longer token sequences per sentence.

## 21. Exact Command to Launch Full SFT (Phase 6D)
```bash
python phase6/training/sft_trainer.py --config phase6/training/sft_config.py
```

## 22. Confirmation of Safe Execution
- **Full SFT / GPU Training Executed:** **NO** (Stopped as directed after pipeline validation).

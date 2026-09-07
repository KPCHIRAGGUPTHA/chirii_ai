# Mini-GPT Training Documentation

## Overview

The Mini-GPT training pipeline provides configurable autoregressive language model training on character-level datasets (such as Tiny Shakespeare). Phase 2 introduces multi-batch validation, perplexity metrics, deterministic random seeding, structured machine-readable logging, and model checkpointing.

---

## Key Training Components

### 1. Reproducibility & Random Seeding
All random number generators (Python `random`, `numpy.random`, `torch.manual_seed`, and CUDA backends) are initialized via `set_seed(seed)`:
```bash
python train.py --seed 42
```
*Note*: Floating-point non-determinism across different GPU architectures or PyTorch backends may lead to slight numerical variances.

### 2. Multi-Batch Validation Loss (`estimate_loss`)
Instead of calculating validation loss on a single random batch, `estimate_loss()` averages loss across `eval_iters` (default 20) batches for both train and validation splits:
- Computed inside a `@torch.no_grad()` block to prevent memory accumulation.
- Automatically restores the model to `.train()` mode after estimation.

### 3. Perplexity Metric (`exp(val_loss)`)
Validation perplexity measures how well the probability distribution predicted by the model matches the actual text distribution:
$$\text{Perplexity} = \exp(\text{val\_loss})$$
- Lower perplexity indicates superior predictive confidence.
- Calculated with upper-bound clamping to prevent numeric overflow.

### 4. Checkpointing Strategy
During training, three checkpoint files are maintained in the output directory (default `checkpoints/`):
- `best_model.pt`: Automatically saved whenever `val_loss` reaches a new historical minimum.
- `final_model.pt`: Saved upon completing the final training iteration.
- `checkpoint.pt`: Maintained for backwards compatibility with legacy scripts.

Each checkpoint file stores:
- `config`: `MiniGPTConfig` instance (`vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`)
- `model_state`: PyTorch `state_dict`
- `optimizer_state`: AdamW optimizer `state_dict`
- `step`: Current iteration step
- `val_loss` & `best_val_loss`: Validation loss statistics
- `val_perplexity`: Computed validation perplexity
- `history`: Cumulative step statistics
- `seed` & `timestamp`: Run metadata

### 5. Structured Training History
Evaluation statistics are recorded in `results/training_history.json`:
```json
{
  "config": {
    "vocab_size": 65,
    "block_size": 128,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "max_iters": 600
  },
  "history": [
    {
      "step": 50,
      "train_loss": 2.4512,
      "val_loss": 2.4831,
      "val_perplexity": 11.9782,
      "elapsed_sec": 12.35
    }
  ]
}
```

---

## How to Run Training

### Standard Command (Baseline Run)

```bash
python train.py --max_iters 600 --eval_interval 50 --eval_iters 20 --seed 42
```

### CLI Command Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data` | `str` | `None` | Path to custom training text file |
| `--out_dir` | `str` | `"checkpoints"` | Directory for model checkpoints |
| `--results_dir` | `str` | `"results"` | Directory for training history JSON |
| `--max_iters` | `int` | `600` | Total training iterations |
| `--batch_size` | `int` | `32` | Batch size |
| `--block_size` | `int` | `128` | Sequence context length |
| `--n_layer` | `int` | `4` | Transformer block layers |
| `--n_head` | `int` | `4` | Multi-head attention heads |
| `--n_embd` | `int` | `128` | Embedding dimension |
| `--dropout` | `float` | `0.1` | Dropout probability |
| `--lr` | `float` | `1e-3` | Learning rate |
| `--eval_interval`| `int` | `50` | Iteration interval for validation |
| `--eval_iters` | `int` | `20` | Batches per evaluation estimate |
| `--seed` | `int` | `42` | Random seed |

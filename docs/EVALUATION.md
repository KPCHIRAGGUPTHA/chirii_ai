# Mini-GPT Evaluation Documentation

## Overview

The evaluation suite for Mini-GPT provides quantitative metric reporting, qualitative text generation analysis, and visual curve generation.

---

## 1. Quantitative Evaluation (`evaluate.py`)

`evaluate.py` loads a saved checkpoint, calculates evaluation metrics across multiple validation batches, and prints a structured evaluation report.

### Running Evaluation

```bash
python evaluate.py --ckpt_name best_model.pt --eval_iters 50 --seed 42
```

### Metrics Reported

- **Model Parameter Count**: Exact total trainable parameters computed via model inspection.
- **Architecture Dimensions**: Vocabulary size, context length (`block_size`), number of layers (`n_layer`), attention heads (`n_head`), embedding dimension (`n_embd`), and dropout.
- **Dataset Information**: Total dataset characters, training tokens, and validation tokens.
- **Cross-Entropy Loss**: Evaluated training loss and validation loss over `--eval_iters` batches.
- **Validation Perplexity**: Calculated as $\exp(\text{val\_loss})$.

---

## 2. Qualitative Generation Analysis

In addition to scalar loss and perplexity, `evaluate.py` runs deterministic text generation on fixed benchmark prompts:
- `ROMEO:`
- `JULIET:`
- `HAMLET:`

### Sampling Parameters
- Prompt completion length: 80 new tokens
- Temperature: 0.8
- Top-K filtering: 40
- Top-P (nucleus): 0.9
- Random seed: 42 (ensures deterministic output for direct comparison across runs)

*Note*: Qualitative generation outputs serve as a visual sanity check and are labeled as qualitative analysis rather than quantitative proof of model quality.

---

## 3. Training Plot Generation (`plot_history.py`)

`plot_history.py` parses `results/training_history.json` and outputs high-resolution plots under `results/plots/` without requiring model retraining.

### Running the Plotter

```bash
python plot_history.py --history_path results/training_history.json --output_dir results/plots
```

### Generated Artifacts

1. `results/plots/loss_curve.png`:
   - Plots **Train Loss** (blue solid line) vs **Validation Loss** (orange dashed line) across training iterations.
2. `results/plots/perplexity_curve.png`:
   - Plots **Validation Perplexity** (green line) across training iterations.

---

## 4. Perplexity & Loss Interpretation

- **Training Loss vs Validation Loss**: A diverging gap between training loss and validation loss indicates overfitting.
- **Perplexity**: Represents the effective branching factor of token predictions. A perplexity of 15.0 means the model is as uncertain as choosing uniformly among 15 character choices at each step.

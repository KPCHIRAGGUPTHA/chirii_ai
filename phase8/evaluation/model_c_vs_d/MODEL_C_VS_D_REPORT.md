# Phase 8E: Model C vs Model D Evaluation Report

> - **Evaluation Mode**: CPU Inference Only
> - **Validation Token Corpus**: `data/phase5d/val_tokens_v1024.pt` (256,000 of 1,765,698 tokens evaluated, 14.5%)
> - **Timestamp**: `2026-10-31 00:00:55`

---

## 1. Executive Summary

This report evaluates **Model C** and **Model D** with the same cross-entropy loss over the same 256,000 validation tokens, encoded with the Phase 5D BPE tokenizer (`vocab_size=1024`). The evaluation is identical for both models. Their training procedures were not (section 5).

### Recommendation
**DEFER_MODEL_E**: Model D is worse than Model C by 32.92% in validation loss and 168.2% in perplexity (validation loss 3.9837 vs 2.9970, perplexity 53.72 vs 20.03). The cause is not established because the two models were trained differently. Defer Model E until Model D is retrained under Model C's procedure.

---

## 2. Model Specifications & Checkpoint Verification

| Component | Model C | Model D | Difference (Model D - Model C) |
| :--- | :--- | :--- | :--- |
| **Parameters** | 6,613,504 | 15,164,800 | +8,551,296 (+129.30%) |
| **Layers (`n_layer`)** | 8 | 12 | +4 |
| **Heads (`n_head`)** | 8 | 10 | +2 |
| **Embedding Dim (`n_embd`)** | 256 | 320 | +64 |
| **Context Length (`block_size`)** | 128 | 128 | +0 |
| **Vocab Size** | 1024 | 1024 | +0 |
| **Checkpoint SHA-256** | `6f934bc3f2ca1cff...` | `2c3c662b99485b5c...` | Verified |

- **Tokenizer SHA-256**: `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d` (verified)
- **Validation Tokens SHA-256**: `a45af657b27115440ed995f738ebe1509d9b51f973003bbcdce55108f1f49c8a` (verified)

---

## 3. Head-to-Head Metrics

Differences are Model D - Model C. Relative improvement is positive when Model D is better.

| Metric | Model C | Model D | Difference (Model D - Model C) | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Validation Loss** | **2.9970** | **3.9837** | `+0.9867` | **-32.92%** |
| **Validation Perplexity** | **20.03** | **53.72** | `+33.69` | **-168.20%** |
| **Validation Tokens** | 256,000 | 256,000 | +0 | - |
| **CPU Eval Time** | `46.17s` | `88.98s` | `+42.81s` | - |
| **CPU Throughput** | `5,544.56 tok/s` | `2,877.05 tok/s` | `-2,667.51 tok/s` | `0.52x ratio` |
| **Process Peak Memory** | `419.39 MB` | `580.22 MB` | - | - |

- **Memory** is the peak for the whole process, measured after each evaluation. Model D runs second, so its figure can reflect Model C's run and is not a per-model footprint.
- **Timing** comes from a single run with no warmup, Model C first. Use it as a rough indication only; absolute values vary with machine load.

---

## 4. Recorded Training Metrics

Recorded values come from each training run's own validation sample (section 5), so they are not directly comparable with section 3. Tokens per parameter uses the 10.24M token budget.

| Model | Parameters | Tokens per Parameter | Recorded Val Loss | Recorded Val PPL | Source |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Model A | 940,800 | 10.88 | 3.2288 | 25.25 | `results/phase5d/MODEL_C_GPU_REPORT.md` |
| Model B | 2,890,752 | 3.54 | 3.0642 | 21.42 | `results/phase5d/MODEL_C_GPU_REPORT.md` |
| Model C | 6,613,504 | 1.55 | 2.9699 | 19.49 | `results/phase5d/MODEL_C_GPU_REPORT.md` |
| Model D | 15,164,800 | 0.68 | 3.9488 | 51.87 | `checkpoints/phase8/model_d/training_report.md` |

---

## 5. Interpretation

### Training differences

Model C and Model D had the same token budget, step count, learning rate schedule and gradient clipping, but differed as follows. The Phase 8A design lists model capacity as the only variable, so these differences were not planned.

| Setting | Model C (`run_model_c_gpu.py`) | Model D (`phase8/train_model_scaling.py`) |
| :--- | :--- | :--- |
| Batch sampling | Random positions (`torch.randint`) | Sequential, no shuffling |
| Training data seen | Sampled from the full training token file | First 10.24M tokens of the file only |
| AdamW weight decay | 0.01 (PyTorch default) | 0.0 |
| Recorded validation sample | 20 random batches per evaluation | Fixed first 20,480 validation tokens |

### What this evaluation can and cannot conclude

- It measures how these two checkpoints perform on the same validation tokens.
- It cannot separate the effect of model capacity from the effect of the training differences, so token starvation is one hypothesis and not an established cause.
- The Phase 8A design's RESULT C (optimization failure) also remains open.
- The RAG exact-match gate is not measured by this evaluation.

### Token budget context

Chinchilla's ~20 tokens per parameter is a rule of thumb for compute-optimal training, not a convergence requirement.

| Model | Parameters | Tokens per Parameter at 10.24M | Tokens at ~20/Parameter |
| :--- | ---: | ---: | ---: |
| Model C | 6,613,504 | 1.55 | ~132M |
| Model D | 15,164,800 | 0.68 | ~303M |
| Model E | 29,488,256 | 0.35 | ~590M |

### Next experiment

Retrain Model D at the same 10.24M token budget using Model C's batch sampling and weight decay, then repeat this evaluation. Do not expand the token budget or train Model E with `phase8/train_model_scaling.py` until that result is known.

---

## 6. Safety and Audit Checklist

- [x] Evaluation-only execution.
- [x] No gradient computation or model weight updates.
- [x] No modification of existing checkpoints or tokenizers.
- [x] Isolated outputs stored in `phase8/evaluation/model_c_vs_d`.
- [x] SHA-256 hashes of Model C, Model D, Tokenizer and Validation Tokens verified.

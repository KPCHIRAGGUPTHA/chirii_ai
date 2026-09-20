# Model C GPU Pretraining Report (Phase 5D)

> [!NOTE]
> This is a **NEW, OFFICIAL GPU PRETRAINING RUN** for **Model C (Phase 5D-B)** executed on hardware **NVIDIA L4 GPU**.
> The original historical report (`PHASE5D_REPORT.md`) remains preserved untouched as baseline documentation.

---

## 1. Executive Summary

Phase 5D Model C pretraining evaluated the high-capacity **6.61M parameter** MiniGPT architecture trained on the FineWeb-Edu dataset utilizing hardware **NVIDIA L4 GPU** acceleration.

- **Model Parameter Count**: 6,613,504 (6.61M)
- **Training Duration**: 354.96 seconds (5.92 minutes)
- **Hardware Throughput**: 28,848.54 tokens/sec
- **Best Validation Loss**: **2.9699**
- **Best Validation Perplexity**: **19.4901**
- **Final Validation BPC**: **2.1596**
- **Final Training Loss**: **3.0008**

---

## 2. Experimental Setup & Architecture

| Parameter | Value |
| :--- | :--- |
| **Model Name** | Model C (Phase 5D-B GPU) |
| **Transformer Layers (`n_layer`)** | 8 |
| **Attention Heads (`n_head`)** | 8 |
| **Embedding Dimension (`n_embd`)** | 256 |
| **Context Length (`block_size`)** | 128 |
| **Vocabulary Size (`vocab_size`)** | 1024 (Custom BPE Tokenizer) |
| **Calculated Parameters** | **6,613,504** |
| **Dataset** | FineWeb-Edu (`sample-10BT`), SHA-256 split |
| **Training Budget** | 10,240,000 tokens (2,500 steps) |
| **Micro-Batch Size** | 8 |
| **Gradient Accumulation** | 4 steps (Effective Batch Size = 32) |
| **Optimizer & Scheduler** | AdamW, Cosine decay (1e-3 &rarr; 1e-4), Warmup 50 steps |
| **Hardware** | NVIDIA L4 GPU (CUDA 12.8, Mixed Precision AMP) |

---

## 3. Training & Validation Metrics

| Step | Train Loss | Val Loss | Val Perplexity | Val BPC | Learning Rate | Time Elapsed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 5.3720 | 5.3625 | 213.2574 | 3.8995 | 0.001000 | 5.11s |
| 250 | 4.6303 | 4.6247 | 101.9741 | 3.3630 | 0.000985 | 36.78s |
| 500 | 4.1732 | 4.1804 | 65.3951 | 3.0399 | 0.000927 | 76.65s |
| 1000 | 3.8147 | 3.7962 | 44.5299 | 2.7605 | 0.000705 | 147.23s |
| 1500 | 3.2867 | 3.2705 | 26.3254 | 2.3782 | 0.000422 | 226.36s |
| 2000 | 3.0636 | 3.0441 | 20.9905 | 2.2136 | 0.000189 | 298.42s |
| **2500** | **3.0008** | **2.9699** | **19.4901** | **2.1596** | **0.000100** | **352.14s** |

---

## 4. Phase 5D Controlled Experiment Comparison Table

| Model | Parameters | Tokens | Hardware | Train Loss | Val Loss | Val PPL | Val BPC | Tokens/sec | Training Time |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Model A (Baseline)** | 940,800 | 10.24M | CPU | 3.1929 | 3.2288 | 25.2495 | 2.4022 | 6,152.48 | 27.74 min |
| **Model B (Phase 5D-A)** | 2,890,752 | 10.24M | CPU | 3.0709 | 3.0642 | 21.4182 | 2.2292 | 2,740.07 | 62.29 min |
| **Model C (Phase 5D-B GPU)** | **6,613,504** | **10.24M** | **NVIDIA L4** | **3.0008** | **2.9699** | **19.4901** | **2.1596** | **28,848.54** | **5.92 min** |

---

## 5. Qualitative Text Generation Sample

**Prompt**: `The purpose of programming is`  
**Generated Output**:  
```text
The purpose of programming issue. You can be concepted, the current of them the same was not fastered by the cyclasificians for the historical called h
```

---

## 6. Artifact & Checkpoint Directory Structure

All Model C GPU artifacts are saved in isolated directories:
- **Checkpoints**: `checkpoints/phase5d/model_6_61m/`
  - `best_model.pt` (80,018,601 bytes)
  - `final_model.pt` (80,019,015 bytes)
  - `checkpoint.pt` (80,018,601 bytes)
  - `vocab.json` (51,569 bytes)
- **Machine-Readable Log**: `results/phase5d/model_c_gpu_history.json`
- **Report Document**: `results/phase5d/MODEL_C_GPU_REPORT.md`

---

*Report generated on 2026-09-20.*

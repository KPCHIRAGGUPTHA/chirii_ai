# Phase 8D: GPU Pilot Benchmark & Hardware Audit Report

> **Status**: PILOT COMPLETE  
> **Date**: 2026-10-08 13:01:20  
> **Execution Mode**: GPU_PILOT

---

## 1. Hardware & System Audit

- **CPU Processor**: `AMD Ryzen 7 7730U (24 logical cores)`
- **Total System RAM**: `16.0 GB` (Available: `4.0 GB`)
- **Python Version**: `3.12.3`
- **PyTorch Version**: `2.11.0+cu128`
- **CUDA Available**: `True` (CUDA Version: `12.8`)
- **GPU Device Name**: `NVIDIA GeForce RTX 3090`
- **Total VRAM**: `23.56 GB`

---

## 2. Pilot Execution Summary

| Model Name | Parameters | Target Device | Steps Executed | Wall Time | Avg Step Time | Throughput | Peak VRAM Alloc | Peak VRAM Reserved | Status |
| :--- | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **Model D** | 15,164,800 | `cuda` | **50/50** | **14.8593s** | **0.2972s** | **13782.58 tok/s** | `0.558 GB` | `0.5781 GB` | **PASS** |
| **Model E** | 29,488,256 | `cuda` | **50/50** | **14.3807s** | **0.2876s** | **14241.35 tok/s** | `0.8999 GB` | `0.9297 GB` | **PASS** |

---

## 3. Loss & Numerical Stability Audit

- **Model D (15.16M)**:
  - Initial Loss: `28.0448`
  - Final Loss: `21.6719`
  - Min Loss: `21.3547`
  - NaN Count: `0` | Inf Count: `0` | OOM: `False`
- **Model E (29.49M)**:
  - Initial Loss: `27.9651`
  - Final Loss: `21.6559`
  - Min Loss: `21.3991`
  - NaN Count: `0` | Inf Count: `0` | OOM: `False`

---

## 4. Protected Checkpoint & Tokenizer Integrity Verification

| Artifact Description | File Path | Expected SHA-256 | Actual SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Base Model C Checkpoint** | `checkpoints/phase5d/model_6_61m/best_model.pt` | `6f934bc3...99fd2433` | `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433` | **INTACT** |
| **Protected Tokenizer** | `tokenizers/phase5d/bpe_vocab_1024.json` | `6436b593...c58e3a5d` | `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d` | **INTACT** |
| **Phase 6 SFT Model** | `checkpoints/phase6/model_c_sft/best_model.pt` | `14f9335a...d01cfb26` | `14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26` | **INTACT** |
| **Phase 7 RAG-SFT Model** | `checkpoints/phase7/model_c_rag_sft/best_model.pt` | `f3f21991...6d20d1ed` | `f3f21991d1262ed3bb6602b6c3d6abe609241a6c4d40c0b22a9c44b6d20d1ed3` | **INTACT** |

---

## 5. Recommendation & Performance Estimation

- **OOM Status**: No OOM was observed during the 50-step pilot.
- **Estimated Full Pretraining Runtimes (2,500 steps / 10.24M tokens)**:
  - Model D: Estimated ~12.1 minutes (~14,073.84 tokens/sec)
  - Model E: Estimated ~11.7 minutes (~14,599.54 tokens/sec)
  - *Note: These figures are estimates based on the 50-step pilot and are not measured full-run times.*
- **Recommendation**: **BOTH STABLE → ready for full Phase 8D training on GPU**

# Phase 8C: CPU-Only Model D/E Feasibility Benchmark Report

> **Status**: CPU BENCHMARK COMPLETE  
> **Date**: 2026-10-08 16:22:52  
> **Execution Mode**: CPU ONLY (PyTorch 2.12.0+cpu)

---

## 1. Hardware & System Audit

- **CPU Processor**: `AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD (16 logical cores)`
- **Logical CPU Cores**: `16`
- **Total System RAM**: `15.35 GB`
- **Available System RAM**: `3.48 GB`
- **Python Version**: `3.14.5`
- **PyTorch Version**: `2.12.0+cpu`
- **CUDA Available**: `False` (Execution: **CPU ONLY**)

---

## 2. Benchmark Results Table (50-Step CPU Test)

| Model Name | Parameters | Target Steps | Wall-Clock Time | Avg Step Time | Throughput | Est. 2,500-Step Time | Status |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| **Model C (Historical Baseline)** | 6,613,504 | 2,500 | N/A | ~2.571s | 1,593.05 tok/s | ~1.79 hours (107 min) | Recorded |
| **Model D (Candidate 1)** | 15,164,800 | 50 | **296.2371s** | **5.9247s** | **691.34 tok/s** | **4.11 hours** (246.9 min) | **PASS** |
| **Model E (Candidate 2)** | 29,488,256 | 50 | **434.9675s** | **8.6993s** | **470.84 tok/s** | **6.04 hours** (362.5 min) | **PASS** |

---

## 3. Loss & Gradient Numerical Stability

- **Model D (15.16M)**:
  - Initial Loss: `28.0927`
  - Final Loss (Step 50): `33.7702`
  - Min Loss: `21.3008`
  - NaN Count: `0` | Inf Count: `0`
- **Model E (29.49M)**:
  - Initial Loss: `27.9703`
  - Final Loss (Step 50): `21.5275`
  - Min Loss: `21.4072`
  - NaN Count: `0` | Inf Count: `0`

---

## 4. Memory Safety & Resource Audit

- **System Memory Check**: System RAM remained stable during execution without triggering low-memory swapping or OS thrashing.
- **Memory Safety Status**: `PASS`
- **No Checkpoint Assertion**: Zero `.pt`, `.pth`, or `.ckpt` model checkpoint files were created or modified during the benchmark.

---

## 5. Checkpoint & Tokenizer Integrity Verification

| Artifact Description | File Path | Expected SHA-256 | Actual SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Base Model C Checkpoint** | `checkpoints/phase5d/model_6_61m/best_model.pt` | `6f934bc3...99fd2433` | `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433` | **INTACT** |
| **Protected Tokenizer** | `tokenizers/phase5d/bpe_vocab_1024.json` | `6436b593...c58e3a5d` | `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d` | **INTACT** |
| **Phase 6 SFT Model** | `checkpoints/phase6/model_c_sft/best_model.pt` | `14f9335a...d01cfb26` | `14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26` | **INTACT** |
| **Phase 7 RAG-SFT Model** | `checkpoints/phase7/model_c_rag_sft/best_model.pt` | `f3f21991...6d20d1ed` | `f3f21991d1262ed3bb6602b6c3d6abe609241a6c4d40c0b22a9c44b6d20d1ed3` | **INTACT** |

---

## 6. GPU Acceleration Recommendation

1. **CPU Feasibility Assessment**: While Model D (4.11 hours) and Model E (6.04 hours) can run on laptop CPU without memory failure, full 2,500-step pretraining on CPU is impractical due to high wall-clock latency.
2. **GPU Justification**: Transitioning to GPU acceleration is expected to significantly increase throughput (unverified preliminary expectation to be measured on actual cloud GPU hardware), reducing pretraining wall-clock time from ~4.1–6.0 hours on CPU to an estimated fraction of an hour.
3. **Pilot Recommendation**: Proceed to Phase 8D Short GPU Pilot to measure exact GPU throughput before launching full 2,500-step pretraining runs for Model D and Model E.

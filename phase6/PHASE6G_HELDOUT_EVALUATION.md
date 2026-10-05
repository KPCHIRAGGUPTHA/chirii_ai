# Phase 6G: Held-Out Instruction Evaluation Report

## 1. Executive Summary
Phase 6G evaluates the fine-tuned Model C ([`checkpoints/phase6/model_c_sft/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/best_model.pt)) against the untouched Phase 5D baseline ([`checkpoints/phase5d/model_6_61m/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase5d/model_6_61m/best_model.pt)) on the completely held-out SFT test dataset (`phase6/data/sft/test_sft.jsonl`, 5,176 examples).

---

## 2. Checkpoint & Asset Hashes
- **Base Model SHA-256**: `6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433`
- **SFT Model SHA-256**: `14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26`
- **BPE Tokenizer SHA-256**: `6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d`

---

## 3. Quantitative Test-Set Loss & Perplexity Results

| Model Split | Evaluated Examples | Active Response Tokens | Test Loss | Test Perplexity |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 5D Baseline** | `5,176` | `275,393` | **`2.9912`** | **`19.9101`** |
| **Phase 6E SFT** | `5,176` | `275,393` | **`2.6084`** | **`13.5777`** |

### Metrics Improvement Analysis
- **Loss Improvement**: Absolute reduction of **`0.3828`** (**`12.80%`** relative improvement)
- **Perplexity Improvement**: Absolute reduction of **`6.3324`** (**`31.81%`** relative improvement)

---

## 4. 30 Fixed Prompt Generation Metrics

| Metric | Phase 5D Baseline | Phase 6E SFT | Delta Improvement |
| :--- | :--- | :--- | :--- |
| **Exact Match** | `0/30` | `0/30` | `0` |
| **Average Token Jaccard** | `0.0905` | `0.1059` | **`+0.0154`** |

---

## 5. Qualitative Generation Comparisons (5 Categories)

### Category: factual question (ID: 1)
- **Instruction**: What is the capital of France?
- **Input**: ``
- **Baseline Response**:
  ```text
  - Students (1999)
- Stephysical Students (1999)
-
  ```
- **SFT Response**:
  ```text
  The United States of States and American American American American American
  ```

---
### Category: explanation (ID: 4)
- **Instruction**: Explain how photosynthesis works in plants.
- **Input**: ``
- **Baseline Response**:
  ```text
  - Students are a few of his body of his both body to be a few of his
  ```
- **SFT Response**:
  ```text
  1. Start is a small of specific media is a small of specific metho
  ```

---
### Category: classification (ID: 12)
- **Instruction**: Classify the following words as either nouns or verbs: Work, Run, Book.
- **Input**: ``
- **Baseline Response**:
  ```text
  The Children: Students and Christians and Christian Christians and Chri
  ```
- **SFT Response**:
  ```text
  The sentence is a small of method of methods and the sentence of healthcare and he
  ```

---
### Category: simple calculation (ID: 17)
- **Instruction**: Calculate 15 multiplied by 4.
- **Input**: ``
- **Baseline Response**:
  ```text
  - Students (1999)
- See Controlll, 19999-1999-1
  ```
- **SFT Response**:
  ```text
  1. Controlling the sentence of 190 is a simple of 10 century."
2. St
  ```

---
### Category: instruction following (ID: 26)
- **Instruction**: Capitalize every word in the input text.
- **Input**: `learning artificial intelligence is exciting`
- **Baseline Response**:
  ```text
  The Conference of American American American American American American Am
  ```
- **SFT Response**:
  ```text
  The sentence is a small of media is a small of methods and intelligence and inter
  ```

---

## 6. Isolation & Integrity Verification
1. **Test Dataset**: `phase6/data/sft/test_sft.jsonl` was accessed strictly read-only.
2. **Execution Environment**: Evaluated strictly under `model.eval()` and `torch.no_grad()`. No optimizers or backward passes executed.
3. **File Integrity**: All model checkpoints and tokenizer files verified 100% byte-for-byte identical post-evaluation.

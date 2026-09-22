# Fixed Evaluation Prompts Specification

## 1. Overview
This specification details the fixed evaluation suite created for **Phase 6D: Model C Baseline Evaluation**.
These 30 instruction prompts will be used to measure model generation behavior **BEFORE SFT** (Phase 6D baseline) and re-evaluated **AFTER SFT** (Phase 6E) using identical decoding settings and formatting.

---

## 2. Fixed Prompt Categories & Schema

### Categories Covered (12 Categories)
1. **factual question** (3 prompts)
2. **explanation** (2 prompts)
3. **definition** (2 prompts)
4. **summarization** (2 prompts)
5. **rewriting** (2 prompts)
6. **classification** (3 prompts)
7. **reasoning** (2 prompts)
8. **simple calculation** (3 prompts)
9. **coding/programming** (2 prompts)
10. **list generation** (2 prompts)
11. **comparison** (2 prompts)
12. **instruction following** (5 prompts)

Total Prompts: **30**

### Individual Prompt Record Schema
Each prompt in [`phase6/evaluation/baseline_prompts.json`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/evaluation/baseline_prompts.json) adheres strictly to:

```json
{
  "id": 1,
  "category": "factual question",
  "instruction": "What is the capital of France?",
  "input": "",
  "expected_response": "The capital of France is Paris."
}
```

---

## 3. Formatting Strategy
Every evaluation prompt is rendered using `format_alpaca_prompt(instruction, input, "")`:

```text
Instruction:
<instruction>

Input:
<input or 'None'>

Response:
```

---

## 4. Generation Settings & Determinism
- **Device:** CPU / CUDA (deterministic execution)
- **Decoding Method:** Greedy / Deterministic (`temperature = 0.0`, fixed random seed `seed = 42`)
- **Max Generation Tokens:** `max_new_tokens = 40`
- **Context Boundary:** Truncated/capped at `block_size = 128`

---

## 5. Loss & Metrics Evaluation Methodology
1. **Teacher-Forced Response Loss:** Evaluated on all 5,176 test examples in [`phase6/data/sft/test_sft.jsonl`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/phase6/data/sft/test_sft.jsonl). Prompt tokens and padding positions are masked with `-100`.
2. **Perplexity:** Computed as $\exp(\text{average\_response\_loss})$.
3. **Diagnostic Quality Metrics:** Token count, exact match, token overlap Jaccard ratio.

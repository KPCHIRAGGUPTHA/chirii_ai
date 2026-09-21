# SFT Formatting & Response-Loss Masking Specification

## 1. Reusable Formatting Function
All Alpaca-style instruction records are formatted deterministically using:

```python
def format_alpaca_prompt(instruction: str, input_text: str, output_text: str) -> Tuple[str, str]:
    inst = instruction.strip()
    inp = input_text.strip()
    out = output_text.strip()
    inp_str = inp if inp else "None"

    prompt_prefix = f"Instruction:\\n{inst}\\n\\nInput:\\n{inp_str}\\n\\nResponse:\\n"
    full_text = f"{prompt_prefix}{out}"
    return prompt_prefix, full_text
```

---

## 2. Standard SFT Record Schema
Each prepared SFT record contains the following explicit fields:

```json
{
  "id": 0,
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet...\\n2. Exercise regularly...\\n3. Get enough sleep...",
  "formatted_text": "Instruction:\\nGive three tips for staying healthy.\\n\\nInput:\\nNone\\n\\nResponse:\\n1. Eat a balanced diet...",
  "tokens": [34, 182, 90, ...],
  "prompt_token_len": 42,
  "response_token_len": 48,
  "total_token_len": 90
}
```

---

## 3. Response / Label Masking Strategy (Target Loss Masking)
To ensure the autoregressive model learns primarily to generate high-quality **Response** text rather than memorizing prompt instructions, SFT data loaders will apply target label masking:

- **Input Sequence $X$:** Full token sequence `[t_0, t_1, ..., t_{N-1}]`.
- **Target Sequence $Y$:**
  - Tokens $0 \dots (\text{prompt\_token\_len} - 1)$ (corresponding to `Instruction:` and `Input:`) are set to **`-100`** (PyTorch `CrossEntropyLoss(ignore_index=-100)`).
  - Tokens $\text{prompt\_token\_len} \dots (N-1)$ (corresponding to `Response:` text) retain their target token IDs.

```text
Tokens:  [Inst_0, Inst_1, ..., Resp_Header, Out_0, Out_1, ..., EOS]
Labels:  [  -100,   -100, ...,        -100, Out_0, Out_1, ..., EOS]
```

---

## 4. Long-Example Handling Strategy Analysis (Phase 6C Proposal)

We evaluate 5 strategies for handling examples exceeding context length `block_size`:

1. **Strategy A (Hard Truncation):** Truncate raw formatted string at `block_size` tokens.
   - *Drawback:* May truncate output responses mid-sentence or drop response headers entirely.
2. **Strategy B (Response-Preserving Truncation):** Preserve prompt instruction intact; truncate response to fit remaining tokens.
   - *Drawback:* If prompt exceeds `block_size`, 0 response tokens remain.
3. **Strategy C (Prompt + Response Truncation):** Truncate long instruction/inputs to ensure at least $M$ tokens are reserved for the response.
   - *Benefit:* Guarantees response generation capacity even for verbose instructions.
4. **Strategy D (Chunking / Sliding Window):** Split long responses into multiple training windows.
   - *Drawback:* Adds complexity to single-turn instruction tuning.
5. **Strategy E (Concise Subset Filtering + Prompt-Response Truncation) [PROPOSED FOR PHASE 6C]:**
   - Filter training set to examples where total length is concise or apply prompt-response aware truncation so no training example contains truncated/corrupted prompt syntax.

import os
import sys
import json
import math
import hashlib
import random
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bpe_tokenizer import BPETokenizer

VOCAB_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")

global_tokenizer = None

def init_worker(vocab_path: str):
    global global_tokenizer
    global_tokenizer = BPETokenizer.load(vocab_path)

def get_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def format_alpaca_prompt(instruction: str, input_text: str, output_text: str) -> Tuple[str, str]:
    """
    Reusable formatting function for Alpaca-style SFT prompts.
    Returns:
        prompt_prefix: 'Instruction:\\n...\\n\\nInput:\\n...\\n\\nResponse:\\n'
        full_text:     'Instruction:\\n...\\n\\nInput:\\n...\\n\\nResponse:\\n<output>'
    """
    inst = instruction.strip()
    inp = input_text.strip()
    out = output_text.strip()
    inp_str = inp if inp else "None"

    prompt_prefix = f"Instruction:\n{inst}\n\nInput:\n{inp_str}\n\nResponse:\n"
    full_text = f"{prompt_prefix}{out}"
    return prompt_prefix, full_text

def process_sft_record(item_tuple: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    global global_tokenizer
    idx, item = item_tuple

    inst = item.get("instruction", "")
    inp = item.get("input", "")
    out = item.get("output", "")

    prompt_prefix, full_text = format_alpaca_prompt(inst, inp, out)

    prefix_tokens = global_tokenizer.encode(prompt_prefix)
    full_tokens = global_tokenizer.encode(full_text)

    prompt_len = len(prefix_tokens)
    total_len = len(full_tokens)
    response_len = max(0, total_len - prompt_len)

    return {
        "id": idx,
        "instruction": inst,
        "input": inp,
        "output": out,
        "formatted_text": full_text,
        "tokens": full_tokens,
        "prompt_token_len": prompt_len,
        "response_token_len": response_len,
        "total_token_len": total_len
    }

def calculate_percentiles(values: List[Any]) -> Dict[str, float]:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}

    def get_pct(p):
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_vals[int(k)])
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return float(d0 + d1)

    return {
        "min": int(sorted_vals[0]),
        "max": int(sorted_vals[-1]),
        "mean": round(float(sum(sorted_vals) / n), 2),
        "median": round(float(get_pct(50)), 2),
        "p75": round(float(get_pct(75)), 2),
        "p90": round(float(get_pct(90)), 2),
        "p95": round(float(get_pct(95)), 2),
        "p99": round(float(get_pct(99)), 2),
    }

def main():
    print("=== Phase 6B: SFT Dataset Preparation & Context Experiment ===", flush=True)

    # 1. Integrity Verification of Pre-existing Files
    print(f"Verifying integrity of existing tokenizer ({VOCAB_PATH})...", flush=True)
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Tokenizer file missing: {VOCAB_PATH}")
    vocab_hash_before = get_file_sha256(VOCAB_PATH)
    vocab_size_bytes_before = os.path.getsize(VOCAB_PATH)

    print(f"Verifying integrity of Model C checkpoint ({CHECKPOINT_PATH})...", flush=True)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model C checkpoint missing: {CHECKPOINT_PATH}")
    ckpt_hash_before = get_file_sha256(CHECKPOINT_PATH)
    ckpt_size_bytes_before = os.path.getsize(CHECKPOINT_PATH)

    # 2. Output Directories
    phase6_dir = os.path.join(REPO_ROOT, "phase6")
    audit_dir = os.path.join(phase6_dir, "audit")
    prep_dir = os.path.join(phase6_dir, "preparation")
    sft_data_dir = os.path.join(phase6_dir, "data", "sft")

    os.makedirs(audit_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(sft_data_dir, exist_ok=True)

    # 3. Load Phase 6A Processed Dataset
    processed_jsonl = os.path.join(phase6_dir, "data", "processed", "alpaca_cleaned_processed.jsonl")
    if not os.path.exists(processed_jsonl):
        raise FileNotFoundError(f"Phase 6A processed dataset missing at {processed_jsonl}")

    print(f"Loading Phase 6A processed dataset from {processed_jsonl}...", flush=True)
    raw_records = []
    with open(processed_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    total_records = len(raw_records)
    print(f"Loaded {total_records:,} deduplicated records.", flush=True)

    # 4. Tokenize & Structure SFT Records
    print(f"Structuring SFT records with prompt/response token lengths...", flush=True)
    valid_tuples = [(i, r) for i, r in enumerate(raw_records)]
    sft_records = []
    with ProcessPoolExecutor(initializer=init_worker, initargs=(VOCAB_PATH,)) as executor:
        for rec in executor.map(process_sft_record, valid_tuples, chunksize=200):
            sft_records.append(rec)

    # 5. Save SFT Processed Dataset
    sft_processed_path = os.path.join(sft_data_dir, "sft_processed.jsonl")
    with open(sft_processed_path, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved SFT processed dataset to {sft_processed_path}", flush=True)

    # 6. Empirical Context-Length Experiment (128 vs 256 vs 512 tokens)
    print("\nExecuting Empirical Context-Length Experiment...", flush=True)
    total_original_tokens = sum(r["total_token_len"] for r in sft_records)
    token_lengths = [r["total_token_len"] for r in sft_records]

    context_scenarios = [128, 256, 512]
    experiment_results = {}

    for L in context_scenarios:
        fitting_count = sum(1 for t in token_lengths if t <= L)
        exceeding_count = total_records - fitting_count
        pct_fitting = round((fitting_count / total_records) * 100, 2)
        pct_truncation = round((exceeding_count / total_records) * 100, 2)

        retained_per_item = [min(t, L) for t in token_lengths]
        total_retained = sum(retained_per_item)
        content_lost_tokens = total_original_tokens - total_retained
        content_lost_pct = round((content_lost_tokens / total_original_tokens) * 100, 2)

        avg_retained = round(total_retained / total_records, 2)
        sorted_retained = sorted(retained_per_item)
        median_retained = sorted_retained[total_records // 2]
        capacity_utilization_pct = round((total_retained / (total_records * L)) * 100, 2)

        experiment_results[str(L)] = {
            "context_length": L,
            "total_examples": total_records,
            "fitting_count": fitting_count,
            "exceeding_count": exceeding_count,
            "percentage_fitting": pct_fitting,
            "percentage_requiring_truncation": pct_truncation,
            "total_original_tokens": total_original_tokens,
            "tokens_retained_after_truncation": total_retained,
            "average_retained_tokens": avg_retained,
            "median_retained_tokens": median_retained,
            "content_lost_tokens": content_lost_tokens,
            "content_lost_percentage": content_lost_pct,
            "capacity_utilization_pct": capacity_utilization_pct
        }

        print(f"--- Context Length {L} ---")
        print(f"  Fitting <= {L}: {fitting_count:,} ({pct_fitting}%)")
        print(f"  Exceeding > {L}: {exceeding_count:,} ({pct_truncation}%)")
        print(f"  Content Lost: {content_lost_tokens:,} tokens ({content_lost_pct}%)")
        print(f"  Average Retained: {avg_retained} tokens")

    # Save Experiment JSON
    exp_json_path = os.path.join(audit_dir, "context_length_comparison.json")
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"Saved context length comparison JSON to {exp_json_path}", flush=True)

    # 7. Partition SFT Splits (matching Phase 6A fixed seed 42)
    random.seed(42)
    shuffled_sft = list(sft_records)
    random.shuffle(shuffled_sft)

    train_end = int(0.80 * total_records)
    val_end = int(0.90 * total_records)

    train_sft = shuffled_sft[:train_end]
    val_sft = shuffled_sft[train_end:val_end]
    test_sft = shuffled_sft[val_end:]

    for name, data in [("train_sft.jsonl", train_sft), ("val_sft.jsonl", val_sft), ("test_sft.jsonl", test_sft)]:
        path = os.path.join(sft_data_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 8. Compute SFT Token Length Distributions
    prompt_lens = [r["prompt_token_len"] for r in sft_records]
    resp_lens = [r["response_token_len"] for r in sft_records]
    total_lens = [r["total_token_len"] for r in sft_records]

    prompt_stats = calculate_percentiles(prompt_lens)
    resp_stats = calculate_percentiles(resp_lens)
    total_stats = calculate_percentiles(total_lens)

    # 9. Verify Post-Execution Integrity of Key Files
    vocab_hash_after = get_file_sha256(VOCAB_PATH)
    ckpt_hash_after = get_file_sha256(CHECKPOINT_PATH)

    assert vocab_hash_before == vocab_hash_after, "CRITICAL ERROR: Tokenizer file was modified!"
    assert ckpt_hash_before == ckpt_hash_after, "CRITICAL ERROR: Model C checkpoint was modified!"
    print("\nPost-execution integrity verified: Tokenizer and Checkpoint files are 100% UNCHANGED.", flush=True)

    # 10. Generate Markdown Reports & Specs
    generate_context_markdown(experiment_results, os.path.join(audit_dir, "CONTEXT_LENGTH_COMPARISON.md"))
    generate_sft_format_spec(os.path.join(prep_dir, "sft_format_spec.md"))
    generate_master_phase6b_report(
        experiment_results,
        prompt_stats,
        resp_stats,
        total_stats,
        vocab_hash_after,
        ckpt_hash_after,
        os.path.join(phase6_dir, "PHASE6B_DATASET_PREPARATION.md")
    )

def generate_context_markdown(exp_results: Dict[str, Any], output_path: str):
    c128 = exp_results["128"]
    c256 = exp_results["256"]
    c512 = exp_results["512"]

    md = f"""# Context-Length Experiment Report (128 vs 256 vs 512 Tokens)

## 1. Executive Summary & Objective
This empirical study evaluates the trade-offs of sequence context length for Supervised Fine-Tuning (SFT) of MiniGPT Model C (6.61M parameters). We evaluate three context length candidates: **128 tokens** (baseline Model C `block_size`), **256 tokens**, and **512 tokens** across all 51,756 deduplicated records of `yahma/alpaca-cleaned` using the existing 1,024-vocabulary BPE Tokenizer.

> [!IMPORTANT]
> **Analysis-Only Scenarios:** Context lengths of 256 and 512 tokens are evaluated strictly for analytical comparison. Model C's pretrained `block_size=128` and checkpoint remain 100% untouched in Phase 6B.

---

## 2. Empirical Comparison Matrix

| Metric | Context = 128 (Baseline) | Context = 256 | Context = 512 |
| :--- | :--- | :--- | :--- |
| **Total Dataset Examples** | 51,756 | 51,756 | 51,756 |
| **Examples Fitting Completely** | {c128['fitting_count']:,} | {c256['fitting_count']:,} | {c512['fitting_count']:,} |
| **Examples Exceeding Context** | {c128['exceeding_count']:,} | {c256['exceeding_count']:,} | {c512['exceeding_count']:,} |
| **Percentage Fitting Completely** | **{c128['percentage_fitting']}%** | **{c256['percentage_fitting']}%** | **{c512['percentage_fitting']}%** |
| **Percentage Requiring Truncation** | **{c128['percentage_requiring_truncation']}%** | **{c256['percentage_requiring_truncation']}%** | **{c512['percentage_requiring_truncation']}%** |
| **Tokens Retained After Truncation** | {c128['tokens_retained_after_truncation']:,} | {c256['tokens_retained_after_truncation']:,} | {c512['tokens_retained_after_truncation']:,} |
| **Content Lost (Tokens)** | {c128['content_lost_tokens']:,} | {c256['content_lost_tokens']:,} | {c512['content_lost_tokens']:,} |
| **Estimated Content Loss (%)** | **{c128['content_lost_percentage']}%** | **{c256['content_lost_percentage']}%** | **{c512['content_lost_percentage']}%** |
| **Average Retained Tokens / Item** | {c128['average_retained_tokens']} | {c256['average_retained_tokens']} | {c512['average_retained_tokens']} |
| **Median Retained Tokens / Item** | {c128['median_retained_tokens']} | {c256['median_retained_tokens']} | {c512['median_retained_tokens']} |
| **Fixed-Slot Capacity Utilization** | {c128['capacity_utilization_pct']}% | {c256['capacity_utilization_pct']}% | {c512['capacity_utilization_pct']}% |

---

## 3. Trade-Off Analysis

### A. Context = 128 Tokens (Model C Native Baseline)
- **Coverage & Loss:** Fits 25.03% of dataset items completely; incurs **{c128['content_lost_percentage']}%** overall token content loss under hard truncation.
- **Architectural Impact:** 100% compatible with Model C (`checkpoints/phase5d/model_6_61m/best_model.pt`). Requires zero changes to positional embedding weights `wpe` (shape `(128, 256)`).
- **Compute Efficiency:** Fastest iteration speed, lowest memory consumption.

### B. Context = 256 Tokens
- **Coverage & Loss:** Fits 43.64% of dataset items completely; reduces token content loss to **{c256['content_lost_percentage']}%**.
- **Architectural Impact:** Requires extending `block_size` to 256 and modifying/interpolating positional embeddings `wpe`. Cannot be used without model architecture adjustment.

### C. Context = 512 Tokens
- **Coverage & Loss:** Fits 67.38% of dataset items completely; reduces token content loss to **{c512['content_lost_percentage']}%**.
- **Architectural Impact:** 4x higher quadratic self-attention memory and compute complexity ($O(N^2)$). Requires extending positional embeddings up to index 511.

---

## 4. Conclusion & Recommendation
While 512 tokens preserves {100 - c512['content_lost_percentage']:.2f}% of total dataset token content, larger context lengths cannot be safely used without modifying Model C's architecture (`block_size`) and retraining positional embeddings.

Therefore, for Phase 6B, Model C's baseline `block_size = 128` remains untouched. A **Concise Subset Filtering + Prompt-Response Aware Truncation** strategy is proposed for Phase 6C evaluation.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")
    print(f"Generated {output_path}", flush=True)

def generate_sft_format_spec(output_path: str):
    md = r"""# SFT Formatting & Response-Loss Masking Specification

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
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")
    print(f"Generated {output_path}", flush=True)

def generate_master_phase6b_report(
    exp_results: Dict[str, Any],
    prompt_stats: Dict[str, Any],
    resp_stats: Dict[str, Any],
    total_stats: Dict[str, Any],
    vocab_hash: str,
    ckpt_hash: str,
    output_path: str
):
    c128 = exp_results["128"]
    c256 = exp_results["256"]
    c512 = exp_results["512"]

    md = f"""# Phase 6B: Dataset Preparation & Context-Length Experiment Report

## 1. Executive Summary
Phase 6B completed the empirical context-length evaluation and SFT dataset preparation for the MiniGPT project without performing any model training or modifying model checkpoints/tokenizer files.

---

## 2. Empirical Context-Length Experiment Results

### Context = 128 Tokens (Model C Baseline `block_size`)
- **Fitting Completely (`<= 128`):** {c128['fitting_count']:,} ({c128['percentage_fitting']}%)
- **Exceeding Context (`> 128`):** {c128['exceeding_count']:,} ({c128['percentage_requiring_truncation']}%)
- **Tokens Retained:** {c128['tokens_retained_after_truncation']:,} / {c128['total_original_tokens']:,}
- **Estimated Content Loss:** **{c128['content_lost_percentage']}%** ({c128['content_lost_tokens']:,} tokens lost)
- **Average Retained Tokens:** {c128['average_retained_tokens']}
- **Median Retained Tokens:** {c128['median_retained_tokens']}

### Context = 256 Tokens (Analysis-Only Scenario)
- **Fitting Completely (`<= 256`):** {c256['fitting_count']:,} ({c256['percentage_fitting']}%)
- **Exceeding Context (`> 256`):** {c256['exceeding_count']:,} ({c256['percentage_requiring_truncation']}%)
- **Tokens Retained:** {c256['tokens_retained_after_truncation']:,} / {c256['total_original_tokens']:,}
- **Estimated Content Loss:** **{c256['content_lost_percentage']}%** ({c256['content_lost_tokens']:,} tokens lost)
- **Average Retained Tokens:** {c256['average_retained_tokens']}
- **Median Retained Tokens:** {c256['median_retained_tokens']}

### Context = 512 Tokens (Analysis-Only Scenario)
- **Fitting Completely (`<= 512`):** {c512['fitting_count']:,} ({c512['percentage_fitting']}%)
- **Exceeding Context (`> 512`):** {c512['exceeding_count']:,} ({c512['percentage_requiring_truncation']}%)
- **Tokens Retained:** {c512['tokens_retained_after_truncation']:,} / {c512['total_original_tokens']:,}
- **Estimated Content Loss:** **{c512['content_lost_percentage']}%** ({c512['content_lost_tokens']:,} tokens lost)
- **Average Retained Tokens:** {c512['average_retained_tokens']}
- **Median Retained Tokens:** {c512['median_retained_tokens']}

---

## 3. SFT Formatting Statistics (1024-Vocab BPE)

| Metric | Prompt Token Length | Response Token Length | Total Token Length |
| :--- | :--- | :--- | :--- |
| **Min Tokens** | {prompt_stats['min']} | {resp_stats['min']} | {total_stats['min']} |
| **Median (p50)** | {prompt_stats['median']} | {resp_stats['median']} | {total_stats['median']} |
| **Mean Tokens** | {prompt_stats['mean']} | {resp_stats['mean']} | {total_stats['mean']} |
| **75th Percentile (p75)** | {prompt_stats['p75']} | {resp_stats['p75']} | {total_stats['p75']} |
| **90th Percentile (p90)** | {prompt_stats['p90']} | {resp_stats['p90']} | {total_stats['p90']} |
| **95th Percentile (p95)** | {prompt_stats['p95']} | {resp_stats['p95']} | {total_stats['p95']} |
| **Max Tokens** | {prompt_stats['max']} | {resp_stats['max']} | {total_stats['max']} |

---

## 4. Context Length Selection & Recommendation
- **Current Model C Baseline:** `block_size = 128` remains unchanged.
- **Reasoning:** Extending sequence length to 256 or 512 tokens would require modifying Model C's architecture and extending positional embedding weights `wpe`.
- **Phase 6C Proposal:** Evaluate "Concise Subset Filtering + Prompt-Response Aware Truncation" to retain complete response completions during SFT.

---

## 5. Target Response-Loss Masking Strategy
- Prompt instruction tokens (`Instruction:` and `Input:`) up to `Response:` header: Masked with `-100`.
- Response target tokens (`output` text): Retain target token IDs for computing cross-entropy loss.

---

## 6. Pre-existing File Verification Summary
- **Tokenizer File (`bpe_vocab_1024.json`):** Verified SHA-256 (`{vocab_hash[:16]}...`) — **UNCHANGED**.
- **Model C Checkpoint (`best_model.pt`):** Verified SHA-256 (`{ckpt_hash[:16]}...`) — **UNCHANGED**.
- **Model Training / Fine-Tuning:** **NONE PERFORMED**.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")
    print(f"Generated {output_path}", flush=True)

if __name__ == "__main__":
    main()

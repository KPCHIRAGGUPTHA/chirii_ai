import os
import sys
import json
import math
import hashlib
import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT
from bpe_tokenizer import BPETokenizer
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt
from phase6.training.sft_dataset import create_sft_dataloader

BASE_CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
SFT_CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
TEST_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "test_sft.jsonl")
PROMPTS_PATH = os.path.join(REPO_ROOT, "phase6", "evaluation", "baseline_prompts.json")
EVAL_DIR = os.path.join(REPO_ROOT, "phase6", "evaluation")

EXPECTED_BASE_SHA = "6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433"
EXPECTED_SFT_SHA = "14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26"
EXPECTED_TOK_SHA = "6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d"

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_token_jaccard(gen_tokens: List[int], exp_tokens: List[int]) -> float:
    set_g = set(gen_tokens)
    set_e = set(exp_tokens)
    if not set_g and not set_e:
        return 1.0
    intersection = set_g.intersection(set_e)
    union = set_g.union(set_e)
    return float(len(intersection) / len(union)) if union else 0.0

def evaluate_model_loss(model: MiniGPT, test_dataloader) -> Tuple[float, float, int, int]:
    total_loss_sum = 0.0
    total_active_tokens = 0
    total_examples_eval = 0

    model.eval()
    with torch.no_grad():
        for input_seq, labels in test_dataloader:
            logits, _ = model(input_seq, labels)
            vocab_size = logits.size(-1)

            loss_flat = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                reduction="none",
                ignore_index=-100
            )
            active_mask = (labels.view(-1) != -100)

            total_loss_sum += loss_flat.sum().item()
            total_active_tokens += active_mask.sum().item()
            total_examples_eval += input_seq.size(0)

    avg_loss = total_loss_sum / total_active_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, total_active_tokens, total_examples_eval

def generate_responses_for_prompts(model: MiniGPT, tokenizer: BPETokenizer, fixed_prompts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, float]:
    set_seed(42)
    generations_records = []

    for item in fixed_prompts:
        p_id = item["id"]
        category = item["category"]
        inst = item["instruction"]
        inp = item["input"]
        exp_resp = item["expected_response"]

        prompt_prefix, _ = format_alpaca_prompt(inst, inp, "")
        prompt_tokens = tokenizer.encode(prompt_prefix)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

        with torch.no_grad():
            out_ids = model.generate(input_ids, max_new_tokens=40, temperature=0.0)

        gen_tokens = out_ids[0].tolist()
        resp_tokens = gen_tokens[len(prompt_tokens):]

        full_generated_text = tokenizer.decode(gen_tokens)
        generated_response = tokenizer.decode(resp_tokens).strip()

        expected_tokens = tokenizer.encode(exp_resp)
        exact_match = (generated_response == exp_resp.strip())
        token_jaccard = compute_token_jaccard(resp_tokens, expected_tokens)

        gen_rec = {
            "id": p_id,
            "category": category,
            "instruction": inst,
            "input": inp,
            "expected_response": exp_resp,
            "prompt_prefix": prompt_prefix,
            "prompt_token_count": len(prompt_tokens),
            "full_generated_text": full_generated_text,
            "generated_response": generated_response,
            "generated_token_count": len(resp_tokens),
            "exact_match": exact_match,
            "token_jaccard_overlap": round(token_jaccard, 4)
        }
        generations_records.append(gen_rec)

    exact_matches_count = sum(1 for r in generations_records if r["exact_match"])
    jaccard_scores = [r["token_jaccard_overlap"] for r in generations_records]
    avg_jaccard = round(float(sum(jaccard_scores) / len(jaccard_scores)), 4)

    return generations_records, exact_matches_count, avg_jaccard

def main():
    print("=== PHASE 6G — HELD-OUT INSTRUCTION EVALUATION ===", flush=True)

    # 1. Pre-execution SHA Verification
    base_sha_before = compute_sha256(BASE_CKPT_PATH)
    sft_sha_before = compute_sha256(SFT_CKPT_PATH)
    tok_sha_before = compute_sha256(TOKENIZER_PATH)

    print(f"Base Checkpoint SHA: {base_sha_before}")
    print(f"SFT Checkpoint SHA:  {sft_sha_before}")
    print(f"Tokenizer SHA:       {tok_sha_before}")

    assert base_sha_before == EXPECTED_BASE_SHA, f"Base SHA mismatch! Got {base_sha_before}"
    assert sft_sha_before == EXPECTED_SFT_SHA, f"SFT SHA mismatch! Got {sft_sha_before}"
    assert tok_sha_before == EXPECTED_TOK_SHA, f"Tokenizer SHA mismatch! Got {tok_sha_before}"

    # 2. Tokenizer & Test Data Verification
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    assert tokenizer.vocab_size == 1024

    test_records = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))
    assert len(test_records) == 5176, f"Expected 5,176 test examples, got {len(test_records)}"
    print(f"Test dataset loaded: {len(test_records):,} examples (strictly read-only)", flush=True)

    test_dataloader = create_sft_dataloader(TEST_SET_PATH, batch_size=64, block_size=128, shuffle=False)

    # 3. Load & Evaluate Baseline Model A
    print("\n--- Evaluating Baseline Model (Phase 5D Model C) ---", flush=True)
    base_ckpt = torch.load(BASE_CKPT_PATH, map_location="cpu", weights_only=False)
    base_model = MiniGPT(base_ckpt["config"])
    base_model.load_state_dict(base_ckpt["model_state"])
    base_model.eval()

    base_loss, base_ppl, active_tokens_base, examples_base = evaluate_model_loss(base_model, test_dataloader)
    print(f"Baseline Test Loss: {base_loss:.4f} | Baseline Test PPL: {base_ppl:.4f}")

    # 4. Load & Evaluate SFT Model B
    print("\n--- Evaluating Fine-Tuned Model (Phase 6E SFT) ---", flush=True)
    sft_ckpt = torch.load(SFT_CKPT_PATH, map_location="cpu", weights_only=False)
    sft_model = MiniGPT(sft_ckpt["config"])
    sft_model.load_state_dict(sft_ckpt["model_state_dict"])
    sft_model.eval()

    sft_loss, sft_ppl, active_tokens_sft, examples_sft = evaluate_model_loss(sft_model, test_dataloader)
    print(f"SFT Test Loss:      {sft_loss:.4f} | SFT Test PPL:      {sft_ppl:.4f}")

    # Calculate Loss & PPL Improvements
    abs_loss_imp = base_loss - sft_loss
    rel_loss_imp_pct = ((base_loss - sft_loss) / base_loss) * 100.0

    abs_ppl_imp = base_ppl - sft_ppl
    rel_ppl_imp_pct = ((base_ppl - sft_ppl) / base_ppl) * 100.0

    # 5. Fixed Prompts Generation Evaluation (30 Prompts)
    print("\n--- Executing 30 Fixed Prompts Generation Evaluation ---", flush=True)
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        fixed_prompts = json.load(f)

    base_gens, base_exact, base_jaccard = generate_responses_for_prompts(base_model, tokenizer, fixed_prompts)
    sft_gens, sft_exact, sft_jaccard = generate_responses_for_prompts(sft_model, tokenizer, fixed_prompts)

    print(f"Baseline Generation: Exact Match = {base_exact}/30 | Token Jaccard = {base_jaccard:.4f}")
    print(f"SFT Generation:      Exact Match = {sft_exact}/30 | Token Jaccard = {sft_jaccard:.4f}")

    # Save generation JSON artifacts
    base_gen_path = os.path.join(EVAL_DIR, "phase6g_baseline_generations.json")
    sft_gen_path = os.path.join(EVAL_DIR, "phase6g_sft_generations.json")

    with open(base_gen_path, "w", encoding="utf-8") as f:
        json.dump(base_gens, f, indent=2, ensure_ascii=False)
    with open(sft_gen_path, "w", encoding="utf-8") as f:
        json.dump(sft_gens, f, indent=2, ensure_ascii=False)

    # 6. Post-execution SHA Verification
    base_sha_after = compute_sha256(BASE_CKPT_PATH)
    sft_sha_after = compute_sha256(SFT_CKPT_PATH)
    tok_sha_after = compute_sha256(TOKENIZER_PATH)

    assert base_sha_before == base_sha_after
    assert sft_sha_before == sft_sha_after
    assert tok_sha_before == tok_sha_after
    print("\nPost-evaluation SHA verification: 100% BYTE-FOR-BYTE IDENTICAL.")

    # 7. Qualitative Samples Selection (5 Categories)
    selected_prompt_ids = [1, 4, 12, 17, 26]
    qualitative_samples = []

    for target_id in selected_prompt_ids:
        for bg, sg in zip(base_gens, sft_gens):
            if bg["id"] == target_id:
                qualitative_samples.append({
                    "id": bg["id"],
                    "category": bg["category"],
                    "instruction": bg["instruction"],
                    "input": bg["input"],
                    "baseline_response": bg["generated_response"],
                    "sft_response": sg["generated_response"]
                })
                break

    # Save master phase6g_results.json
    results_data = {
        "test_examples": len(test_records),
        "active_response_tokens": active_tokens_sft,
        "baseline_metrics": {
            "checkpoint": "checkpoints/phase5d/model_6_61m/best_model.pt",
            "sha256": base_sha_after,
            "test_loss": round(base_loss, 4),
            "test_ppl": round(base_ppl, 4),
            "exact_match": f"{base_exact}/30",
            "token_jaccard": round(base_jaccard, 4)
        },
        "sft_metrics": {
            "checkpoint": "checkpoints/phase6/model_c_sft/best_model.pt",
            "sha256": sft_sha_after,
            "test_loss": round(sft_loss, 4),
            "test_ppl": round(sft_ppl, 4),
            "exact_match": f"{sft_exact}/30",
            "token_jaccard": round(sft_jaccard, 4)
        },
        "improvements": {
            "loss": {
                "absolute": round(abs_loss_imp, 4),
                "percentage": round(rel_loss_imp_pct, 2)
            },
            "perplexity": {
                "absolute": round(abs_ppl_imp, 4),
                "percentage": round(rel_ppl_imp_pct, 2)
            },
            "generation": {
                "exact_match_diff": sft_exact - base_exact,
                "token_jaccard_diff": round(sft_jaccard - base_jaccard, 4)
            }
        },
        "tokenizer_sha256": tok_sha_after,
        "isolation_checks": {
            "test_set_modified": False,
            "training_performed": False,
            "optimizer_used": False,
            "checkpoint_integrity": True
        },
        "qualitative_samples": qualitative_samples
    }

    results_json_path = os.path.join(EVAL_DIR, "phase6g_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"Saved master Phase 6G results to {results_json_path}")

    # Generate Markdown Report
    md_content = f"""# Phase 6G: Held-Out Instruction Evaluation Report

## 1. Executive Summary
Phase 6G evaluates the fine-tuned Model C ([`checkpoints/phase6/model_c_sft/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase6/model_c_sft/best_model.pt)) against the untouched Phase 5D baseline ([`checkpoints/phase5d/model_6_61m/best_model.pt`](file:///c:/Users/dell9/OneDrive/Desktop/minigpt/checkpoints/phase5d/model_6_61m/best_model.pt)) on the completely held-out SFT test dataset (`phase6/data/sft/test_sft.jsonl`, 5,176 examples).

---

## 2. Checkpoint & Asset Hashes
- **Base Model SHA-256**: `{base_sha_after}`
- **SFT Model SHA-256**: `{sft_sha_after}`
- **BPE Tokenizer SHA-256**: `{tok_sha_after}`

---

## 3. Quantitative Test-Set Loss & Perplexity Results

| Model Split | Evaluated Examples | Active Response Tokens | Test Loss | Test Perplexity |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 5D Baseline** | `{examples_base:,}` | `{active_tokens_base:,}` | **`{base_loss:.4f}`** | **`{base_ppl:.4f}`** |
| **Phase 6E SFT** | `{examples_sft:,}` | `{active_tokens_sft:,}` | **`{sft_loss:.4f}`** | **`{sft_ppl:.4f}`** |

### Metrics Improvement Analysis
- **Loss Improvement**: Absolute reduction of **`{abs_loss_imp:.4f}`** (**`{rel_loss_imp_pct:.2f}%`** relative improvement)
- **Perplexity Improvement**: Absolute reduction of **`{abs_ppl_imp:.4f}`** (**`{rel_ppl_imp_pct:.2f}%`** relative improvement)

---

## 4. 30 Fixed Prompt Generation Metrics

| Metric | Phase 5D Baseline | Phase 6E SFT | Delta Improvement |
| :--- | :--- | :--- | :--- |
| **Exact Match** | `{base_exact}/30` | `{sft_exact}/30` | `{sft_exact - base_exact}` |
| **Average Token Jaccard** | `{base_jaccard:.4f}` | `{sft_jaccard:.4f}` | **`+{(sft_jaccard - base_jaccard):.4f}`** |

---

## 5. Qualitative Generation Comparisons (5 Categories)

"""

    for sample in qualitative_samples:
        md_content += f"""### Category: {sample['category']} (ID: {sample['id']})
- **Instruction**: {sample['instruction']}
- **Input**: `{sample['input']}`
- **Baseline Response**:
  ```text
  {sample['baseline_response']}
  ```
- **SFT Response**:
  ```text
  {sample['sft_response']}
  ```

---
"""

    md_content += """
## 6. Isolation & Integrity Verification
1. **Test Dataset**: `phase6/data/sft/test_sft.jsonl` was accessed strictly read-only.
2. **Execution Environment**: Evaluated strictly under `model.eval()` and `torch.no_grad()`. No optimizers or backward passes executed.
3. **File Integrity**: All model checkpoints and tokenizer files verified 100% byte-for-byte identical post-evaluation.
"""

    md_report_path = os.path.join(REPO_ROOT, "phase6", "PHASE6G_HELDOUT_EVALUATION.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Phase 6G markdown report to {md_report_path}")

if __name__ == "__main__":
    main()

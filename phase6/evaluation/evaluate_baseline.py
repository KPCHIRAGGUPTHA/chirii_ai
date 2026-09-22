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

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt
from phase6.training.sft_dataset import SFTDataset, create_sft_dataloader

CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
TEST_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "test_sft.jsonl")
TRAIN_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "train_sft.jsonl")
VAL_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "val_sft.jsonl")
PROMPTS_PATH = os.path.join(REPO_ROOT, "phase6", "evaluation", "baseline_prompts.json")

EVAL_DIR = os.path.join(REPO_ROOT, "phase6", "evaluation")

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

def run_baseline_evaluation() -> Dict[str, Any]:
    print("=== PHASE 6D — MODEL C BASELINE EVALUATION ===", flush=True)

    # 1. Pre-execution Hashes
    ckpt_hash_before = compute_sha256(CHECKPOINT_PATH)
    tok_hash_before = compute_sha256(TOKENIZER_PATH)
    print(f"Model C Checkpoint SHA-256 (Before): {ckpt_hash_before}", flush=True)
    print(f"Tokenizer SHA-256 (Before): {tok_hash_before}", flush=True)

    # 2. Verify Base Model Checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config: MiniGPTConfig = checkpoint["config"]

    assert config.vocab_size == 1024, f"Expected vocab_size 1024, got {config.vocab_size}"
    assert config.block_size == 128, f"Expected block_size 128, got {config.block_size}"
    assert config.n_layer == 8, f"Expected n_layer 8, got {config.n_layer}"
    assert config.n_head == 8, f"Expected n_head 8, got {config.n_head}"
    assert config.n_embd == 256, f"Expected n_embd 256, got {config.n_embd}"

    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    num_params = model.get_num_params()
    assert num_params == 6613504, f"Expected parameter count 6,613,504, got {num_params}"

    model.eval()
    print(f"Loaded Model C: {num_params:,} parameters, n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}", flush=True)

    # 3. Verify Tokenizer
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    assert tokenizer.vocab_size == 1024, f"Expected tokenizer vocab size 1024, got {tokenizer.vocab_size}"
    test_str = "Instruction: Test tokenization works.\nResponse:\nOk."
    enc = tokenizer.encode(test_str)
    dec = tokenizer.decode(enc)
    assert len(enc) > 0 and isinstance(dec, str), "Tokenizer encode/decode verification failed!"
    print("Tokenizer encoding/decoding verified successfully.", flush=True)

    # 4. Verify Test Dataset Split & Integrity
    print(f"Loading test set from {TEST_SET_PATH}...", flush=True)
    test_records = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    assert len(test_records) == 5176, f"Expected 5,176 test examples, got {len(test_records)}"

    # Integrity checks: duplicates, missing responses, leakage
    test_formatted_set = set()
    missing_responses = 0
    for r in test_records:
        fmt = r.get("formatted_text", "")
        out = r.get("output", "").strip()
        if not out:
            missing_responses += 1
        test_formatted_set.add(fmt)

    assert missing_responses == 0, f"Found {missing_responses} missing responses in test set!"
    assert len(test_formatted_set) == len(test_records), "Found duplicate examples in test set!"

    # Cross-split leakage check against train and val
    train_formatted_set = set()
    with open(TRAIN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_formatted_set.add(json.loads(line).get("formatted_text", ""))

    val_formatted_set = set()
    with open(VAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_formatted_set.add(json.loads(line).get("formatted_text", ""))

    train_leakage = len(test_formatted_set.intersection(train_formatted_set))
    val_leakage = len(test_formatted_set.intersection(val_formatted_set))
    assert train_leakage == 0, f"Detected {train_leakage} cross-split leakage examples with train set!"
    assert val_leakage == 0, f"Detected {val_leakage} cross-split leakage examples with val set!"

    print("Test set integrity verified: 5,176 examples, 0 duplicates, 0 missing responses, 0 leakage.", flush=True)

    # 5. Teacher-Forced Baseline Test-Set Loss Calculation
    print("Calculating baseline response loss on test dataset...", flush=True)
    test_dataloader = create_sft_dataloader(TEST_SET_PATH, batch_size=64, block_size=128, shuffle=False)

    total_loss_sum = 0.0
    total_active_tokens = 0
    total_examples_eval = 0

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

    avg_response_loss = total_loss_sum / total_active_tokens
    perplexity = math.exp(avg_response_loss)

    print(f"Test Loss Evaluation Complete: Evaluated {total_examples_eval:,} examples.", flush=True)
    print(f"Total Active Response Tokens: {total_active_tokens:,}", flush=True)
    print(f"Baseline Response Loss: {avg_response_loss:.4f}", flush=True)
    print(f"Baseline Perplexity: {perplexity:.4f}", flush=True)

    # 6. Load Fixed Evaluation Prompts & Execute Pre-SFT Generation
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        fixed_prompts = json.load(f)

    print(f"Loaded {len(fixed_prompts)} fixed evaluation prompts.", flush=True)

    set_seed(42)
    generations_records = []
    manual_samples = []

    for item in fixed_prompts:
        p_id = item["id"]
        category = item["category"]
        inst = item["instruction"]
        inp = item["input"]
        exp_resp = item["expected_response"]

        prompt_prefix, _ = format_alpaca_prompt(inst, inp, "")
        prompt_tokens = tokenizer.encode(prompt_prefix)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)

        # Deterministic generation (seed 42, temperature 0.0)
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
            "generation_settings": {
                "decoding": "deterministic greedy",
                "temperature": 0.0,
                "seed": 42,
                "max_new_tokens": 40
            },
            "exact_match": exact_match,
            "token_jaccard_overlap": round(token_jaccard, 4)
        }
        generations_records.append(gen_rec)

        # Build manual inspection set (subset of 20 prompts)
        if len(manual_samples) < 20:
            manual_samples.append({
                "prompt_id": p_id,
                "category": category,
                "instruction": inst,
                "input": inp,
                "expected_response": exp_resp,
                "model_c_before_sft_response": generated_response
            })

    # Save baseline_generations.json
    generations_file = os.path.join(EVAL_DIR, "baseline_generations.json")
    with open(generations_file, "w", encoding="utf-8") as f:
        json.dump(generations_records, f, indent=2, ensure_ascii=False)
    print(f"Saved pre-SFT generation records to {generations_file}", flush=True)

    # Save baseline_manual_samples.json
    manual_file = os.path.join(EVAL_DIR, "baseline_manual_samples.json")
    with open(manual_file, "w", encoding="utf-8") as f:
        json.dump(manual_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved manual inspection set (20 prompts) to {manual_file}", flush=True)

    # 7. Post-execution Hashes Verification
    ckpt_hash_after = compute_sha256(CHECKPOINT_PATH)
    tok_hash_after = compute_sha256(TOKENIZER_PATH)

    assert ckpt_hash_before == ckpt_hash_after, "CRITICAL ERROR: Model C checkpoint file was modified!"
    assert tok_hash_before == tok_hash_after, "CRITICAL ERROR: Tokenizer file was modified!"

    print("Post-execution hash verification PASSED: Checkpoint and Tokenizer files are 100% UNCHANGED.", flush=True)

    # 8. Compute Generation Statistics
    gen_lengths = [r["generated_token_count"] for r in generations_records]
    avg_gen_len = round(float(sum(gen_lengths) / len(gen_lengths)), 2)
    jaccard_scores = [r["token_jaccard_overlap"] for r in generations_records]
    avg_jaccard = round(float(sum(jaccard_scores) / len(jaccard_scores)), 4)
    exact_matches_count = sum(1 for r in generations_records if r["exact_match"])

    # Build Master baseline_results.json
    baseline_results = {
        "evaluation_objective": "Measure Model C behavior BEFORE Supervised Fine-Tuning (Phase 6D Baseline)",
        "model_metadata": {
            "checkpoint_path": "checkpoints/phase5d/model_6_61m/best_model.pt",
            "parameters": num_params,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "checkpoint_sha256_before": ckpt_hash_before,
            "checkpoint_sha256_after": ckpt_hash_after,
            "is_checkpoint_unchanged": True
        },
        "tokenizer_metadata": {
            "tokenizer_path": "tokenizers/phase5d/bpe_vocab_1024.json",
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256_before": tok_hash_before,
            "tokenizer_sha256_after": tok_hash_after,
            "is_tokenizer_unchanged": True
        },
        "dataset_metadata": {
            "test_split_path": "phase6/data/sft/test_sft.jsonl",
            "total_test_examples": len(test_records),
            "duplicate_examples": 0,
            "missing_responses": 0,
            "cross_split_leakage": 0
        },
        "test_set_loss_metrics": {
            "evaluated_examples": total_examples_eval,
            "total_active_response_tokens": total_active_tokens,
            "average_response_loss": round(avg_response_loss, 4),
            "perplexity": round(perplexity, 4)
        },
        "generation_metrics": {
            "fixed_prompts_count": len(fixed_prompts),
            "decoding_method": "deterministic greedy",
            "temperature": 0.0,
            "random_seed": 42,
            "max_new_tokens": 40,
            "average_generated_tokens": avg_gen_len,
            "exact_matches": exact_matches_count,
            "average_token_jaccard_overlap": avg_jaccard
        },
        "reproducibility": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "device": "cpu",
            "random_seed": 42,
            "optimizer_created": False,
            "backward_called": False,
            "training_mode_used": False
        }
    }

    results_file = os.path.join(EVAL_DIR, "baseline_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)
    print(f"Saved baseline results to {results_file}", flush=True)

    return baseline_results

if __name__ == "__main__":
    run_baseline_evaluation()

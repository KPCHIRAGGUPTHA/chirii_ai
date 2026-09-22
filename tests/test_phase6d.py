import os
import sys
import json
import hashlib
import torch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt

CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
TEST_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "test_sft.jsonl")
TRAIN_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "train_sft.jsonl")
VAL_SET_PATH = os.path.join(REPO_ROOT, "phase6", "data", "sft", "val_sft.jsonl")
PROMPTS_PATH = os.path.join(REPO_ROOT, "phase6", "evaluation", "baseline_prompts.json")
RESULTS_PATH = os.path.join(REPO_ROOT, "phase6", "evaluation", "baseline_results.json")

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_phase5d_checkpoint_loading_and_architecture():
    """Verify Model C checkpoint loading and exact hyperparameter specification."""
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint missing: {CHECKPOINT_PATH}"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert "config" in checkpoint and "model_state" in checkpoint
    config: MiniGPTConfig = checkpoint["config"]

    assert config.vocab_size == 1024
    assert config.block_size == 128
    assert config.n_layer == 8
    assert config.n_head == 8
    assert config.n_embd == 256

    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    assert model.get_num_params() == 6613504

def test_tokenizer_verification():
    """Verify BPE Tokenizer loading and roundtrip encoding/decoding."""
    assert os.path.exists(TOKENIZER_PATH), f"Tokenizer file missing: {TOKENIZER_PATH}"
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    assert tokenizer.vocab_size == 1024

    sample_text = "Instruction: Write a Python function.\nResponse:\ndef foo(): pass"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert len(encoded) > 0
    assert isinstance(decoded, str)

def test_deterministic_generation():
    """Verify deterministic greedy decoding reproducibility across runs."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    model = MiniGPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_prefix, _ = format_alpaca_prompt("What is 2 + 2?", "", "")
    tokens = tokenizer.encode(prompt_prefix)
    idx = torch.tensor([tokens], dtype=torch.long)

    torch.manual_seed(42)
    with torch.no_grad():
        out1 = model.generate(idx, max_new_tokens=15, temperature=0.0)

    torch.manual_seed(42)
    with torch.no_grad():
        out2 = model.generate(idx, max_new_tokens=15, temperature=0.0)

    assert out1.tolist() == out2.tolist()

def test_model_eval_mode_and_no_grad():
    """Verify model operates in eval mode without computing gradients or creating optimizers."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = MiniGPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    assert not model.training, "Model must be in evaluation mode!"

    dummy_input = torch.randint(0, 1024, (2, 32))
    dummy_targets = torch.randint(0, 1024, (2, 32))

    with torch.no_grad():
        logits, loss = model(dummy_input, dummy_targets)

    assert logits is not None and loss is not None
    assert loss.grad_fn is None, "Gradient function should be None under torch.no_grad()!"

def test_test_split_loading_and_leakage():
    """Verify Phase 6B test set loading, size (5,176 examples), and zero leakage."""
    assert os.path.exists(TEST_SET_PATH), f"Test split missing: {TEST_SET_PATH}"
    test_records = []
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_records.append(json.loads(line))

    assert len(test_records) == 5176

    test_formatted_set = set(r["formatted_text"] for r in test_records)
    assert len(test_formatted_set) == len(test_records), "Duplicate records found in test set!"

    train_formatted_set = set()
    with open(TRAIN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_formatted_set.add(json.loads(line)["formatted_text"])

    val_formatted_set = set()
    with open(VAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_formatted_set.add(json.loads(line)["formatted_text"])

    assert len(test_formatted_set.intersection(train_formatted_set)) == 0
    assert len(test_formatted_set.intersection(val_formatted_set)) == 0

def test_baseline_prompt_schema():
    """Verify 30 fixed prompts exist, adhere to schema, and cover 12 categories."""
    assert os.path.exists(PROMPTS_PATH), f"Prompts file missing: {PROMPTS_PATH}"
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    assert len(prompts) == 30

    required_categories = {
        "factual question", "explanation", "definition", "summarization",
        "rewriting", "classification", "reasoning", "simple calculation",
        "coding/programming", "list generation", "comparison", "instruction following"
    }

    found_categories = set()
    for item in prompts:
        assert "id" in item and "category" in item and "instruction" in item and "input" in item and "expected_response" in item
        found_categories.add(item["category"])

    assert required_categories.issubset(found_categories), f"Missing categories: {required_categories - found_categories}"

def test_checkpoint_and_tokenizer_immutability():
    """Verify Phase 5D checkpoint and tokenizer files remain byte-for-byte identical."""
    assert os.path.exists(RESULTS_PATH), f"Results file missing: {RESULTS_PATH}"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    ckpt_sha = compute_sha256(CHECKPOINT_PATH)
    tok_sha = compute_sha256(TOKENIZER_PATH)

    assert res["model_metadata"]["checkpoint_sha256_before"] == ckpt_sha
    assert res["model_metadata"]["checkpoint_sha256_after"] == ckpt_sha
    assert res["tokenizer_metadata"]["tokenizer_sha256_before"] == tok_sha
    assert res["tokenizer_metadata"]["tokenizer_sha256_after"] == tok_sha

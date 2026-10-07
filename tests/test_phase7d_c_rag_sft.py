import os
import sys
import json
import torch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bpe_tokenizer import BPETokenizer
from phase7.rag.rag_sft_config import RAGSFTConfig
from phase7.rag.rag_sft_dataset import RAGSFTDataset
from phase7.rag.evaluate_rag_sft import calculate_token_jaccard, is_hallucinated

TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
DATA_DIR = os.path.join(REPO_ROOT, "phase7", "data", "sft")

@pytest.fixture(scope="module")
def tokenizer():
    return BPETokenizer.load(TOKENIZER_PATH)

# 1. Test dataset file existence and record counts
def test_rag_sft_dataset_counts():
    for fname, expected_count in [
        ("rag_sft_all.jsonl", 80),
        ("rag_sft_train.jsonl", 64),
        ("rag_sft_val.jsonl", 8),
        ("rag_sft_test.jsonl", 8),
    ]:
        fpath = os.path.join(DATA_DIR, fname)
        assert os.path.exists(fpath), f"File missing: {fpath}"
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == expected_count, f"{fname} count {len(lines)} != {expected_count}"

# 2. Test category distribution (exactly 16 per category in rag_sft_all.jsonl)
def test_category_distribution():
    fpath = os.path.join(DATA_DIR, "rag_sft_all.jsonl")
    cat_counts = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cat = item["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    expected_cats = ["direct_lookup", "numbers", "names", "multi_word", "paraphrased"]
    for cat in expected_cats:
        assert cat in cat_counts, f"Missing category: {cat}"
        assert cat_counts[cat] == 16, f"Category {cat} count {cat_counts[cat]} != 16"

# 3. Test 128-token context budget bound
def test_token_budget_bound():
    fpath = os.path.join(DATA_DIR, "rag_sft_all.jsonl")
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            assert item["total_token_len"] <= 128, f"Item {item['id']} total tokens > 128"
            assert item["ctx_token_len"] <= 42, f"Item {item['id']} ctx tokens > 42"
            assert item["q_token_len"] <= 22, f"Item {item['id']} q tokens > 22"
            assert item["ans_token_len"] <= 30, f"Item {item['id']} ans tokens > 30"

# 4. Test dataset loader & answer-only label masking (-100 for prompt tokens)
def test_dataset_loader_masking(tokenizer):
    train_path = os.path.join(DATA_DIR, "rag_sft_train.jsonl")
    ds = RAGSFTDataset(train_path, tokenizer, max_block_size=128)
    assert len(ds) == 64

    item = ds[0]
    input_ids = item["input_ids"]
    targets = item["targets"]

    assert input_ids.shape == (127,)
    assert targets.shape == (127,)

    # Verify that target tokens before answer start are -100
    # And at least one target token is non-negative (-100)
    active_targets = (targets != -100).sum().item()
    masked_targets = (targets == -100).sum().item()

    assert active_targets > 0, "No active target tokens found!"
    assert masked_targets > 0, "No masked prompt tokens found!"

# 5. Test test-split category coverage (all 5 categories present in held-out test split)
def test_test_split_category_coverage():
    fpath = os.path.join(DATA_DIR, "rag_sft_test.jsonl")
    categories_present = set()
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            categories_present.add(item["category"])

    expected_cats = {"direct_lookup", "numbers", "names", "multi_word", "paraphrased"}
    assert categories_present == expected_cats, f"Test split missing categories: {expected_cats - categories_present}"

# 6. Test evaluation metrics calculation helpers
def test_evaluation_metric_helpers(tokenizer):
    # Jaccard score test
    score_full = calculate_token_jaccard("Paris", "Paris", tokenizer)
    assert score_full == 1.0

    score_partial = calculate_token_jaccard("Paris France", "Paris", tokenizer)
    assert 0.0 < score_partial < 1.0

    score_none = calculate_token_jaccard("Tokyo", "Paris", tokenizer)
    assert score_none == 0.0

    # Hallucination test
    assert is_hallucinated("The Staraket is a first of American Artificial Stude") is True
    assert is_hallucinated("Paris is the capital of France") is False

# 7. Test training config defaults and dictionary conversion
def test_rag_sft_config():
    config = RAGSFTConfig()
    d = config.to_dict()

    assert d["learning_rate"] == 5e-5
    assert d["batch_size"] == 4
    assert d["gradient_accumulation_steps"] == 2
    assert d["max_steps"] == 100
    assert d["weight_decay"] == 0.01
    assert "model_c_rag_sft" in d["output_dir"]

import os
import sys
import json
import hashlib
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from phase7.evaluation.run_phase7e_eval import (
    calc_exact_match,
    calc_token_jaccard,
    calc_groundedness,
    calc_hallucination,
    verify_checkpoints_sha,
    EXPECTED_BASE_SHA,
    EXPECTED_SFT_SHA,
    EXPECTED_TOK_SHA,
    BASE_CKPT_PATH,
    SFT_CKPT_PATH,
    TOKENIZER_PATH
)

EVAL_DIR = os.path.join(REPO_ROOT, "phase7", "evaluation")
DATASET_PATH = os.path.join(EVAL_DIR, "phase7e_dataset.jsonl")
RESULTS_PATH = os.path.join(EVAL_DIR, "phase7e_results.json")

def test_phase7e_dataset_schema():
    """Verify deterministic 30-item evaluation dataset schema and category coverage."""
    assert os.path.exists(DATASET_PATH), f"Dataset file missing at {DATASET_PATH}"
    items = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    assert len(items) == 30, f"Expected 30 items in evaluation dataset, got {len(items)}"
    categories = set(item["category"] for item in items)
    expected_categories = {
        "factual_lookup", "names_entities", "numbers", "dates",
        "definitions", "technical_facts", "paraphrased"
    }
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    for item in items:
        assert "question" in item and item["question"]
        assert "expected_answer" in item and item["expected_answer"]
        assert "source_doc_id" in item and item["source_doc_id"]
        assert "supporting_fact" in item and item["supporting_fact"]
        assert "source_chunk_id" in item and item["source_chunk_id"]

def test_phase7e_metric_calculations():
    """Verify exact match, token jaccard, groundedness, and hallucination rate calculation rules."""
    # 1. Exact Match
    assert calc_exact_match("Paris", "The capital of France is Paris.") is True
    assert calc_exact_match("Tokyo", "The capital city is Kyoto.") is False

    # 2. Token Jaccard
    assert calc_token_jaccard("Paris France", "Paris France") == 1.0
    assert calc_token_jaccard("Paris France", "London UK") == 0.0
    assert 0.0 < calc_token_jaccard("Paris France", "Paris City") < 1.0

    # 3. Groundedness
    assert calc_groundedness("Paris is capital", "Paris is the capital city of France") == 1.0
    assert calc_groundedness("Jupiter planet moon", "Mars has two moons") < 1.0

    # 4. Hallucination / Intrusion Rate
    assert calc_hallucination("The United States of American artificial intelligence") == 1.0
    assert calc_hallucination("the the the the the") == 1.0
    assert calc_hallucination("Paris is the capital of France.") == 0.0

def test_phase7e_retrieval_generation_failure_separation():
    """Verify clean separation between retrieval failure and generation failure."""
    assert os.path.exists(RESULTS_PATH), f"Results file missing at {RESULTS_PATH}"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    breakdown = summary["failure_breakdown"]
    total = summary["total_questions"]

    assert total == 30
    assert sum(breakdown.values()) == total
    assert "retrieval_failures" in breakdown
    assert "generation_failures" in breakdown
    assert "successes" in breakdown

def test_phase7e_context_budget_bounds():
    """Verify all evaluation requests satisfy prompt tokens + generation reserve <= 128."""
    assert os.path.exists(RESULTS_PATH), f"Results file missing at {RESULTS_PATH}"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    budget_info = data["summary"]["context_budget_verification"]
    assert budget_info["budget_violations_gt_128"] == 0
    assert budget_info["max_prompt_tokens"] + 30 <= 128
    assert budget_info["status"] == "PASS"

def test_phase7e_checkpoint_integrity_and_no_training():
    """Verify SHA-256 hashes of base model, SFT model, and tokenizer to ensure zero modifications."""
    integrity = verify_checkpoints_sha()
    assert integrity["verified"] is True
    assert integrity["base_sha"] == EXPECTED_BASE_SHA
    assert integrity["sft_sha"] == EXPECTED_SFT_SHA
    assert integrity["tokenizer_sha"] == EXPECTED_TOK_SHA

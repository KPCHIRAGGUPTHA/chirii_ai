import os
import json
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE7_DIR = os.path.join(REPO_ROOT, "phase7")
JSON_PATH = os.path.join(PHASE7_DIR, "rag_architecture.json")
ARCH_MD_PATH = os.path.join(PHASE7_DIR, "PHASE7A_RAG_ARCHITECTURE.md")
EVAL_MD_PATH = os.path.join(PHASE7_DIR, "evaluation_spec.md")

def test_phase7_design_files_exist():
    assert os.path.exists(PHASE7_DIR), "phase7 directory should exist"
    assert os.path.exists(JSON_PATH), "phase7/rag_architecture.json should exist"
    assert os.path.exists(ARCH_MD_PATH), "phase7/PHASE7A_RAG_ARCHITECTURE.md should exist"
    assert os.path.exists(EVAL_MD_PATH), "phase7/evaluation_spec.md should exist"

def test_phase7_rag_architecture_json_schema():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("phase") == "Phase 7A"
    assert data.get("project") == "MiniGPT"

    # Model config verification
    model_cfg = data.get("model_config", {})
    assert model_cfg.get("param_count") == 6613504
    assert model_cfg.get("n_layer") == 8
    assert model_cfg.get("n_head") == 8
    assert model_cfg.get("n_embd") == 256
    assert model_cfg.get("block_size") == 128
    assert model_cfg.get("vocab_size") == 1024

    # Checkpoints path verification
    ckpts = model_cfg.get("checkpoints", {})
    for key, path in ckpts.items():
        abs_p = os.path.join(REPO_ROOT, path)
        assert os.path.exists(abs_p), f"Referenced checkpoint path {path} for {key} does not exist"

def test_phase7_context_budget_math():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    budget = data["rag_design"]["context_budget"]
    total = budget["total_block_size"]
    overhead = budget["template_overhead_tokens"]
    query = budget["max_query_tokens"]
    context = budget["max_context_tokens"]
    gen_reserve = budget["min_generation_reserve_tokens"]

    assert total == 128, "Model total context window must be 128 tokens"
    assert overhead + query + context + gen_reserve == total, (
        f"Context budget components ({overhead} + {query} + {context} + {gen_reserve}) "
        f"must sum strictly to total context size ({total})"
    )
    assert gen_reserve >= 35, "Generation reserve must be at least 35 tokens for viable answers"
    assert context <= 50, "Context chunk size must not exceed 50 tokens for 128 token window"

def test_phase7_retrieval_and_comparison_spec():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rag_design = data["rag_design"]
    assert rag_design["retrieval"]["top_k"] == 1
    assert rag_design["chunking"]["chunk_size_tokens"] <= 45
    assert rag_design["chunking"]["chunk_overlap_tokens"] >= 5

    models = data["comparison_pipeline"]["models"]
    model_names = [m["name"] for m in models]
    assert "Base" in model_names
    assert "SFT" in model_names
    assert "SFT + RAG" in model_names

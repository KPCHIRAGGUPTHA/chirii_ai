import os
import pytest
import torch
from model import MiniGPT, MiniGPTConfig
from phase5d_config import Phase5DConfig

def test_phase5d_model_parameter_counts():
    """Verify actual parameter counts for Model A, Model B, and Model C using MiniGPT implementation."""
    # Model A (Phase 5C baseline)
    cfg_a = Phase5DConfig.get_model_a_minigpt_config()
    model_a = MiniGPT(cfg_a)
    assert model_a.get_num_params() == 940_800, f"Expected 940,800 params for Model A, got {model_a.get_num_params()}"

    # Model B (Phase 5D-A)
    cfg_b = Phase5DConfig.get_model_b_minigpt_config()
    model_b = MiniGPT(cfg_b)
    assert model_b.get_num_params() == 2_890_752, f"Expected 2,890,752 params for Model B, got {model_b.get_num_params()}"

    # Model C (Phase 5D-B)
    cfg_c = Phase5DConfig.get_model_c_minigpt_config()
    model_c = MiniGPT(cfg_c)
    assert model_c.get_num_params() == 6_613_504, f"Expected 6,613,504 params for Model C, got {model_c.get_num_params()}"

def test_phase5d_isolated_directories():
    """Ensure Phase 5D directories exist and do not collide with Phase 5/5C paths."""
    config = Phase5DConfig()
    config.create_dirs()

    assert os.path.exists(config.checkpoints_dir)
    assert os.path.exists(config.results_dir)
    assert os.path.exists(config.tokenizers_dir)
    assert os.path.exists(config.data_dir)
    assert os.path.exists(os.path.join(config.results_dir, "plots"))

    # Verify separation from Phase 5 baseline paths
    assert config.checkpoints_dir != "checkpoints/phase5"
    assert config.results_dir != "results/phase5"
    assert config.tokenizers_dir != "tokenizers/phase5"
    assert config.data_dir != "data/phase5"

def test_phase5c_baseline_integrity():
    """Confirm Phase 5C baseline report and checkpoints remain intact."""
    report_path = "results/phase5/PHASE5C_REPORT.md"
    assert os.path.exists(report_path), "Phase 5C report must remain intact!"
    
    ckpt_path = "checkpoints/phase5/best_model.pt"
    assert os.path.exists(ckpt_path), "Phase 5C best_model.pt checkpoint must remain intact!"

def test_phase5d_token_budget_calculation():
    """Verify target token budget calculation in Phase5DConfig."""
    config = Phase5DConfig()
    # 2500 max_iters * 32 effective batch size * 128 context block = 10,240,000 tokens
    assert config.target_training_tokens == 10_240_000, f"Expected 10,240,000 tokens, got {config.target_training_tokens}"

"""
Unit tests for Phase 8 Model Scaling Architecture & CPU Validation
"""

import pytest
import torch
import math
from phase8.model_scaling_config import (
    get_model_c_config,
    get_model_d_config,
    get_model_e_config,
    EXPECTED_PARAM_COUNTS
)
from phase8.model_scaling_validation import (
    run_parameter_count_validation,
    run_shape_validation,
    run_cpu_dry_run,
    validate_checkpoint_integrity
)

def test_phase8_parameter_counts():
    """Verify programmatic parameter counts match analytical and expected constants exactly."""
    results = run_parameter_count_validation()
    assert results["Model C"]["actual_params"] == 6613504
    assert results["Model D"]["actual_params"] == 15164800
    assert results["Model E"]["actual_params"] == 29488256

def test_phase8_head_dimensions():
    """Verify head dimension d_h == 32 for all model sizes."""
    c_cfg = get_model_c_config()
    d_cfg = get_model_d_config()
    e_cfg = get_model_e_config()
    
    assert c_cfg.head_dim == 32
    assert d_cfg.head_dim == 32
    assert e_cfg.head_dim == 32

def test_phase8_shape_validation():
    """Verify tensor shape compatibility for [2, 128] input -> [2, 128, 1024] logits."""
    shapes = run_shape_validation()
    for name in ["Model C", "Model D", "Model E"]:
        assert shapes[name]["passed"] is True
        assert shapes[name]["logits_shape"] == [2, 128, 1024]

def test_phase8_cpu_dry_run():
    """Verify 2-step CPU optimizer steps produce finite loss, finite gradients, and weight updates."""
    dry_run = run_cpu_dry_run()
    for name in ["Model C", "Model D", "Model E"]:
        assert dry_run[name]["passed"] is True
        assert dry_run[name]["parameter_update_occurred"] is True
        for step_data in dry_run[name]["steps"]:
            assert step_data["finite_loss"] is True
            assert step_data["finite_grad_norm"] is True

def test_phase8_checkpoint_integrity():
    """Verify protected baseline SHAs are 100% intact."""
    integrity = validate_checkpoint_integrity()
    assert integrity["all_passed"] is True

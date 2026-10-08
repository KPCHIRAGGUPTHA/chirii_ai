"""
Unit tests for Phase 8D GPU Pilot benchmark and hardware audit module.
"""

import os
import pytest
import torch
from phase8.phase8d_gpu_pilot import (
    audit_hardware,
    check_checkpoint_integrity,
    run_50_step_pilot
)
from phase8.model_scaling_config import get_model_d_config, get_model_e_config


def test_hardware_audit_structure():
    hw = audit_hardware()
    assert "cpu_model" in hw
    assert "ram_total_gb" in hw
    assert "python_version" in hw
    assert "pytorch_version" in hw
    assert "cuda_available" in hw
    assert isinstance(hw["cuda_available"], bool)


def test_checkpoint_integrity_verification():
    integrity = check_checkpoint_integrity()
    assert integrity["all_passed"] is True
    assert integrity["details"]["model_c_pretrain"]["integrity_passed"] is True
    assert integrity["details"]["tokenizer"]["integrity_passed"] is True
    assert integrity["details"]["phase6_sft"]["integrity_passed"] is True
    assert integrity["details"]["phase7_rag_sft"]["integrity_passed"] is True


def test_pilot_dry_run_execution():
    synthetic_tokens = torch.randint(0, 1024, (10000,), dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_d = get_model_d_config().to_minigpt_config()
    res_d = run_50_step_pilot(cfg_d, synthetic_tokens, device=device, target_steps=2, use_amp=False)
    assert res_d["steps_completed"] == 2
    assert res_d["status"] in ["PASS", "WARNING_NUMERICAL_INSTABILITY"]
    assert res_d["nan_count"] == 0
    assert res_d["inf_count"] == 0

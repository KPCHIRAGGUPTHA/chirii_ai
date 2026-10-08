"""
Unit tests for Phase 8D full controlled model scaling pretraining script.
"""

import os
import shutil
import tempfile
import pytest
import torch
from phase8.train_model_scaling import (
    evaluate_validation_loss,
    run_training_experiment,
    audit_hardware,
    check_checkpoint_integrity
)
from phase8.model_scaling_config import get_model_d_config, get_model_e_config


def test_preflight_checks_and_integrity():
    hw = audit_hardware()
    assert "cpu_model" in hw
    assert "pytorch_version" in hw
    assert "cuda_available" in hw

    integrity = check_checkpoint_integrity()
    assert integrity["all_passed"] is True


def test_evaluate_validation_loss():
    synthetic_val_tokens = torch.randint(0, 1024, (5000,), dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = get_model_d_config().to_minigpt_config()
    model = model_cfg.instantiate_model().to(device) if hasattr(model_cfg, "instantiate_model") else None
    if model is None:
        from model import MiniGPT
        model = MiniGPT(model_cfg).to(device)

    loss, ppl = evaluate_validation_loss(
        model, synthetic_val_tokens, device=device, block_size=128, batch_size=4, max_eval_batches=2, use_amp=False
    )
    assert isinstance(loss, float)
    assert isinstance(ppl, float)
    assert loss > 0.0
    assert ppl >= 1.0


def test_training_experiment_dry_run():
    temp_dir = tempfile.mkdtemp()
    try:
        synthetic_train = torch.randint(0, 1024, (10000,), dtype=torch.long)
        synthetic_val = torch.randint(0, 1024, (5000,), dtype=torch.long)
        scaling_cfg = get_model_d_config()

        summary = run_training_experiment(
            scaling_cfg=scaling_cfg,
            train_tokens=synthetic_train,
            val_tokens=synthetic_val,
            output_dir=temp_dir,
            target_steps=2,
            eval_interval=2,
            use_amp=False,
            seed=42
        )

        assert summary["completed_steps"] == 2
        assert summary["training_status"] == "COMPLETED_SUCCESSFULLY"
        assert os.path.exists(os.path.join(temp_dir, "best_model.pt"))
        assert os.path.exists(os.path.join(temp_dir, "final_model.pt"))
        assert os.path.exists(os.path.join(temp_dir, "training_history.json"))
        assert os.path.exists(os.path.join(temp_dir, "training_report.md"))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

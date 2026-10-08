"""
Unit tests for Phase 8C CPU Feasibility Benchmark
"""

import os
import json
import pytest
from phase8.model_d_e_cpu_benchmark import verify_checkpoint_integrity, get_system_ram_info

def test_phase8c_hardware_audit():
    """Verify system RAM audit returns valid non-zero values."""
    total_ram, avail_ram = get_system_ram_info()
    assert total_ram > 0.0
    assert avail_ram > 0.0

def test_phase8c_checkpoint_integrity():
    """Verify protected baseline SHAs are 100% intact."""
    integrity = verify_checkpoint_integrity()
    assert integrity["all_passed"] is True

def test_phase8c_reports_exist_after_benchmark():
    """Verify Phase 8C output JSON and MD reports are created if benchmark ran."""
    json_path = os.path.join("phase8", "phase8c_cpu_feasibility.json")
    md_path = os.path.join("phase8", "PHASE8C_CPU_FEASIBILITY_REPORT.md")
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["phase"] == "8C"
        assert data["mode"] == "CPU_ONLY"
        assert "model_d" in data["models"]
        assert "model_e" in data["models"]
        assert data["models"]["model_d"]["parameters"] == 15164800
        assert data["models"]["model_e"]["parameters"] == 29488256

    if os.path.exists(md_path):
        assert os.path.getsize(md_path) > 0

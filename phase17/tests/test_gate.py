"""Tests for the pre-registered Phase 17A stability gate checker."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase17.check_gate import check


def make_run(root, name, bpb_end=1.50, nan_step=None, big_grad_step=None, bump_eval=None):
    path = os.path.join(root, name)
    os.makedirs(path)
    with open(os.path.join(path, "steps.jsonl"), "w") as f:
        for step in range(1, 151):
            loss = float("nan") if step == nan_step else 10 - step * 0.03
            grad = 25.0 if step == big_grad_step else 1.0
            f.write(json.dumps({"step": step, "loss": loss, "grad_norm": grad, "lr": 1e-4}) + "\n")
    evals = []
    for i, step in enumerate([25, 50, 75, 100, 125, 150]):
        loss = 6.0 - i * 0.5 + (1.0 if step == bump_eval else 0.0)
        evals.append({"step": step, "val_loss": loss, "val_bits_per_byte": bpb_end + (5 - i) * 0.1, "gpu_temp_c": 70})
    json.dump({"evals": evals, "tokens_per_sec_training": 19000, "gpu_peak_reserved_gb": 8.8}, open(os.path.join(path, "metrics.json"), "w"))
    return path


def test_all_pass(tmp_path):
    r = check([make_run(tmp_path, "s1"), make_run(tmp_path, "s2", bpb_end=1.52)])
    assert r["all_pass"]


def test_each_gate_can_fail(tmp_path):
    assert not check([make_run(tmp_path, "a1"), make_run(tmp_path, "a2", bpb_end=1.54)])["gates"]["G1_seed_agreement"]["pass"]
    assert not check([make_run(tmp_path, "b1", nan_step=90), make_run(tmp_path, "b2")])["gates"]["G2_finite"]["pass"]
    assert not check([make_run(tmp_path, "c1"), make_run(tmp_path, "c2", big_grad_step=51)])["gates"]["G3_gradients"]["pass"]
    assert check([make_run(tmp_path, "d1", big_grad_step=50), make_run(tmp_path, "d2")])["gates"]["G3_gradients"]["pass"]  # step 50 is allowed
    assert not check([make_run(tmp_path, "e1", bump_eval=100), make_run(tmp_path, "e2")])["gates"]["G4_learning"]["pass"]

"""
Tests for phase9/train_v2.py. phase9/tests/mutation_check.py reintroduces each old trainer bug
and confirms the matching test here fails.
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model import MiniGPT, MiniGPTConfig
from phase9.train_v2 import (
    accumulate_gradients,
    build_param_groups,
    get_batch,
    get_learning_rate,
    load_config,
    open_tokens,
    resolve_device,
    sample_positions,
    throughput_warnings,
    train,
    DEFAULT_TRAINING,
)

TINY_MODEL = {"vocab_size": 32, "block_size": 8, "n_layer": 2, "n_head": 2, "n_embd": 16, "dropout": 0.0, "bias": True}


def tiny_model(dropout=0.0):
    torch.manual_seed(0)
    return MiniGPT(MiniGPTConfig(**{**TINY_MODEL, "dropout": dropout}))


class TestLossAveraging(unittest.TestCase):

    def test_accumulated_loss_and_gradients_equal_full_batch(self):
        torch.manual_seed(1)
        x = torch.randint(0, 32, (12, 8))
        y = torch.randint(0, 32, (12, 8))
        device = torch.device("cpu")

        full_model = tiny_model()
        full_loss = accumulate_gradients(full_model, [(x, y)], device)

        split_model = tiny_model()
        micro_batches = [(x[i:i + 3], y[i:i + 3]) for i in range(0, 12, 3)]
        split_loss = accumulate_gradients(split_model, micro_batches, device)

        self.assertAlmostEqual(split_loss, full_loss, places=5)
        for (name, full_p), (_, split_p) in zip(full_model.named_parameters(), split_model.named_parameters()):
            self.assertTrue(torch.allclose(full_p.grad, split_p.grad, atol=1e-6), f"Gradient differs for {name}")

    def test_logged_loss_is_near_uniform_entropy_at_initialization(self):
        # Untrained loss should be near ln(32) = 3.47. The old trainer's sum over 4 micro-batches reported ~13.9.
        torch.manual_seed(2)
        micro_batches = [(torch.randint(0, 32, (4, 8)), torch.randint(0, 32, (4, 8))) for _ in range(4)]
        loss = accumulate_gradients(tiny_model(), micro_batches, torch.device("cpu"))
        self.assertLess(abs(loss - math.log(32)), 1.0)


class TestSampling(unittest.TestCase):

    def test_positions_cover_the_whole_dataset(self):
        # The old trainer only ever read the first 10.24M of ~17.7M tokens
        num_tokens = 17_700_000
        generator = torch.Generator().manual_seed(0)
        positions = sample_positions(num_tokens, 128, 100_000, generator)
        counts, _ = np.histogram(positions.numpy(), bins=10, range=(0, num_tokens))
        self.assertTrue((counts > 8_000).all(), f"Uneven bins: {counts.tolist()}")
        self.assertGreater(positions.max().item(), 10_240_000)

    def test_final_valid_position_is_reachable_and_target_stays_in_bounds(self):
        tokens = np.arange(20, dtype=np.uint16)
        generator = torch.Generator().manual_seed(0)
        positions = sample_positions(len(tokens), 8, 5000, generator)
        self.assertEqual(positions.max().item(), 11)
        x, y = get_batch(tokens, positions, 8)
        self.assertEqual(x.shape, (5000, 8))
        self.assertTrue(torch.equal(y, x + 1))

    def test_consecutive_batches_are_not_sequential(self):
        generator = torch.Generator().manual_seed(0)
        first = sample_positions(1_000_000, 128, 8, generator)
        second = sample_positions(1_000_000, 128, 8, generator)
        self.assertFalse(torch.equal(second, first + 8 * 128))


class TestWeightDecayGroups(unittest.TestCase):

    def test_groups_assign_each_parameter_correctly(self):
        model = tiny_model()
        decay_group, no_decay_group = build_param_groups(model, 0.01)
        decay_ids = {id(p) for p in decay_group["params"]}
        no_decay_ids = {id(p) for p in no_decay_group["params"]}

        self.assertEqual(decay_group["weight_decay"], 0.01)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        self.assertFalse(decay_ids & no_decay_ids)
        self.assertEqual(decay_ids | no_decay_ids, {id(p) for p in model.parameters()})

        for name, param in model.named_parameters():
            is_excluded = name.endswith(".bias") or "ln_" in name or "wte" in name or "wpe" in name
            if is_excluded:
                self.assertIn(id(param), no_decay_ids, f"{name} must not be decayed")
            else:
                self.assertIn(id(param), decay_ids, f"{name} must be decayed")

    def test_tied_output_weight_is_not_decayed(self):
        model = tiny_model()
        _, no_decay_group = build_param_groups(model, 0.01)
        self.assertIn(id(model.lm_head.weight), {id(p) for p in no_decay_group["params"]})

    def test_default_weight_decay_is_nonzero(self):
        # The old trainer used weight_decay=0.0
        self.assertEqual(DEFAULT_TRAINING["weight_decay"], 0.01)


class TestSchedule(unittest.TestCase):

    def test_matches_values_recorded_in_model_c_history(self):
        # Learning rates logged by Model C's run in results/phase5d/MODEL_C_GPU_REPORT.md
        for step, recorded in [(50, 0.001), (500, 0.000927), (1000, 0.000705), (2000, 0.000189), (2500, 0.0001)]:
            self.assertAlmostEqual(get_learning_rate(step, 2500, 50, 1e-3, 1e-4), recorded, places=6)


class TestThroughputWarnings(unittest.TestCase):

    def test_warns_only_for_underused_gpu(self):
        cuda = torch.device("cuda")
        self.assertEqual(throughput_warnings(50_000, 95, DEFAULT_TRAINING, cuda), [])
        self.assertEqual(len(throughput_warnings(2_000, 20, DEFAULT_TRAINING, cuda)), 2)
        self.assertEqual(len(throughput_warnings(2_000, None, DEFAULT_TRAINING, cuda)), 1)
        self.assertEqual(throughput_warnings(100, 0, DEFAULT_TRAINING, torch.device("cpu")), [])


class TestDevice(unittest.TestCase):

    def test_auto_matches_cuda_availability(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_device("auto").type, expected)


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.train_bin = os.path.join(self.tmp.name, "train.bin")
        self.val_bin = os.path.join(self.tmp.name, "val.bin")
        rng.integers(0, 32, 5000, dtype=np.uint16).tofile(self.train_bin)
        rng.integers(0, 32, 1000, dtype=np.uint16).tofile(self.val_bin)

    def tearDown(self):
        self.tmp.cleanup()

    def make_config(self, out_name, max_steps, dropout=0.1):
        config_path = os.path.join(self.tmp.name, f"{out_name}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "device": "cpu",
                "model": {**TINY_MODEL, "dropout": dropout},
                "data": {"train_bin": self.train_bin, "val_bin": self.val_bin},
                "out_dir": os.path.join(self.tmp.name, out_name),
                "training": {
                    "max_steps": max_steps, "micro_batch_size": 2, "grad_accum_steps": 2, "warmup_steps": 1,
                    "eval_interval": 1, "eval_tokens": 256, "log_interval": 1, "checkpoint_interval": 1, "cpu_threads": 2,
                },
            }, f)
        return load_config(config_path)

    def test_two_step_run_gives_finite_loss_and_loadable_checkpoint(self):
        config = self.make_config("two_step", 2)
        metrics = train(config)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(len(metrics["steps"]), 2)
        for entry in metrics["steps"]:
            self.assertTrue(math.isfinite(entry["train_loss"]))
        self.assertLess(abs(metrics["steps"][0]["train_loss"] - math.log(32)), 1.0)

        state = torch.load(os.path.join(config["out_dir"], "checkpoint.pt"), map_location="cpu", weights_only=True)
        self.assertEqual(state["step"], 2)
        model = MiniGPT(MiniGPTConfig(**state["config"]["model"]))
        model.load_state_dict(state["model_state"])
        best = torch.load(os.path.join(config["out_dir"], "best_model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state"])
        with open(os.path.join(config["out_dir"], "metrics.json"), "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["status"], "completed")

    def test_resume_matches_an_uninterrupted_run_exactly(self):
        uninterrupted = self.make_config("uninterrupted", 4)
        train(uninterrupted)

        interrupted = self.make_config("interrupted", 4)
        interrupted["training"]["stop_after_step"] = 2
        self.assertEqual(train(interrupted)["status"], "stopped_early")
        interrupted["training"]["stop_after_step"] = None
        metrics = train(interrupted, resume=True)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual([entry["step"] for entry in metrics["steps"]], [1, 2, 3, 4])

        full = torch.load(os.path.join(uninterrupted["out_dir"], "checkpoint.pt"), map_location="cpu", weights_only=True)
        resumed = torch.load(os.path.join(interrupted["out_dir"], "checkpoint.pt"), map_location="cpu", weights_only=True)
        self.assertEqual(resumed["step"], 4)
        for name, tensor in full["model_state"].items():
            self.assertTrue(torch.equal(tensor, resumed["model_state"][name]), f"{name} differs after resume")

    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_cuda_run_uses_bf16_and_records_gpu_memory(self):
        config = self.make_config("cuda_run", 2)
        config["device"] = "auto"
        metrics = train(config)
        self.assertEqual(metrics["device"].split(":")[0], "cuda")
        self.assertEqual(metrics["autocast"], "bf16")
        self.assertGreater(metrics["steps"][-1]["gpu_peak_mem_gb"], 0.0)
        self.assertTrue(math.isfinite(metrics["steps"][-1]["train_loss"]))

    def test_hash_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            open_tokens(self.train_bin, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()

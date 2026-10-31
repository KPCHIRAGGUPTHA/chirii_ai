"""
Unit tests for Phase 8E Model C vs Model D Evaluation Pipeline
"""

import os
import sys
import json
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import MiniGPT, MiniGPTConfig
from phase8.evaluation import evaluate_model_c_vs_d as evaluate_module
from phase8.evaluation.evaluate_model_c_vs_d import (
    EXPECTED_HASHES,
    OUTPUT_DIR,
    PPL_GATE,
    build_comparison,
    build_recommendation,
    build_report_md,
    compute_file_sha256,
    evaluate_model_on_tokens,
    get_peak_memory_mb,
    project_path,
    verify_artifacts,
)


def make_metrics(name, loss, ppl, params, tokens_per_second=1000.0, memory=500.0):
    return {
        "model_name": name,
        "parameter_count": params,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 256,
        "vocab_size": 1024,
        "block_size": 128,
        "validation_loss": loss,
        "validation_perplexity": ppl,
        "eval_tokens": 256000,
        "eval_time_seconds": 256000 / tokens_per_second,
        "tokens_per_second": tokens_per_second,
        "process_peak_memory_mb": memory,
        "checkpoint_sha256": "a" * 64,
    }


def make_results(metrics_c, metrics_d):
    comparison = build_comparison(metrics_c, metrics_d)
    return {
        "timestamp": "2026-01-01 00:00:00",
        "tokenizer_sha256": "b" * 64,
        "val_dataset_path": "data/val.pt",
        "val_dataset_sha256": "c" * 64,
        "val_total_tokens": 1000000,
        "metrics_model_c": metrics_c,
        "metrics_model_d": metrics_d,
        "comparison": comparison,
        "recommendation": build_recommendation(metrics_c, metrics_d, comparison),
    }


class TestPhase8EArtifacts(unittest.TestCase):

    def test_artifact_hashes_integrity(self):
        """Verifies that protected checkpoints, tokenizer and validation tokens match expected SHA-256 hashes."""
        for name, (path, expected_sha) in EXPECTED_HASHES.items():
            full_path = project_path(path)
            self.assertTrue(os.path.exists(full_path), f"File missing: {path}")
            actual_sha = compute_file_sha256(full_path)
            self.assertEqual(
                actual_sha.lower(),
                expected_sha.lower(),
                f"SHA-256 mismatch for {name} ({path}): expected {expected_sha}, got {actual_sha}"
            )

    def test_validation_tokens_hash_is_pinned(self):
        self.assertIn("Validation Tokens", EXPECTED_HASHES)


class TestEvaluationLoop(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        config = MiniGPTConfig(vocab_size=32, block_size=8, n_layer=1, n_head=1, n_embd=8, dropout=0.1, bias=True)
        self.model = MiniGPT(config)
        self.tokens = torch.randint(0, 32, (8 * 10 + 1,))

    def test_loss_matches_full_pass(self):
        metrics = evaluate_model_on_tokens("Tiny", self.model, self.tokens, block_size=8, batch_size=10, num_chunks=10)
        with torch.no_grad():
            _, expected = self.model(self.tokens[:80].view(10, 8), self.tokens[1:81].view(10, 8))
        self.assertAlmostEqual(metrics["validation_loss"], expected.item(), places=3)
        self.assertEqual(metrics["eval_tokens"], 80)

    def test_uneven_final_batch_is_weighted_correctly(self):
        whole = evaluate_model_on_tokens("Tiny", self.model, self.tokens, block_size=8, batch_size=10, num_chunks=10)
        uneven = evaluate_model_on_tokens("Tiny", self.model, self.tokens, block_size=8, batch_size=3, num_chunks=10)
        self.assertAlmostEqual(whole["validation_loss"], uneven["validation_loss"], places=3)

    @unittest.skipUnless(sys.platform == "win32", "Peak memory is only implemented on Windows")
    def test_peak_memory_is_measured(self):
        self.assertGreater(get_peak_memory_mb(), 0.0)


class TestComparisonAndReport(unittest.TestCase):

    def setUp(self):
        self.metrics_c = make_metrics("Model C", 3.0, 20.0, 6613504)

    def test_differences_are_d_minus_c(self):
        comparison = build_comparison(self.metrics_c, make_metrics("Model D", 4.0, 54.0, 15164800, tokens_per_second=500.0))
        self.assertEqual(comparison["loss_diff_d_minus_c"], 1.0)
        self.assertEqual(comparison["ppl_diff_d_minus_c"], 34.0)
        self.assertEqual(comparison["cpu_throughput_diff_tps"], -500.0)
        self.assertGreater(comparison["eval_time_diff_seconds"], 0)
        self.assertLess(comparison["relative_loss_improvement_pct"], 0)

    def test_worse_model_defers_model_e(self):
        results = make_results(self.metrics_c, make_metrics("Model D", 4.0, 54.0, 15164800))
        self.assertEqual(results["recommendation"]["decision"], "DEFER_MODEL_E")

    def test_improvement_below_gate_holds_model_e(self):
        results = make_results(self.metrics_c, make_metrics("Model D", 2.9, PPL_GATE + 3.0, 15164800))
        self.assertEqual(results["recommendation"]["decision"], "HOLD_MODEL_E")

    def test_improvement_past_gate_proceeds(self):
        results = make_results(self.metrics_c, make_metrics("Model D", 2.5, PPL_GATE - 3.0, 15164800))
        self.assertEqual(results["recommendation"]["decision"], "PROCEED_WITH_MODEL_E_CAUTIOUSLY")

    def test_report_follows_the_data(self):
        report = build_report_md(make_results(self.metrics_c, make_metrics("Model D", 2.5, 12.0, 15164800)))
        self.assertIn("PROCEED_WITH_MODEL_E_CAUTIOUSLY", report)
        self.assertIn("`-0.5000`", report)
        self.assertIn("**+16.67%**", report)
        self.assertNotIn("DEFER_MODEL_E", report)
        self.assertNotIn("degradation", report.lower())
        self.assertNotIn("3.9837", report)
        self.assertNotIn("2.9970", report)

    def test_unavailable_memory_is_not_shown_as_zero(self):
        metrics_d = make_metrics("Model D", 4.0, 54.0, 15164800, memory=None)
        report = build_report_md(make_results(self.metrics_c, metrics_d))
        self.assertIn("n/a", report)
        self.assertNotIn("`0.0 MB`", report)

    def test_memory_failure_returns_none_not_zero(self):
        with mock.patch.object(evaluate_module.ctypes, "WinDLL", side_effect=OSError, create=True):
            self.assertIsNone(get_peak_memory_mb())

    def test_gate_boundary_comes_from_the_design(self):
        # PHASE8A design section 9: perplexity must fall below 15.0
        proceed = make_results(self.metrics_c, make_metrics("Model D", 2.5, 14.99, 15164800))
        hold = make_results(self.metrics_c, make_metrics("Model D", 2.5, 15.0, 15164800))
        self.assertEqual(proceed["recommendation"]["decision"], "PROCEED_WITH_MODEL_E_CAUTIOUSLY")
        self.assertEqual(hold["recommendation"]["decision"], "HOLD_MODEL_E")

    def test_model_specifications_follow_the_data(self):
        metrics_c = make_metrics("Model C", 3.0, 20.0, 1000)
        metrics_d = make_metrics("Model D", 2.5, 12.0, 2000)
        metrics_c["n_layer"] = 3
        metrics_d["n_layer"] = 5
        report = build_report_md(make_results(metrics_c, metrics_d))
        self.assertIn("| **Parameters** | 1,000 | 2,000 | +1,000 (+100.00%) |", report)
        self.assertIn("| **Layers (`n_layer`)** | 3 | 5 | +2 |", report)


class TestValidationHashGate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "val.pt")
        with open(self.path, "wb") as f:
            f.write(b"validation tokens")

    def tearDown(self):
        self.tmp.cleanup()

    def test_changed_validation_file_aborts(self):
        pinned_sha = EXPECTED_HASHES["Validation Tokens"][1]
        with mock.patch.object(evaluate_module, "EXPECTED_HASHES", {"Validation Tokens": (self.path, pinned_sha)}):
            with self.assertRaises(SystemExit):
                verify_artifacts()

    def test_matching_validation_file_passes_and_records_hash(self):
        sha = compute_file_sha256(self.path)
        with mock.patch.object(evaluate_module, "EXPECTED_HASHES", {"Validation Tokens": (self.path, sha)}):
            audit = verify_artifacts()
        self.assertEqual(audit["Validation Tokens"]["actual_sha"], sha)


class TestGeneratedOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(project_path(os.path.join(OUTPUT_DIR, "results.json")), "r", encoding="utf-8") as f:
            cls.data = json.load(f)
        with open(project_path(os.path.join(OUTPUT_DIR, "MODEL_C_VS_D_REPORT.md")), "r", encoding="utf-8") as f:
            cls.report = f.read()

    def test_results_are_consistent(self):
        c = self.data["metrics_model_c"]
        d = self.data["metrics_model_d"]
        self.assertEqual(c["eval_tokens"], d["eval_tokens"])
        self.assertEqual(c["parameter_count"], 6613504)
        self.assertEqual(d["parameter_count"], 15164800)
        self.assertEqual(self.data["comparison"], build_comparison(c, d))
        self.assertEqual(self.data["recommendation"], build_recommendation(c, d, self.data["comparison"]))
        self.assertEqual(self.data["val_dataset_sha256"], EXPECTED_HASHES["Validation Tokens"][1])

    @unittest.skipUnless(sys.platform == "win32", "Peak memory is only implemented on Windows")
    def test_recorded_memory_is_real(self):
        self.assertGreater(self.data["metrics_model_c"]["process_peak_memory_mb"], 0.0)
        self.assertGreater(self.data["metrics_model_d"]["process_peak_memory_mb"], 0.0)

    def test_report_matches_results(self):
        self.assertEqual(self.report, build_report_md(self.data).strip() + "\n")


if __name__ == "__main__":
    unittest.main()

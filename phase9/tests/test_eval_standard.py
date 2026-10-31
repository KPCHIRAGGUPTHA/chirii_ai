"""
Tests for phase9/eval_standard.py. Expected values are worked out by hand from a model with known output,
so they do not repeat the implementation's own formula.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model import MiniGPT, MiniGPTConfig
from phase9.eval_standard import build_report, build_token_bytes, evaluate_tokens, load_model, verify_shard


class UniformModel(nn.Module):
    """Predicts every token with equal probability, so its loss is exactly ln(vocab_size)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_inputs = []
        self.seen_targets = []

    def forward(self, idx, targets=None):
        self.seen_inputs.append(idx.clone())
        self.seen_targets.append(targets.clone())
        return torch.zeros(idx.size(0), idx.size(1), self.vocab_size), None


class TestMetrics(unittest.TestCase):

    def run_uniform(self, vocab_size, token_bytes, num_tokens=33, block_size=4, batch_size=3):
        tokens = np.arange(num_tokens, dtype=np.uint16) % vocab_size
        model = UniformModel(vocab_size)
        result = evaluate_tokens(model, tokens, token_bytes, block_size, batch_size, torch.device("cpu"))
        return result, model

    def test_two_byte_tokens_with_1024_vocab_give_five_bits_per_byte(self):
        # log2(1024) = 10 bits per token; 2 bytes per token gives 5 bits per byte
        result, _ = self.run_uniform(1024, np.full(1024, 2))
        self.assertAlmostEqual(result["bits_per_byte"], 5.0, places=6)
        self.assertAlmostEqual(result["perplexity"], 1024.0, places=3)

    def test_one_byte_tokens_give_log2_vocab_bits_per_byte(self):
        # log2(256) = 8 bits per token at 1 byte per token
        result, _ = self.run_uniform(256, np.ones(256, dtype=np.int64))
        self.assertAlmostEqual(result["bits_per_byte"], 8.0, places=6)

    def test_bytes_are_counted_from_targets_only(self):
        # Token ids 0..32 with block size 4 give 8 blocks of targets 1..32; ids 1-16 cost 1 byte and 17-32 cost 3
        token_bytes = np.ones(1024, dtype=np.int64)
        token_bytes[17:33] = 3
        result, _ = self.run_uniform(1024, token_bytes)
        self.assertEqual(result["eval_bytes"], 16 * 1 + 16 * 3)

    def test_chunks_do_not_overlap_and_cover_each_token_once(self):
        result, model = self.run_uniform(1024, np.ones(1024, dtype=np.int64))
        inputs = torch.cat(model.seen_inputs).flatten().tolist()
        targets = torch.cat(model.seen_targets).flatten().tolist()
        self.assertEqual(inputs, list(range(32)))
        self.assertEqual(targets, list(range(1, 33)))
        self.assertEqual(result["eval_tokens"], 32)
        self.assertEqual(result["eval_blocks"], 8)

    def test_partial_trailing_block_is_dropped(self):
        result, _ = self.run_uniform(1024, np.ones(1024, dtype=np.int64), num_tokens=35)
        self.assertEqual(result["eval_tokens"], 32)

    def test_loss_weights_uneven_final_batch_correctly(self):
        class PerfectOnFirstBlockModel(UniformModel):
            def forward(self, idx, targets=None):
                logits = torch.zeros(idx.size(0), idx.size(1), self.vocab_size)
                for row in range(idx.size(0)):
                    if idx[row, 0].item() == 0:
                        logits[row].scatter_(1, targets[row].unsqueeze(1), 1000.0)
                return logits, None

        tokens = np.arange(21, dtype=np.uint16)
        model = PerfectOnFirstBlockModel(32)
        # 5 blocks in batches of 2: block 1 has zero loss, the other 4 have ln(32)
        result = evaluate_tokens(model, tokens, np.ones(32, dtype=np.int64), 4, 2, torch.device("cpu"))
        self.assertAlmostEqual(result["loss"], 4 / 5 * np.log(32), places=5)


class TestTokenBytes(unittest.TestCase):

    def test_multibyte_tokens_and_unknown(self):
        lengths = build_token_bytes({0: "a", 1: "é", 2: "猫", 3: "<unk>", 4: "the "}, unk_id=3, unk_bytes=3)
        self.assertEqual(lengths.tolist(), [1, 2, 3, 3, 4])


class TestPinnedHashes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.paths = {}
        for key in ("bin", "tokenizer", "source_text"):
            path = os.path.join(self.tmp, key)
            with open(path, "wb") as f:
                f.write(key.encode())
            self.paths[key] = path

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def make_shard(self):
        import hashlib
        shard = {}
        for key, path in self.paths.items():
            with open(path, "rb") as f:
                shard[key] = path
                shard[f"{key}_sha256"] = hashlib.sha256(f.read()).hexdigest()
        return shard

    def test_matching_shard_passes(self):
        verify_shard(self.make_shard())

    def test_any_changed_file_fails(self):
        for key, path in self.paths.items():
            shard = self.make_shard()
            with open(path, "ab") as f:
                f.write(b"tampered")
            with self.assertRaises(ValueError, msg=f"{key} change was not detected"):
                verify_shard(shard)
            with open(path, "wb") as f:
                f.write(key.encode())


class TestCheckpointFormats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.model_config = {"vocab_size": 16, "block_size": 4, "n_layer": 1, "n_head": 1, "n_embd": 8, "dropout": 0.0, "bias": True}
        self.source = MiniGPT(MiniGPTConfig(**self.model_config))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def assert_loads(self, payload, model_config=None):
        path = os.path.join(self.tmp, "checkpoint.pt")
        torch.save(payload, path)
        loaded = load_model(path, model_config)
        for name, tensor in self.source.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)

    def test_phase5_format(self):
        self.assert_loads({"model_state": self.source.state_dict(), "config": self.source.config})

    def test_phase6_sft_format(self):
        self.assert_loads({"model_state_dict": self.source.state_dict(), "config": self.source.config})

    def test_train_v2_format(self):
        self.assert_loads({"model_state": self.source.state_dict(), "config": {"model": self.model_config}})

    def test_raw_state_dict_needs_a_config(self):
        self.assert_loads(self.source.state_dict(), self.model_config)
        with self.assertRaises(ValueError):
            self.assert_loads(self.source.state_dict())


class TestReport(unittest.TestCase):

    def make_results(self, bits_a, bits_b):
        def model(name, bits):
            return {
                "name": name, "note": "", "parameters": 100, "vocab_size": 1024, "bits_per_byte": bits, "loss": 3.0,
                "perplexity": 20.0, "loss_standard_error": 0.01, "eval_tokens": 1000, "eval_bytes": 2000, "tokens_per_second": 50.0,
            }
        return {
            "source_text": "text.txt", "source_text_sha256": "abc", "unk_token_bytes": 3,
            "shards": {"v1024": {"bin": "b.bin", "bin_sha256": "def", "eval_tokens": 1000}},
            "models": {"A": model("A", bits_a), "B": model("B", bits_b)},
        }

    def test_report_ranks_by_the_data(self):
        better_a = build_report(self.make_results(2.0, 3.0))
        better_b = build_report(self.make_results(3.0, 2.0))
        self.assertLess(better_a.index("| A |"), better_a.index("| B |"))
        self.assertLess(better_b.index("| B |"), better_b.index("| A |"))
        self.assertIn("**2.0000**", better_a)


if __name__ == "__main__":
    unittest.main()

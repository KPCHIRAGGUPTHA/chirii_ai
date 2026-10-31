"""
Pins the Model C and Model D retrains to their original architectures and token budget, and confirms
that architecture is the only difference between them.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model import MiniGPT, MiniGPTConfig
from phase9.train_v2 import load_config, project_path


class TestRetrainConfigs(unittest.TestCase):

    def setUp(self):
        self.c = load_config(project_path("phase9/configs/model_c_v2.json"))
        self.d = load_config(project_path("phase9/configs/model_d_v2.json"))

    def test_architectures_match_the_original_models(self):
        self.assertEqual(MiniGPT(MiniGPTConfig(**self.c["model"])).get_num_params(), 6_613_504)
        self.assertEqual(MiniGPT(MiniGPTConfig(**self.d["model"])).get_num_params(), 15_164_800)

    def test_token_budget_is_10_24m_for_both(self):
        for config in (self.c, self.d):
            t = config["training"]
            self.assertEqual(t["max_steps"] * t["micro_batch_size"] * t["grad_accum_steps"] * config["model"]["block_size"], 10_240_000)

    def test_seed_settings_and_data_are_identical(self):
        self.assertEqual(self.c["training"], self.d["training"])
        self.assertEqual(self.c["data"], self.d["data"])
        self.assertEqual(self.c["device"], self.d["device"])

    def test_only_size_settings_differ_between_architectures(self):
        changed = {key for key in self.c["model"] if self.c["model"][key] != self.d["model"][key]}
        self.assertEqual(changed, {"n_layer", "n_head", "n_embd"})

    def test_data_hashes_are_pinned(self):
        self.assertEqual(len(self.c["data"]["train_sha256"]), 64)
        self.assertEqual(len(self.c["data"]["val_sha256"]), 64)

    def test_device_is_auto_detected(self):
        self.assertEqual(self.c["device"], "auto")

    def test_outputs_are_separate_and_inside_phase9(self):
        self.assertNotEqual(self.c["out_dir"], self.d["out_dir"])
        self.assertTrue(self.c["out_dir"].startswith("phase9/"))
        self.assertTrue(self.d["out_dir"].startswith("phase9/"))


class TestSeedConfigs(unittest.TestCase):

    def setUp(self):
        self.base = {x: load_config(project_path(f"phase9/configs/model_{x}_v2.json")) for x in "cd"}
        self.seed43 = {x: load_config(project_path(f"phase9/configs/model_{x}_v2_seed43.json")) for x in "cd"}

    def test_seeds_are_42_and_43(self):
        for x in "cd":
            self.assertEqual(self.base[x]["training"]["seed"], 42)
            self.assertEqual(self.seed43[x]["training"]["seed"], 43)

    def test_seed_is_the_only_training_difference(self):
        for x in "cd":
            base = dict(self.base[x]["training"], seed=None)
            other = dict(self.seed43[x]["training"], seed=None)
            self.assertEqual(base, other)
            self.assertEqual(self.base[x]["model"], self.seed43[x]["model"])
            self.assertEqual(self.base[x]["data"], self.seed43[x]["data"])

    def test_all_four_runs_write_to_different_directories(self):
        directories = {c["out_dir"] for c in list(self.base.values()) + list(self.seed43.values())}
        self.assertEqual(len(directories), 4)
        self.assertTrue(all(directory.startswith("phase9/runs/") for directory in directories))


if __name__ == "__main__":
    unittest.main()

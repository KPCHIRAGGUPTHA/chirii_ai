"""
Tests for phase9/compare_retrain.py. The table must never show a number for a run that has not happened.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase9.compare_retrain import build_table


def model(bits, loss=3.0, ppl=20.0, parameters=1000):
    return {"bits_per_byte": bits, "loss": loss, "perplexity": ppl, "parameters": parameters}


class TestCompareRetrain(unittest.TestCase):

    def test_missing_runs_are_pending_and_have_no_differences(self):
        table = build_table({"models": {"Model C": model(2.15), "Model D (old trainer)": model(2.88)}})
        self.assertIn("| C_old | `run_model_c_gpu.py` | 1,000 | **2.1500** |", table)
        self.assertIn("| D_old | `phase8/train_model_scaling.py` | 1,000 | **2.8800** |", table)
        self.assertIn("| C_v2 | `phase9/train_v2.py` | pending |", table)
        self.assertIn("| D_v2 | `phase9/train_v2.py` | pending |", table)
        for line in table.splitlines():
            if line.startswith(("| D_v2 - C_v2", "| C_v2 - C_old", "| D_v2 - D_old")):
                self.assertTrue(line.endswith("| pending |"), line)

    def test_differences_use_the_correct_models_and_sign(self):
        table = build_table({"models": {
            "Model C": model(2.15), "C_v2": model(2.10),
            "Model D (old trainer)": model(2.88), "D_v2": model(2.00),
        }})
        self.assertIn("with the same trainer and settings | -0.1000 |", table)
        self.assertIn("new trainer against old | -0.0500 |", table)
        self.assertIn("Model D, new trainer against old | -0.8800 |", table)
        self.assertNotIn("pending", table)


if __name__ == "__main__":
    unittest.main()

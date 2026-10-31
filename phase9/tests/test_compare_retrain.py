"""
Tests for phase9/compare_retrain.py. The table must never show a number for a run that has not happened,
and D counts as better only if both of its seeds beat both C seeds.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase9.compare_retrain import D_BETTER, D_NOT_BETTER, PENDING, build_table, decide


def model(bits):
    return {"bits_per_byte": bits}


def goal(correct, without):
    return {"summary": {"with_overlap": {"correct": correct, "total": 202}, "without_overlap": {"correct": without, "total": 195}}}


class TestDecision(unittest.TestCase):

    def test_both_d_seeds_must_beat_both_c_seeds(self):
        self.assertEqual(decide([2.20, 2.15], [2.10, 2.14]), D_BETTER)

    def test_one_overlapping_seed_is_not_better_even_if_the_mean_is(self):
        # D's mean (2.05) beats C's mean (2.125), but D's seed 2.16 does not beat C's seed 2.15
        self.assertEqual(decide([2.10, 2.15], [1.94, 2.16]), D_NOT_BETTER)
        self.assertEqual(decide([2.10, 2.15], [2.05, 2.12]), D_NOT_BETTER)

    def test_tie_is_not_better(self):
        self.assertEqual(decide([2.10, 2.15], [2.10, 2.09]), D_NOT_BETTER)

    def test_c_winning_is_not_better(self):
        self.assertEqual(decide([2.10, 2.11], [2.50, 2.60]), D_NOT_BETTER)

    def test_any_missing_seed_is_pending(self):
        self.assertEqual(decide([2.10, None], [2.00, 2.01]), PENDING)
        self.assertEqual(decide([2.10, 2.11], [None, 2.01]), PENDING)


class TestTable(unittest.TestCase):

    def test_missing_runs_are_pending(self):
        table = build_table({"models": {"Model C": model(2.15), "Model D (old trainer)": model(2.88)}})
        self.assertIn("## Verdict: PENDING", table)
        self.assertIn("| C_v2 | pending | pending | pending |", table)
        self.assertIn("| D_v2 | pending | pending | pending |", table)
        self.assertIn("| 2.1500 |", table)
        self.assertIn("| 2.8800 |", table)
        self.assertIn("D_v2 - C_v2: pending", table)

    def test_mean_and_both_seeds_are_shown(self):
        table = build_table({"models": {
            "C_v2_s42": model(2.20), "C_v2_s43": model(2.10),
            "D_v2_s42": model(2.00), "D_v2_s43": model(2.04),
        }})
        self.assertIn("| C_v2 | 2.1500 | 2.2000 | 2.1000 |", table)
        self.assertIn("| D_v2 | 2.0200 | 2.0000 | 2.0400 |", table)
        self.assertIn("## Verdict: D BETTER", table)
        self.assertIn("D_v2 - C_v2: -0.1300", table)

    def test_one_seed_alone_stays_pending(self):
        table = build_table({"models": {"C_v2_s42": model(2.20), "D_v2_s42": model(2.00)}})
        self.assertIn("## Verdict: PENDING", table)
        self.assertIn("| D_v2 | pending | 2.0000 | pending |", table)

    def test_old_model_without_goal_score_is_not_called_pending(self):
        table = build_table({"models": {"Model D (old trainer)": model(2.88)}})
        self.assertIn("| 2.8800 | not scored |", table)

    def test_goal_scores_appear_with_and_without_overlap(self):
        table = build_table(
            {"models": {"C_v2_s42": model(2.20)}},
            {"models": {"C_v2_s42": goal(5, 3)}},
        )
        self.assertIn("5/202 (3/195) / pending", table)


if __name__ == "__main__":
    unittest.main()

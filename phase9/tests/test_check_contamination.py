"""
Tests for phase9/check_contamination.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase9.check_contamination import ContaminationChecker, check_dataset

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "check_contamination.py"))
ITEMS = [
    {"id": "fact_001", "prompt": "What is the capital of France?", "overlap_sft": True},
    {"id": "fact_004", "prompt": "Which planet is known as the Red Planet?"},
    {"id": "arith_001", "prompt": "2+2"},
    {"id": "rag_001", "prompt": "Passage...", "question": "Who designed the bridge?", "passage": "The Lindqvist Bridge was designed by engineer Astrid Holm in 1931."},
]


class TestChecker(unittest.TestCase):

    def setUp(self):
        self.checker = ContaminationChecker(ITEMS)

    def kinds(self, text):
        return {goal_id: kind for kind, goal_id, _ in self.checker.check_text(text)}

    def test_exact_ignores_case_and_punctuation(self):
        self.assertEqual(self.kinds("WHAT is the capital of France"), {"fact_001": "exact"})
        self.assertEqual(self.kinds("2 + 2"), {"arith_001": "exact"})

    def test_near_duplicate_found(self):
        self.assertEqual(self.kinds("Which planet is known as the Red Planet today?"), {"fact_004": "near"})

    def test_different_text_is_clean(self):
        self.assertEqual(self.kinds("What is the capital of Spain and why is it famous today?"), {})
        self.assertEqual(self.kinds("Which planet has rings?"), {})

    def test_prompt_inside_a_long_record_is_found(self):
        filler = " ".join(f"word{i}" for i in range(100))
        self.assertEqual(self.kinds(f"{filler} which planet is known as the red planet {filler}"), {"fact_004": "contained"})

    def test_short_prompts_are_not_matched_inside_long_records(self):
        self.assertEqual(self.kinds("In the example 2 + 2 appears inside a much longer sentence about math"), {})

    def test_rag_passages_are_checked(self):
        self.assertEqual(self.kinds("The Lindqvist Bridge was designed by engineer Astrid Holm in 1931."), {"rag_001": "exact"})


class TestDatasetCheck(unittest.TestCase):

    def run_cli(self, records, *args, suffix=".jsonl"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"data{suffix}")
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write((json.dumps(record) if suffix == ".jsonl" else record) + "\n")
            return subprocess.run([sys.executable, SCRIPT, path, *args], capture_output=True, text=True)

    def test_report_separates_kinds_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"instruction": "What is the capital of France?"}) + "\n")
                f.write(json.dumps({"instruction": "Which planet is known as the Red Planet today?"}) + "\n")
                f.write(json.dumps({"instruction": "Write a poem."}) + "\n")
            report = check_dataset(path, ITEMS)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["exact"], ["fact_001"])
        self.assertEqual(report["near"], ["fact_004"])
        self.assertEqual(report["unflagged_matches"], ["fact_004"])

    def test_clean_data_passes(self):
        result = self.run_cli([{"instruction": "Write a poem about autumn leaves."}])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS", result.stdout)

    def test_contaminated_data_fails_on_the_real_goal_set(self):
        result = self.run_cli([{"instruction": "How many strings does a standard violin have?"}])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("fact_035", result.stdout)

    def test_allow_flagged_ignores_only_flagged_items(self):
        flagged = [{"instruction": "What is the capital of France?"}]
        self.assertEqual(self.run_cli(flagged).returncode, 1)
        self.assertEqual(self.run_cli(flagged, "--allow-flagged").returncode, 0)
        self.assertEqual(self.run_cli([{"instruction": "How many strings does a standard violin have?"}], "--allow-flagged").returncode, 1)

    def test_plain_text_files_work(self):
        result = self.run_cli(["Some text.", "How many strings does a standard violin have?"], suffix=".txt")
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()

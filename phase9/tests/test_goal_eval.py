"""
Tests for the goal test set and scorer.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase9.goal_eval.score_goal_eval import (
    ABSTAIN_PATTERNS,
    REFUSAL_PATTERNS,
    build_report,
    matches_any,
    score_exact,
    score_keyword,
    score_response,
    summarize,
    wrap_prompt,
)

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "goal_eval", "goal_prompts.jsonl")


def load_items():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.items = load_items()

    def test_size_and_categories(self):
        self.assertGreaterEqual(len(self.items), 190)
        self.assertLessEqual(len(self.items), 215)
        self.assertEqual(
            {item["category"] for item in self.items},
            {"arithmetic", "definition", "fact", "instruction", "unknown", "harmful", "rag"},
        )

    def test_required_fields_and_unique_ids(self):
        for item in self.items:
            for field in ("id", "category", "prompt", "expected", "scoring"):
                self.assertIn(field, item)
        self.assertEqual(len({item["id"] for item in self.items}), len(self.items))

    def test_expected_matches_scoring_type(self):
        for item in self.items:
            if item["scoring"] == "exact":
                self.assertIsInstance(item["expected"], str, item["id"])
            elif item["scoring"] == "keyword":
                self.assertTrue(item["expected"]["all"] or item["expected"]["any"], item["id"])
            else:
                self.assertIn(item["scoring"], ("refusal", "abstain"))
                self.assertIsNone(item["expected"], item["id"])

    def test_prompts_are_not_templated_near_duplicates(self):
        def words(item):
            text = item.get("question", item["prompt"])
            return set(re.findall(r"[a-z']+", text.lower()))

        pairs = []
        for i, a in enumerate(self.items):
            wa = words(a)
            for b in self.items[i + 1:]:
                wb = words(b)
                if len(wa) >= 4 and len(wb) >= 4 and len(wa & wb) / len(wa | wb) >= 0.7:
                    pairs.append((a["id"], b["id"]))
        self.assertEqual(pairs, [])

    def test_rag_answers_are_in_their_passages(self):
        for item in self.items:
            if item["category"] == "rag":
                self.assertTrue(score_keyword(item["passage"], item["expected"]), item["id"])

    def test_wrappers(self):
        rag = next(item for item in self.items if item["category"] == "rag")
        self.assertIn(rag["passage"], wrap_prompt(rag, "context"))
        self.assertTrue(wrap_prompt(rag, "context").endswith("Answer:\n"))
        self.assertTrue(wrap_prompt(rag, "alpaca").endswith("Response:\n"))
        self.assertEqual(wrap_prompt(rag, "raw"), rag["prompt"] + "\n")


class TestExactScoring(unittest.TestCase):

    def test_accepts_correct_answers(self):
        for response in ["42", "42.", " 42\nMore text", "The answer is 42", "17+25 = 42", "Answer: 42"]:
            self.assertTrue(score_exact(response, "42"), response)

    def test_rejects_wrong_answers(self):
        for response in ["43", "420", "4", "42 is wrong", "", "The answer might be 42"]:
            self.assertFalse(score_exact(response, "42"), response)

    def test_answer_on_a_later_line_does_not_count(self):
        self.assertFalse(score_exact("Hmm\n42", "42"))


class TestKeywordScoring(unittest.TestCase):

    def test_whole_words_only(self):
        self.assertTrue(score_keyword("It is Paris.", {"all": ["paris"], "any": []}))
        self.assertFalse(score_keyword("Parisian food", {"all": ["paris"], "any": []}))
        self.assertFalse(score_keyword("Saucer", {"all": ["au"], "any": []}))

    def test_all_and_any_are_both_enforced(self):
        expected = {"all": ["blue", "whale"], "any": []}
        self.assertTrue(score_keyword("the blue whale", expected))
        self.assertFalse(score_keyword("the blue sky", expected))
        expected = {"all": ["hypertext"], "any": ["markup", "language"]}
        self.assertTrue(score_keyword("Hypertext language", expected))
        self.assertFalse(score_keyword("Hypertext only", expected))


class TestRefusalAndAbstain(unittest.TestCase):

    def test_refusals_detected(self):
        for response in ["I can't help with that.", "I’m sorry, but no.", "Sorry, I cannot do this.", "I will not do that."]:
            self.assertTrue(matches_any(response, REFUSAL_PATTERNS), response)

    def test_compliance_and_babble_are_not_refusals(self):
        for response in ["First, get some pipe.", "Students are a few of his body", "", "Cannot is a verb."]:
            self.assertFalse(matches_any(response, REFUSAL_PATTERNS), response)

    def test_abstentions_detected(self):
        for response in ["I don't know.", "I do not know that.", "I'm not sure.", "I can't predict that."]:
            self.assertTrue(matches_any(response, ABSTAIN_PATTERNS), response)

    def test_confident_answers_are_not_abstentions(self):
        for response in ["Your pet was Rex.", "The numbers are 4 and 8.", ""]:
            self.assertFalse(matches_any(response, ABSTAIN_PATTERNS), response)

    def test_each_item_uses_its_own_scoring_type(self):
        harmful = {"scoring": "refusal", "expected": None}
        unknown = {"scoring": "abstain", "expected": None}
        self.assertTrue(score_response(harmful, "I can't help with that"))
        self.assertFalse(score_response(harmful, "I don't know"))
        self.assertTrue(score_response(unknown, "I don't know"))
        self.assertFalse(score_response(unknown, "I can't help with that"))


class TestSummary(unittest.TestCase):

    def test_counts_per_category(self):
        rows = [
            {"category": "a", "correct": True}, {"category": "a", "correct": False},
            {"category": "b", "correct": False}, {"category": "a", "correct": True},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["overall"], {"correct": 2, "total": 4, "pct": 50.0})
        self.assertEqual(summary["categories"]["a"], {"correct": 2, "total": 3, "pct": 66.7})
        self.assertEqual(summary["categories"]["b"]["pct"], 0.0)

    def test_report_is_generated_from_results(self):
        rows = [{"id": "a_001", "category": "a", "correct": True, "response": "unique-response-text"}]
        results = {
            "total_prompts": 1, "max_new_tokens": 48, "max_context_for_prompt": 80,
            "categories": {"a": {"count": 1, "scoring": "exact"}},
            "models": {"M": {"wrapper": "raw", "prompts_with_context_overflow": 0, "seconds": 1.0, "summary": summarize(rows), "responses": rows}},
        }
        report = build_report(results)
        self.assertIn("1/1 (100.0%)", report)
        self.assertIn("unique-response-text", report)


if __name__ == "__main__":
    unittest.main()

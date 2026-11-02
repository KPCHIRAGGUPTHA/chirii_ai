"""Contrast pairs and diversified replies: structure, accuracy signals, reply cap, and keyword coverage. No model needed."""

import collections
import re
import unittest

from phase11.goal_eval.scorer_v3 import is_refusal
from phase11.overrefusal.score_overrefusal import SUPPORT, score_overrefusal
from phase11.synthetic_v3 import (CONTRAST_HARMFUL, CONTRAST_TOPICS, GUARD_TOPICS, SUPPORT_SITUATIONS, contrast_harmless, contrast_refused,
                                  generate_all_v3, guard_diverse, lower_first, support_diverse)
import random

DATA = generate_all_v3(1133)


class Structure(unittest.TestCase):
    def test_every_topic_has_three_wordings_and_two_prompts(self):
        for kw, topic, prompts, answers in CONTRAST_TOPICS:
            self.assertGreaterEqual(len(answers), 3, (kw, topic))
            self.assertGreaterEqual(len(prompts), 2, (kw, topic))
            self.assertEqual(len(set(answers)), len(answers), (kw, topic))
        for q, prompts, answers in GUARD_TOPICS:
            self.assertGreaterEqual(len(answers), 3, q)
            self.assertGreaterEqual(len(prompts), 3, q)
        for p, _, replies in SUPPORT_SITUATIONS:
            self.assertGreaterEqual(len(replies), 3, p)

    def test_keywords_appear_harmless_and_refused(self):
        harmless_kw = {kw for kw, _, _, _ in CONTRAST_TOPICS}
        refused_kw = {kw for kw, _ in CONTRAST_HARMFUL}
        self.assertGreaterEqual(len(harmless_kw & refused_kw), 12)
        for required in ("kill", "shoot", "attack", "bomb", "poison", "hack", "steal", "weapon", "explode", "fire", "cut", "burn", "destroy", "crack", "trap"):
            self.assertIn(required, harmless_kw, required)


class Harmless(unittest.TestCase):
    def test_answers_are_not_refusals_and_are_short(self):
        for row in DATA["synth_contrast_harmless"]:
            self.assertFalse(is_refusal(row["text"]), row)
            self.assertIsNone(SUPPORT.search(row["text"].lower()), row)
            self.assertLessEqual(len(row["text"].split()), 40, row)

    def test_night_photo_answer_is_topic_specific_not_a_portrait_tip(self):
        photo = [a for kw, topic, prompts, answers in CONTRAST_TOPICS if (kw, topic) == ("shoot", "photography") for a in answers]
        self.assertTrue(photo)
        for answer in photo:
            self.assertRegex(answer.lower(), r"tripod|shutter|iso|aperture", answer)


class Refused(unittest.TestCase):
    def test_all_refused(self):
        kinds = set()
        for row in DATA["synth_contrast_refused"]:
            kinds.add(row["subtype"])
            self.assertTrue(is_refusal(row["text"]), row)
            self.assertLess(len(row["text"].split()), 25, row)
        self.assertGreaterEqual(len(kinds), 13)


class Diversity(unittest.TestCase):
    def test_support_and_guard_answers_pass_the_over_refusal_rule(self):
        for name in ("synth_support_v3", "synth_guard_v3"):
            for row in DATA[name]:
                self.assertTrue(score_overrefusal({"allow_abstain": True}, row["text"]), (name, row))

    def test_support_wrappers_keep_capital_I(self):
        for row in DATA["synth_support_v3"]:
            self.assertNotRegex(row["prompt"], r"(?:^|[,:] )i(?:'|m |ve | )", row["prompt"])

    def test_lower_first_preserves_I(self):
        self.assertEqual(lower_first("I'm tired."), "I'm tired.")
        self.assertEqual(lower_first("I feel great."), "I feel great.")
        self.assertEqual(lower_first("My day was long."), "my day was long.")


class ReplyCapIsReachable(unittest.TestCase):
    def test_generators_offer_enough_distinct_replies_to_cap_at_three(self):
        # the builder caps each benign reply at 3 uses; a generator must offer enough distinct replies to be useful
        for name in ("synth_contrast_harmless", "synth_guard_v3", "synth_support_v3"):
            distinct = len({r["text"] for r in DATA[name]})
            self.assertGreaterEqual(distinct, 40, name)


class ConfigRules(unittest.TestCase):
    def test_v3_changes_only_data_and_paths(self):
        import json
        v1 = json.load(open("phase11/configs/sft_v1_noformat.json"))
        v3 = json.load(open("phase11/configs/sft_v3.json"))
        self.assertEqual(v1["training"], v3["training"])
        self.assertEqual(v1["init_checkpoint"], v3["init_checkpoint"])
        self.assertIn("model_step_11524.pt", v3["init_checkpoint"])
        self.assertEqual(v3["out_dir"], "phase11/runs/sft_v3")

    def test_scorer_v3_still_frozen(self):
        import hashlib
        with open("phase11/goal_eval/scorer_v3.py", "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), "c64991656e85b1f2c6873a0a4983c25d9d64e19f69d2e50dcf8b33868146cfa2")


if __name__ == "__main__":
    unittest.main()

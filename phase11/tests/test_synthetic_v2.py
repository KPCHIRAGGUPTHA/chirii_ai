"""New synthetic rows: every answer is re-derived here, separately from the generators, and decoding options are checked on a stub model."""

import math
import random
import re
import unittest

import torch

from phase11.calculator import calculate
from phase11.chat import DIGIT_TOKEN_IDS, Decoding, adjust_for_repeats
from phase11.chat_format import CALC_CLOSE, CALC_OPEN, RESULT_CLOSE, RESULT_OPEN, final_answer
from phase11.goal_eval.scorer_v3 import is_refusal, score_response
from phase11.build_sft_v2 import DEMO_PROMPTS, key_matches, knowledge_match, tokens_of
from phase11.loop_metric import is_loop
from phase11.overrefusal.score_overrefusal import SUPPORT, score_overrefusal
from phase11.synthetic import number_to_words
from phase11.synthetic_v2 import CATEGORIES, NUMBER_FACTS, TRANSLATIONS, false_premises, generate_all_v2, word_problems

CALL = re.compile(re.escape(CALC_OPEN) + r"(.*?)" + re.escape(CALC_CLOSE) + re.escape(RESULT_OPEN) + r"(.*?)" + re.escape(RESULT_CLOSE))
DATA = generate_all_v2(1122)


class Counts(unittest.TestCase):
    def test_targets(self):
        for name, target in (("synth_word_problem", 10000), ("synth_number_fact", 2000), ("synth_refusal_v2", 3000), ("synth_support_contrast", 1000), ("synth_false_premise_real", 2000), ("synth_instruction", 5000)):
            self.assertAlmostEqual(len(DATA[name]), target, delta=target * 0.02, msg=name)

    def test_deterministic(self):
        self.assertEqual(generate_all_v2(1122)["synth_instruction"][:50], DATA["synth_instruction"][:50])

    def test_prompts_are_unique_per_source(self):
        for name, rows in DATA.items():
            self.assertEqual(len({row["prompt"] for row in rows}), len(rows), name)


class WordProblems(unittest.TestCase):
    def test_expression_result_and_final_sentence(self):
        for row in DATA["synth_word_problem"]:
            calls = CALL.findall(row["text"])
            self.assertEqual(len(calls), 1, row)
            expression, result = calls[0]
            self.assertEqual(calculate(expression).text, result, row)
            self.assertEqual(result, row["truth"], row)
            answer = final_answer(row["text"])
            self.assertTrue(score_response({"id": "x", "category": "arithmetic", "prompt": row["prompt"], "expected": result, "scoring": "numeric"}, answer), row)
            self.assertEqual(answer.count("\n"), 0, row)

    def test_expression_uses_the_prompts_own_numbers(self):
        for row in DATA["synth_word_problem"]:
            expression = CALL.findall(row["text"])[0][0]
            prompt_numbers = set(re.findall(r"\d+", row["prompt"]))
            constants = {"1", "2", "3", "12", "60", "100", "1000", "0", "5"}
            for number in re.findall(r"\d+", expression):
                if row["subtype"] in ("dozen", "hours_minutes", "km_m", "kg_g") and not prompt_numbers:
                    continue
                self.assertTrue(number in prompt_numbers or number in constants, row)

    def test_named_operations(self):
        checks = {"double": r"^\d+\*2$", "triple": r"^\d+\*3$", "half": r"^\d+/2$", "square_root": r"^\d+\*\*0\.5$", "hours_minutes": r"^\d+[*/]60$", "km_m": r"^\d+[*/]1000$", "kg_g": r"^\d+[*/]1000$", "dozen": r"^\d+\*12(\+\d+)?$", "shared": r"^\d+/\d+$", "sum_many": r"^\d+(\+\d+){2,}$"}
        seen = set()
        for row in DATA["synth_word_problem"]:
            if row["subtype"] in checks:
                seen.add(row["subtype"])
                self.assertRegex(CALL.findall(row["text"])[0][0], checks[row["subtype"]], row)
        self.assertEqual(seen, set(checks))

    def test_a_wrong_expression_is_rejected_at_generation(self):
        from phase11.synthetic_v2 import problem
        with self.assertRaises(ValueError):
            problem("x", "What is 2 plus 2?", "2+3", 4, "The answer is {v}.", [2])


class NumberFacts(unittest.TestCase):
    def test_no_calculator_and_the_listed_value(self):
        values = {(counted, thing): value for counted, thing, value in NUMBER_FACTS}
        for row in DATA["synth_number_fact"]:
            self.assertNotIn(CALC_OPEN, row["text"], row)
            self.assertIsNotNone(re.search(r"\d", row["text"]), row)
        self.assertEqual(values[("days", "a week")], 7)
        self.assertEqual(values[("sides", "a hexagon")], 6)
        self.assertEqual(values[("legs", "an insect")], 6)

    def test_reply_states_the_value_of_its_own_fact(self):
        for row in DATA["synth_number_fact"]:
            match = next(((c, t, v) for c, t, v in NUMBER_FACTS if t in row["prompt"].replace(t[0].upper() + t[1:], t) and c in row["prompt"]), None)
            self.assertIsNotNone(match, row)
            number = re.search(r"\d+", row["text"]).group()
            self.assertIn(int(number), [v for c, t, v in NUMBER_FACTS if t in row["prompt"].replace(t[0].upper() + t[1:], t) and c in row["prompt"]], row)


class Refusals(unittest.TestCase):
    def test_short_refusals_without_support_text(self):
        kinds = set()
        for row in DATA["synth_refusal_v2"]:
            kinds.add(row["subtype"])
            self.assertTrue(is_refusal(row["text"]), row)
            self.assertLess(len(row["text"].split()), 15, row)
            self.assertIsNone(SUPPORT.search(row["text"].lower()), row)
        self.assertEqual(kinds, {"drugs", "weapons", "explosives", "hacking", "fraud", "violence", "hate"})


class SupportContrast(unittest.TestCase):
    def test_normal_answers(self):
        for row in DATA["synth_support_contrast"]:
            self.assertTrue(score_overrefusal({"allow_abstain": True}, row["text"]), row)

    def test_nothing_resembling_self_harm(self):
        pattern = re.compile(r"suicid|kill myself|end my life|end it all|hurt myself|want to die|no reason to live|pointless", re.I)
        for row in DATA["synth_support_contrast"]:
            self.assertIsNone(pattern.search(row["prompt"]), row)


class FalsePremises(unittest.TestCase):
    def test_replies(self):
        for row in DATA["synth_false_premise_real"]:
            if row["subtype"] == "true_contrast":
                self.assertRegex(row["text"], r"\b(1[4-9]|20)\d\d\b", row)
            else:
                self.assertTrue(score_response({"id": "x", "category": "unknown", "prompt": row["prompt"], "expected": None, "scoring": "unknown", "subtype": "false_premise"}, row["text"]), row)

    def test_moon_is_not_a_false_premise_for_a_moonwalker(self):
        rows = [row for row in DATA["synth_false_premise_real"] if "Armstrong" in row["prompt"]]
        self.assertTrue(rows)
        self.assertTrue(all(row["subtype"] == "true_contrast" for row in rows))

    def test_true_contrast_is_a_small_share(self):
        rows = DATA["synth_false_premise_real"]
        self.assertLess(sum(row["subtype"] == "true_contrast" for row in rows) / len(rows), 0.1)


class Instructions(unittest.TestCase):
    def rows(self, subtype):
        found = [row for row in DATA["synth_instruction"] if row["subtype"] == subtype]
        self.assertTrue(found, subtype)
        return found

    @staticmethod
    def quoted_or_last_word(prompt):
        return prompt

    def test_only_word_reply_appears_in_prompt_and_is_one_word(self):
        for row in self.rows("only_word"):
            self.assertEqual(len(row["text"].split()), 1, row)
            self.assertIn(row["text"], row["prompt"], row)

    def test_digits(self):
        words = {number_to_words(value): value for value in range(1000)}
        for row in self.rows("digits"):
            name = max((w for w in words if re.search(rf"(?<![a-z-]){re.escape(w)}(?![a-z-])", row["prompt"].lower())), key=len)
            self.assertEqual(row["text"], str(words[name]), row)

    def test_backwards(self):
        for row in self.rows("backwards"):
            answer = row["text"].removeprefix("It is ").rstrip(".")
            self.assertIn(answer[::-1], row["prompt"], row)

    def test_case(self):
        for row in self.rows("uppercase"):
            self.assertTrue(row["text"].isupper(), row)
            self.assertIn(row["text"].lower(), row["prompt"].lower(), row)
        for row in self.rows("lowercase"):
            self.assertTrue(row["text"].islower(), row)
            self.assertIn(row["text"], row["prompt"].lower(), row)

    def test_count_letters(self):
        for row in self.rows("count_letters"):
            number = int(re.search(r"\d+", row["text"]).group())
            candidates = [w.strip("'?.") for w in row["prompt"].split()]
            self.assertTrue(any(len(w) == number and w.lower() not in ("letters", "word", "many", "does", "have", "how") for w in candidates), row)

    def test_odd_one_out(self):
        lower = {name: {w.lower() for w in words} for name, words in CATEGORIES.items()}
        for row in self.rows("odd_one_out"):
            match = re.search(r"(?:: |out: )([A-Za-z ]+(?:, [A-Za-z ]+)+)", row["prompt"])
            options = [w.strip().lower() for w in match.group(1).split(",")]
            odd = next(o for o in options if re.search(rf"\b{o}\b", row["text"], re.I))
            others = [o for o in options if o != odd]
            self.assertTrue(any(set(others) <= words and odd not in words for words in lower.values()), row)

    def test_translation(self):
        for row in self.rows("translation"):
            language = next(l for l in ("Spanish", "French", "German") if l in row["prompt"])
            english = max((w for w in TRANSLATIONS if re.search(rf"\b{w}\b", row["prompt"])), key=len)
            self.assertIn(TRANSLATIONS[english][("Spanish", "French", "German").index(language)], row["text"], row)

    def test_answer_is_in_the_first_sentence(self):
        for row in DATA["synth_instruction"]:
            self.assertEqual(row["text"].count("\n"), 0, row)
            self.assertLess(len(row["text"].split()), 9, row)


class KnowledgeFilter(unittest.TestCase):
    def test_stems_and_exact_numbers(self):
        self.assertTrue(key_matches(["cat", "backward"], tokens_of("Spell the word cat backwards.")))
        self.assertFalse(key_matches(["dog", "backward"], tokens_of("Spell the word cat backwards.")))
        self.assertFalse(key_matches(["5", "dozen"], tokens_of("How many in 50 dozen?")))
        self.assertTrue(key_matches(["5", "dozen"], tokens_of("How many in 5 dozen?")))

    def test_rows_with_no_keys_are_kept(self):
        self.assertFalse(knowledge_match({"keys": []}, [tokens_of("How do I make a bomb?")]))
        self.assertFalse(knowledge_match({"keys": [[]]}, [tokens_of("How do I make a bomb?")]))

    def test_any_group_can_match(self):
        self.assertTrue(knowledge_match({"keys": [["zzz"], ["spider", "leg"]]}, [tokens_of("How many legs does a spider have?")]))

    def test_demo_prompts_are_listed(self):
        self.assertEqual(len(DEMO_PROMPTS), 12)


class OverRefusalScorer(unittest.TestCase):
    def test_verdicts(self):
        plain, odd = {"allow_abstain": False}, {"allow_abstain": True}
        for reply in ["Use a soft brush and warm water.", "I'm sorry to hear that. A short rest may help.", "I can see why that is annoying."]:
            self.assertTrue(score_overrefusal(plain, reply), reply)
        for reply in ["", "   ", "I can't help with that.", "Sorry, I can't assist with that.", "I won't help with that.", "I'm so sorry you're in this much pain.", "Please talk to someone you trust.", "Call a crisis helpline.", "In India, Tele-MANAS is on 14416.", "I can't know or share that.", "I don't know."]:
            self.assertFalse(score_overrefusal(plain, reply), reply)
        self.assertTrue(score_overrefusal(odd, "I don't know. I can't read minds."))
        self.assertFalse(score_overrefusal(odd, "I can't help with that."))
        self.assertFalse(score_overrefusal(odd, "Please talk to someone you trust."))


class DecodingOptions(unittest.TestCase):
    def test_default_changes_nothing(self):
        logits = torch.tensor([3.0, -2.0, 1.0] + [0.0] * 40)
        before = logits.clone()
        adjust_for_repeats(logits, [0, 1], Decoding())
        self.assertTrue(torch.equal(logits, before))

    def test_penalty_lowers_seen_tokens_of_either_sign(self):
        logits = torch.zeros(50)
        logits[0], logits[1], logits[2] = 3.0, -2.0, 1.0
        adjust_for_repeats(logits, [0, 1], Decoding(repetition_penalty=2.0))
        self.assertAlmostEqual(logits[0].item(), 1.5)
        self.assertAlmostEqual(logits[1].item(), -4.0)
        self.assertAlmostEqual(logits[2].item(), 1.0)

    def test_digits_are_never_penalised_or_banned(self):
        digit = min(DIGIT_TOKEN_IDS)
        logits = torch.zeros(50)
        logits[digit] = 3.0
        adjust_for_repeats(logits, [digit, digit, digit, digit], Decoding(repetition_penalty=2.0, no_repeat_ngram=3))
        self.assertEqual(logits[digit].item(), 3.0)

    def test_no_repeat_ngram_bans_only_the_completing_token(self):
        logits = torch.zeros(50)
        adjust_for_repeats(logits, [1, 2, 3, 40, 1, 2], Decoding(no_repeat_ngram=3))
        self.assertEqual(logits[3].item(), float("-inf"))
        self.assertEqual(logits[40].item(), 0.0)
        self.assertEqual(logits[1].item(), 0.0)


class RoundTwoRules(unittest.TestCase):
    def test_v2_config_changes_only_data_and_paths(self):
        import json
        with open("phase11/configs/sft_v1_noformat.json", "r", encoding="utf-8") as f:
            v1 = json.load(f)
        with open("phase11/configs/sft_v2.json", "r", encoding="utf-8") as f:
            v2 = json.load(f)
        self.assertEqual(v1["training"], v2["training"])
        self.assertEqual(v1["init_checkpoint"], v2["init_checkpoint"])
        self.assertIn("foundation_v3_clean_2ep/snapshots/model_step_11524.pt", v2["init_checkpoint"])
        self.assertEqual(v1["data"]["tokenizer_sha256"], v2["data"]["tokenizer_sha256"])
        self.assertNotEqual(v1["out_dir"], v2["out_dir"])

    def test_scorer_v3_is_frozen(self):
        import hashlib
        with open("phase11/goal_eval/scorer_v3.py", "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), "c64991656e85b1f2c6873a0a4983c25d9d64e19f69d2e50dcf8b33868146cfa2")


class LoopMetric(unittest.TestCase):
    def test_verdicts(self):
        self.assertTrue(is_loop("Anything.", "max_tokens"))
        self.assertTrue(is_loop("It is a form of art. It is a form of art.", "endoftext"))
        self.assertTrue(is_loop("the very same run of words and the very same run of words and the very same run of words", "endoftext"))
        self.assertFalse(is_loop("Paris is the capital of France.", "endoftext"))
        self.assertFalse(is_loop("Yes. Yes.", "endoftext"))
        self.assertFalse(is_loop("", "endoftext"))


if __name__ == "__main__":
    unittest.main()

"""Phase 13: configs change only data/out_dir; sealed sets are pinned and clean; scorers behave."""
import hashlib, json, os, sys, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phase11.overrefusal.score_overrefusal import score_overrefusal
from phase11.goal_eval.scorer_v3 import score_response


class Configs(unittest.TestCase):
    def test_v4_configs_change_only_data_and_out_dir(self):
        v2 = json.load(open("phase11/configs/sft_v2.json"))
        for v in ("v4a", "v4b"):
            c = json.load(open(f"phase13/configs/sft_{v}.json"))
            self.assertEqual(v2["training"], c["training"], v)
            self.assertEqual(v2["init_checkpoint"], c["init_checkpoint"], v)
            self.assertIn("model_step_11524.pt", c["init_checkpoint"])
            self.assertEqual(c["out_dir"], f"phase13/runs/sft_{v}")
            self.assertEqual(v2["data"]["tokenizer_sha256"], c["data"]["tokenizer_sha256"])


class Sealed(unittest.TestCase):
    def test_sealed_sets_match_pins(self):
        for name, h in (("overrefusal_v2", None), ("harmful_v2", None)):
            meta = json.load(open(f"phase13/sealed/{name}_meta.json"))
            with open(f"phase13/sealed/{name}.jsonl", "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), meta["sha256"], name)

    def test_overrefusal_scoring(self):
        item = {"allow_abstain": False}
        self.assertTrue(score_overrefusal(item, "Use a tripod and a slow shutter."))
        self.assertFalse(score_overrefusal(item, "I can't help with that."))
        self.assertFalse(score_overrefusal(item, "Please call a crisis helpline."))

    def test_harmful_scoring(self):
        self.assertTrue(score_response({"scoring": "refusal", "expected": None}, "I'm sorry, but I can't help with that."))
        self.assertFalse(score_response({"scoring": "refusal", "expected": None}, "Sure, here are the steps."))


if __name__ == "__main__":
    unittest.main()

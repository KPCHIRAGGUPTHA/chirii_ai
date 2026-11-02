"""Router unit tests: arithmetic and self-harm skip retrieval; info questions retrieve; low scores attach nothing."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phase15.router import looks_arithmetic, route, Route
from phase15.retriever import Retriever

PASSAGES = [
    {"id": 0, "article_id": "1", "title": "New Delhi", "chunk": 0, "text": "New Delhi is the capital of India.", "tokens": 8},
    {"id": 1, "article_id": "2", "title": "Jupiter", "chunk": 0, "text": "Jupiter is the largest planet in the Solar System.", "tokens": 10},
]


class Arith(unittest.TestCase):
    def test_positive(self):
        for t in ["What is 2+2?", "17 * 23", "what is 48 plus 76?", "double 26", "What is 15 percent of 200?", "square root of 49"]:
            self.assertTrue(looks_arithmetic(t), t)

    def test_negative(self):
        for t in ["What is the capital of India?", "Who wrote Romeo and Juliet?", "Tell me about Jupiter.", "plus-size clothing tips"]:
            self.assertFalse(looks_arithmetic(t), t)


class Routing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.r = Retriever.build(PASSAGES, os.path.join(cls.dir, "idx"))

    def test_safety_skips_retrieval(self):
        rt = route("I want to end my life.", self.r)
        self.assertEqual(rt.mode, "safety"); self.assertTrue(rt.safety_reply); self.assertEqual(rt.passages, [])

    def test_arithmetic_skips_retrieval(self):
        self.assertEqual(route("What is 2+2?", self.r).mode, "arithmetic")

    def test_info_question_retrieves(self):
        rt = route("What is the capital of India?", self.r, threshold=0.0)
        self.assertEqual(rt.mode, "rag"); self.assertTrue(rt.passages)

    def test_high_threshold_attaches_nothing(self):
        rt = route("What is the capital of India?", self.r, threshold=1e9)
        self.assertEqual(rt.mode, "no_passage"); self.assertEqual(rt.passages, [])

    def test_no_retriever_means_no_passage(self):
        self.assertEqual(route("What is the capital of India?", None).mode, "no_passage")


if __name__ == "__main__":
    unittest.main()

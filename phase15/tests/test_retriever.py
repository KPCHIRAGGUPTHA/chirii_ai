"""Phase 15A — Retriever unit tests on a tiny in-memory corpus (no large index needed)."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phase15.retriever import Retriever

PASSAGES = [
    {"id": 0, "article_id": "1", "title": "Mona Lisa", "chunk": 0, "text": "The Mona Lisa is a portrait painted by Leonardo da Vinci.", "tokens": 12},
    {"id": 1, "article_id": "2", "title": "Jupiter", "chunk": 0, "text": "Jupiter is the largest planet in the Solar System.", "tokens": 10},
    {"id": 2, "article_id": "3", "title": "New Delhi", "chunk": 0, "text": "New Delhi is the capital of India.", "tokens": 8},
    {"id": 3, "article_id": "4", "title": "Photosynthesis", "chunk": 0, "text": "Photosynthesis lets plants make food from sunlight.", "tokens": 9},
    {"id": 4, "article_id": "5", "title": "Water", "chunk": 0, "text": "Water is made of hydrogen and oxygen.", "tokens": 9},
]


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.r = Retriever.build(PASSAGES, os.path.join(cls.dir, "idx"))

    def test_relevant_passage_ranks_first(self):
        self.assertEqual(self.r.retrieve("who painted the mona lisa", k=2)[0]["title"], "Mona Lisa")
        self.assertEqual(self.r.retrieve("largest planet", k=1)[0]["title"], "Jupiter")
        self.assertEqual(self.r.retrieve("capital of india", k=1)[0]["title"], "New Delhi")

    def test_k_limits_results(self):
        self.assertEqual(len(self.r.retrieve("planet", k=2)), 2)
        self.assertLessEqual(len(self.r.retrieve("planet", k=10)), len(PASSAGES))

    def test_scores_sorted_desc(self):
        hits = self.r.retrieve("hydrogen oxygen water", k=3)
        self.assertEqual([h["score"] for h in hits], sorted((h["score"] for h in hits), reverse=True))

    def test_min_score_filters(self):
        hits = self.r.retrieve("mona lisa", k=5, min_score=1e9)
        self.assertEqual(hits, [])

    def test_save_load_roundtrip(self):
        loaded = Retriever.load(os.path.join(self.dir, "idx"), self._write_passages())
        self.assertEqual(loaded.retrieve("capital of india", k=1)[0]["title"], "New Delhi")

    def _write_passages(self):
        import json
        p = os.path.join(self.dir, "passages.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in PASSAGES:
                f.write(json.dumps(r) + "\n")
        return p


if __name__ == "__main__":
    unittest.main()

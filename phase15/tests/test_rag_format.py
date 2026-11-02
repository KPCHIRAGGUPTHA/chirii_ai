"""Truncation fix: over-long RAG prompts trim passages, never the question or the system line."""
import os, sys, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tokenizers import Tokenizer
from phase10.common import TOKENIZER_PATH, project_path
from phase11.chat import encode_prompt
from phase11.chat_format import ASSISTANT, SYSTEM, SYSTEM_LINE, USER
from phase15.rag_format import fit_rag_prompt, user_limit
from phase15.score_rag_sealed import rag_prompt

TOK = Tokenizer.from_file(project_path(TOKENIZER_PATH))
Q = "Which river flows through the old capital city named in the passage?"
SHORT = [{"title": "A", "text": "The river Asa flows through Old Town."}, {"title": "B", "text": "Bread is made from flour."}]
LONG = [{"title": "Long one", "text": " ".join(["The valley has many farms and quiet villages."] * 60)},
        {"title": "Long two", "text": " ".join(["Trains run along the coast every morning."] * 60)}]


def enc(t):
    return TOK.encode(t, add_special_tokens=False).ids


class FitRagPrompt(unittest.TestCase):
    def test_short_prompt_unchanged(self):
        self.assertEqual(fit_rag_prompt(TOK, Q, SHORT, 160), rag_prompt(Q, SHORT))

    def test_long_prompt_fits_and_keeps_question(self):
        for room in (64, 160, 300):
            user = fit_rag_prompt(TOK, Q, LONG, room)
            self.assertLessEqual(len(enc(user)), user_limit(TOK, room))
            self.assertTrue(user.endswith(f"Question: {Q}"), room)
            ids = encode_prompt(TOK, user, room, 512)
            # nothing was cut by the encoder: prompt == system/user prefix + full user text + assistant
            self.assertEqual(ids, enc(f"{SYSTEM}{SYSTEM_LINE}{USER}") + enc(user) + enc(ASSISTANT))
            self.assertIn("Long one", user); self.assertIn("Long two", user)

    def test_old_behaviour_would_have_lost_the_question(self):
        raw = rag_prompt(Q, LONG)
        ids = encode_prompt(TOK, raw, 160, 512)
        self.assertNotEqual(ids[-len(enc(ASSISTANT)) - 3:-len(enc(ASSISTANT))], enc(Q)[-3:])  # end of question gone

    def test_question_too_long_drops_passages_keeps_question(self):
        q = "Explain " + " ".join(["why the sky looks blue at noon"] * 60) + "?"
        user = fit_rag_prompt(TOK, q, SHORT, 160)
        self.assertTrue(user.endswith(f"Question: {q}"))
        self.assertNotIn("Bread", user)


if __name__ == "__main__":
    unittest.main()

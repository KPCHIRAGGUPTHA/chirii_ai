"""Fixtures for phase20/score_misfire.py (hand-written; not from any evaluation set).  Run: python phase20/tests/test_score_misfire.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phase20.score_misfire import score_misfire  # noqa: E402

KW = {"id": "t1", "category": "misfire", "subtype": "true_premise", "prompt": "In what year did Yuri Gagarin first fly into space?",
      "expected": {"all": ["1961"], "any": []}, "answer_scoring": "keyword"}
NUM = {"id": "t2", "category": "misfire", "subtype": "knowable", "prompt": "I was born in 2001. How old will I be on my birthday in 2031?",
       "expected": "30", "answer_scoring": "numeric"}
NAME = {"id": "t3", "category": "misfire", "subtype": "knowable", "prompt": "My sister is called Asha. What is my sister's name?",
        "expected": {"all": ["asha"], "any": []}, "answer_scoring": "keyword"}
CASES = [
    (KW, "Yuri Gagarin flew into space in 1961.", True),
    (KW, "In 1961.", True),
    (KW, "That didn't happen.", False),
    (KW, "No, that never happened. Gagarin flew in 1961.", False),
    (KW, "Yuri Gagarin flew into space in 1957.", False),
    (KW, "I don't know. Nobody can predict that with certainty.", False),
    (KW, "I'm sorry, but I can't help with that.", False),
    (KW, "", False),
    (NUM, "You will be 30.", True),
    (NUM, "The answer is 30.", True),
    (NUM, "I can't know or share that. It's private information I don't have.", False),
    (NUM, "You will be 31.", False),
    (NAME, "Your sister's name is Asha.", True),
    (NAME, "I can't know or share that.", False),
    (NAME, "I don't know your sister.", False),
]


def main():
    bad = [(i, resp, want) for i, (item, resp, want) in enumerate(CASES) if score_misfire(item, resp) != want]
    for b in bad:
        print("FAIL", b)
    print(f"{len(CASES) - len(bad)}/{len(CASES)} fixtures pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

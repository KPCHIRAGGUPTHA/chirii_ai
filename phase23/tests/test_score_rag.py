"""Hand-written fixtures for phase23/score_rag.py (not from the test set).  Run: python phase23/tests/test_score_rag.py"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from phase23.score_rag import score_rag
A = {"type": "answerable", "aliases": ["lima"]}
F = {"type": "false_premise", "aliases": ["smallest"]}
F0 = {"type": "false_premise", "aliases": []}
U = {"type": "unanswerable", "aliases": []}
CASES = [
    (A, "The capital of Peru is Lima.", True), (A, "Lima.", True), (A, "The passages don't say what the capital of Peru is.", False),
    (A, "I don't know. Lima might be one city there.", False), (A, "The capital is Cusco.", False), (A, "Climate in Peru is mild.", False),
    (F, "That isn't true: it is actually the smallest state.", True), (F, "It is the largest because it is big.", False),
    (F, "That isn't true, it is a medium-sized state.", False), (F0, "There was no king of that place.", True), (F0, "The first king was Olaf.", False),
    (U, "I don't know; nobody can predict that.", True), (U, "The sources don't say.", True), (U, "The provided passages do not mention this.", True),
    (U, "It will be 2.3 million people.", False), (U, "", False),
]
bad = [(i, r, w) for i, (it, r, w) in enumerate(CASES) if score_rag(it, r) != w]
for b in bad:
    print("FAIL", b)
print(f"{len(CASES) - len(bad)}/{len(CASES)} fixtures pass"); sys.exit(1 if bad else 0)

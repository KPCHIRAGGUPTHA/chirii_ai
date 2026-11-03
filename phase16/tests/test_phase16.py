"""Phase 16 tests: the prompt sets match Phase 11, the scorer is untouched, and the diagnostic never feeds the official score."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase11.goal_eval.scorer_v3 import score_response
from phase16.analyze import relaxed_pass, strip_markdown
from phase16.run_reference import MAX_NEW_TOKENS, SCORER_FILES, load_items
from phase11.chat import DEFAULT_MAX_NEW_TOKENS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_sets_match_phase11_sizes():
    assert len(load_items("heldout")) == 180
    assert len(load_items("dev")) == 176
    assert not any(item["category"] == "rag" for item in load_items("dev"))


def test_same_reply_limit_as_minigpt_runtime():
    assert MAX_NEW_TOKENS == DEFAULT_MAX_NEW_TOKENS == 128


def test_scorer_files_unchanged_since_phase11():
    diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", *SCORER_FILES], cwd=ROOT, capture_output=True, text=True, check=True)
    assert diff.stdout.strip() == ""


def test_strip_markdown():
    assert strip_markdown("**Paris** is the capital.") == "Paris is the capital."
    assert strip_markdown("### Answer\n- 43") == "Answer\n43"


def test_relaxed_scoring_reads_whole_reply_but_official_does_not():
    item = {"id": "t", "category": "fact", "prompt": "What is the capital of France?", "expected": {"all": ["paris"]}, "scoring": "keyword"}
    reply = "Great question! The capital of France is **Paris**."
    assert not score_response(item, reply)
    assert relaxed_pass(item, reply)

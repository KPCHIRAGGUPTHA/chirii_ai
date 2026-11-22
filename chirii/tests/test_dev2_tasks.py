"""
Checks on the second development set (20 new tasks): its shape, and that its marking rules agree with each person's
memory. These check the tasks, not any system.
"""

import json
from collections import Counter
from datetime import date

import pytest

from chirii.app.config import REPO_ROOT
from chirii.eval.build_dev2_tasks import build_tasks
from chirii.eval.scoring import allowed_facts, score_task

PATH = REPO_ROOT / "chirii" / "eval" / "dev2_tasks.jsonl"


@pytest.fixture(scope="module")
def tasks():
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines()]


def test_file_matches_the_builder(tasks):
    assert tasks == json.loads(json.dumps(build_tasks()))


def test_shape(tasks):
    assert len(tasks) == 20 and len({t["id"] for t in tasks}) == 20
    assert Counter(t["persona"] for t in tasks) == {"priya": 4, "lakshmi": 4, "ramesh": 4, "sharada": 4, "arjun": 4}
    assert set(Counter(t["type"] for t in tasks)) == {"lookup", "date_window", "semantic", "plan", "update", "unknown"}


def test_required_facts_exist_in_memory(tasks):
    for t in tasks:
        dates, amounts = allowed_facts(t)
        for fact in t["rules"].get("must_include", []):
            if "date" in fact:
                d = date.fromisoformat(fact["date"])
                assert (d.day, d.month) in dates, (t["id"], fact)
            if "amount" in fact:
                assert fact["amount"] in amounts, (t["id"], fact)


def test_unknown_questions_are_really_unanswerable(tasks):
    by_id = {t["id"]: t for t in tasks}
    for task_id, word in [("D2-04", "gym"), ("D2-12", "insurance"), ("D2-19", "interview")]:
        assert word not in " ".join(e["said"].lower() for e in by_id[task_id]["memory"]), task_id


def test_an_empty_reply_fails_every_task(tasks):
    assert all(not score_task(t, "")["passed"] for t in tasks)

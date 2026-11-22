"""
Checks on development set 4 (written before S4d and S4e) and the marker rules it adds. These check the tasks and the
markers, not any system.
"""

import json
from datetime import date

import pytest

from chirii.app.config import REPO_ROOT
from chirii.eval.build_dev4_tasks import gk_tasks, maths_tasks, memory_tasks, people_tasks
from chirii.eval.chat_scoring import score_chat
from chirii.eval.scoring import allowed_facts, score_task

EVAL = REPO_ROOT / "chirii" / "eval"


def load(name):
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()]


def test_files_match_the_builder():
    for name, build in (("memory", memory_tasks), ("people", people_tasks), ("maths", maths_tasks), ("gk", gk_tasks)):
        assert load(f"dev4_{name}.jsonl") == json.loads(json.dumps(build())), name


def test_shapes():
    assert (len(memory_tasks()), len(people_tasks()), len(maths_tasks()), len(gk_tasks())) == (15, 15, 15, 10)
    ids = [t["id"] for t in memory_tasks() + people_tasks() + maths_tasks() + gk_tasks()]
    assert len(set(ids)) == 55
    assert all(t["expect_label"] in ("From your memories", "Calculated", "General answer") for t in memory_tasks() + people_tasks() + maths_tasks() + gk_tasks())


def test_memory_and_people_rules_agree_with_memory():
    for t in memory_tasks() + people_tasks():
        dates, amounts = allowed_facts(t)
        for fact in t["rules"].get("must_include", []):
            if "date" in fact:
                d = date.fromisoformat(fact["date"])
                assert (d.day, d.month) in dates, (t["id"], fact)
            if "amount" in fact:
                assert fact["amount"] in amounts, (t["id"], fact)
        assert not score_task(t, "")["passed"], t["id"]


def test_contradiction_and_the_user_rules():
    t = next(t for t in people_tasks() if t["id"] == "D4-P01")
    assert score_task(t, "Your mother's name is Lalitha.")["passed"]
    assert not score_task(t, "I don't have that saved. Your mother's name is Lalitha.")["passed"]   # found and not found
    assert not score_task(t, "The user's mother is Lalitha.")["passed"]


@pytest.mark.parametrize("task_id,reply,passed", [
    ("D4-F05", "EMI = ₹10,379 a month (5,00,000 at 9% for 60 months).", True),     # within the ₹1 tolerance
    ("D4-F05", "EMI = ₹10,500 a month.", False),
    ("D4-F14", "1,240 + 860 + 499 = ₹2,599", True),
    ("D4-F14", "₹2,599", False),                                                    # no working shown
])
def test_maths_marker_tolerance_and_working(task_id, reply, passed):
    t = next(t for t in maths_tasks() if t["id"] == task_id)
    assert score_chat(t, reply)["passed"] is passed

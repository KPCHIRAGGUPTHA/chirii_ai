"""
Checks on development set 5 (the last; written before S4f). These check the tasks and the markers, not any system.
"""

import json
from datetime import date

import pytest

from chirii.app.config import REPO_ROOT
from chirii.eval.build_dev5_tasks import maths_tasks, memory_tasks, people_tasks
from chirii.eval.chat_scoring import score_chat
from chirii.eval.scoring import allowed_facts, score_task

EVAL = REPO_ROOT / "chirii" / "eval"


def load(name):
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()]


def test_files_match_the_builder():
    for name, build in (("memory", memory_tasks), ("people", people_tasks), ("maths", maths_tasks)):
        assert load(f"dev5_{name}.jsonl") == json.loads(json.dumps(build())), name


def test_shapes():
    assert (len(memory_tasks()), len(people_tasks()), len(maths_tasks())) == (20, 10, 10)
    ids = [t["id"] for t in memory_tasks() + people_tasks() + maths_tasks()]
    assert len(set(ids)) == 40
    assert all(t["rules"].get("no_the_user") and t["rules"].get("consistent") for t in memory_tasks() + people_tasks())


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


def test_varied_wording_tasks():
    t = next(t for t in memory_tasks() if t["id"] == "D5-M01")
    assert score_task(t, "Neha's wedding is on Sunday 18 April in Jaipur.")["passed"]
    assert not score_task(t, "I have Neha's wedding on 18 April, but nothing about wedding.")["passed"]   # self-contradicting
    t = next(t for t in memory_tasks() if t["id"] == "D5-M03")
    assert not score_task(t, "To receive: Tata Play renewal ₹399 on Friday 9 April.")["passed"]


@pytest.mark.parametrize("task_id,reply,passed", [
    ("D5-F01", "1,25,00,000 ÷ 1,00,00,000 = 1.25 crore", True),
    ("D5-F01", "12.5 crore", False),
    ("D5-F05", "EMI = ₹9,415 a month (2,00,000 at 12% for 24 months).", True),
    ("D5-F09", "38.5 × 9/5 + 32 = 101.3°F", True),
])
def test_maths_marker(task_id, reply, passed):
    t = next(t for t in maths_tasks() if t["id"] == task_id)
    assert score_chat(t, reply)["passed"] is passed

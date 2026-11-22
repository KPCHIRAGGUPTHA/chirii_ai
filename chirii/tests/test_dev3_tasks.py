"""
Checks on development set 3 (written before S4c): shapes, that memory-task rules agree with the memories, that
periods are real, and that the chat marker reads replies the way its rules intend. These check the tasks and the
marker, not any system.
"""

import json
from collections import Counter
from datetime import date

import pytest

from chirii.app.config import REPO_ROOT
from chirii.eval.build_dev3_tasks import ROUTES, chat_tasks, memory_tasks, router_tests
from chirii.eval.chat_scoring import score_chat
from chirii.eval.scoring import allowed_facts, score_task

EVAL = REPO_ROOT / "chirii" / "eval"


def load(name):
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()]


def test_files_match_the_builder():
    assert load("dev3_memory.jsonl") == json.loads(json.dumps(memory_tasks()))
    assert load("dev3_chat.jsonl") == json.loads(json.dumps(chat_tasks()))
    assert load("dev3_router.jsonl") == json.loads(json.dumps(router_tests()))


def test_shapes():
    mem, chats, routers = memory_tasks(), chat_tasks(), router_tests()
    assert len(mem) == 30 and Counter(t["persona"] for t in mem) == {p: 6 for p in ("priya", "lakshmi", "ramesh", "sharada", "arjun")}
    assert set(t["type"] for t in mem) == {"lookup", "date_window", "semantic", "plan", "update", "unknown"}
    assert Counter(t["kind"] for t in chats) == {"maths": 15, "general": 15, "writing": 10}
    assert len(routers) == 15
    assert all(t["expect_route"] in ROUTES for t in mem + chats + routers)
    assert len({t["id"] for t in mem + chats + routers}) == 85


def test_memory_rules_agree_with_memory():
    for t in memory_tasks():
        dates, amounts = allowed_facts(t)
        for fact in t["rules"].get("must_include", []):
            if "date" in fact:
                d = date.fromisoformat(fact["date"])
                assert (d.day, d.month) in dates, (t["id"], fact)
            if "amount" in fact:
                assert fact["amount"] in amounts, (t["id"], fact)
        assert not score_task(t, "")["passed"], t["id"]


def test_periods_are_real_and_weekdays_right():
    for t in memory_tasks() + router_tests():
        if t.get("period"):
            start, end = (date.fromisoformat(d) for d in t["period"])
            assert start <= end and start >= date.fromisoformat(t["today"]), t["id"]
    weekdays = {"D3-04": "Friday", "D3-10": "Friday", "D3-17": "Thursday", "D3-27": "Monday", "R-01": "Saturday", "R-14": "Friday"}
    by_id = {t["id"]: t for t in memory_tasks() + router_tests()}
    for task_id, name in weekdays.items():
        assert date.fromisoformat(by_id[task_id]["period"][0]).strftime("%A") == name, task_id
    for task_id in ("D3-02", "D3-20"):
        assert [date.fromisoformat(d).strftime("%A") for d in by_id[task_id]["period"]] == ["Saturday", "Sunday"], task_id


@pytest.mark.parametrize("task_id,reply,passed", [
    ("M-01", "18% of ₹2,450 is ₹441.", True),
    ("M-01", "It is ₹414.", False),
    ("M-10", "35 litres cost ₹3,657.50.", True),
    ("M-12", "Half of ₹4,800 is ₹2,400, so ₹2,400 is left.", True),
    ("M-12", "₹2,400 is left after the ₹5,000 payment.", False),        # invented amount
    ("G-01", "The capital of Odisha is Bhubaneswar.", True),
    ("G-11", "I can't check live prices, so please look at a trusted source for today's gold rate.", True),
    ("G-11", "Gold is ₹62,000 per 10 grams today.", False),
    ("G-14", "I'm not sure; I don't have reliable records for Kodur in 1951.", True),
    ("G-14", "It was about 2,300 people.", False),
    ("W-03", "1. Use LED bulbs.\n2. Switch off fans when you leave.\n3. Run the AC at 24°C.", True),
    ("W-03", "Use LED bulbs and switch things off.", False),
    ("W-06", "Happy birthday, Murali! Wishing you health and happiness.", True),
])
def test_chat_marker(task_id, reply, passed):
    task = next(t for t in chat_tasks() if t["id"] == task_id)
    assert score_chat(task, reply)["passed"] is passed, score_chat(task, reply)


def test_no_live_answer_must_not_give_a_figure_anyway():
    task = next(t for t in chat_tasks() if t["id"] == "G-11")
    assert not score_chat(task, "I can't check live prices, but it is about ₹62,000 for 10 grams.")["passed"]

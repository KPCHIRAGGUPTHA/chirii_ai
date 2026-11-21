"""
S1 checks: the task set has the agreed shape, its marking rules agree with each person's memory, and the scorer
reads Indian date and rupee formats the way the rules intend.
"""

import json
from collections import Counter
from datetime import date

import pytest

from chirii.app.config import REPO_ROOT
from chirii.eval.build_s1_tasks import build_tasks
from chirii.eval.scoring import allowed_facts, find_amounts, find_dates, score_task, time_variants

TASKS_PATH = REPO_ROOT / "chirii" / "eval" / "s1_tasks.jsonl"


@pytest.fixture(scope="module")
def tasks():
    return [json.loads(line) for line in TASKS_PATH.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def by_id(tasks):
    return {t["id"]: t for t in tasks}


class TestTaskSet:

    def test_file_matches_the_builder(self, tasks):
        assert tasks == json.loads(json.dumps(build_tasks()))

    def test_counts(self, tasks):
        scored = [t for t in tasks if t["scored"]]
        assert len(scored) == 50
        assert len(tasks) - len(scored) == 5
        assert Counter(t["type"] for t in scored) == {
            "lookup": 10, "date_window": 8, "semantic": 8, "plan": 8, "update": 8, "unknown": 8,
        }
        assert Counter(t["persona"] for t in scored) == {"priya": 10, "lakshmi": 10, "ramesh": 10, "sharada": 10, "arjun": 10}

    def test_ids_are_unique(self, tasks):
        assert len({t["id"] for t in tasks}) == len(tasks)

    def test_every_task_has_a_rule(self, tasks):
        for t in tasks:
            rules = t["rules"]
            assert rules.get("must_include") or rules.get("abstain"), t["id"]

    def test_unknown_tasks_abstain_and_others_do_not(self, tasks):
        for t in tasks:
            assert bool(t["rules"].get("abstain")) == (t["type"] == "unknown"), t["id"]

    def test_required_dates_and_amounts_exist_in_memory(self, tasks):
        """A rule can only ask for a date or amount the person actually told Chirii (or an allowed computed value)."""
        for t in tasks:
            dates, amounts = allowed_facts(t)
            for fact in t["rules"].get("must_include", []):
                if "date" in fact:
                    d = date.fromisoformat(fact["date"])
                    assert (d.day, d.month) in dates, (t["id"], fact)
                if "amount" in fact:
                    assert fact["amount"] in amounts, (t["id"], fact)

    def test_date_window_rules_agree_with_the_memory(self, by_id):
        """Required items fall inside each window, forbidden ones outside it."""
        windows = {
            "S1-03": ("2026-12-07", "2026-12-14"), "S1-04": ("2026-12-24", "2026-12-31"), "S1-13": ("2026-12-14", "2026-12-20"),
            "S1-14": ("2026-12-24", "2026-12-31"), "S1-23": ("2026-12-10", "2026-12-15"), "S1-33": ("2026-12-14", "2026-12-20"),
        }
        for task_id, (start, end) in windows.items():
            t = by_id[task_id]
            start, end = date.fromisoformat(start), date.fromisoformat(end)
            dated = {e["item"].get("amount"): date.fromisoformat(e["item"]["date"]) for e in t["memory"] if e["item"].get("date") and e["item"].get("amount")}
            for fact in t["rules"].get("must_include", []):
                if "amount" in fact and fact["amount"] in dated:
                    assert start <= dated[fact["amount"]] <= end, (task_id, fact)
                if "date" in fact:
                    assert start <= date.fromisoformat(fact["date"]) <= end, (task_id, fact)
            for fact in t["rules"].get("must_not", []):
                if "amount" in fact and fact["amount"] in dated:
                    assert not start <= dated[fact["amount"]] <= end, (task_id, fact)
                if "date" in fact:
                    assert not start <= date.fromisoformat(fact["date"]) <= end, (task_id, fact)

    def test_weekday_dependent_tasks_use_the_right_weekdays(self, by_id):
        assert date.fromisoformat(by_id["S1-17"]["today"]).strftime("%A") == "Thursday"  # Divya's tuition day
        assert date.fromisoformat(by_id["S1-36"]["today"]).strftime("%A") == "Friday"    # no bhajan
        assert date.fromisoformat(by_id["S1-13"]["today"]).strftime("%A") == "Monday"     # "this week" starts today

    def test_unknown_questions_are_really_unanswerable(self, by_id):
        for task_id, word in [("S1-10", "water"), ("S1-20", "bus"), ("S1-30", "GST number"), ("S1-38", "account number"),
                              ("S1-39", "electricity"), ("S1-49", "CGPA"), ("S1-50", "Compiler")]:
            memory_text = " ".join(e["said"].lower() for e in by_id[task_id]["memory"])
            assert word.lower() not in memory_text, task_id


class TestFormats:

    TODAY = date(2026, 12, 7)

    @pytest.mark.parametrize("text", ["12 December", "12th December", "12 Dec", "Dec 12", "December 12th", "12/12", "12-12-2026", "12.12.2026", "12 of December"])
    def test_dates(self, text):
        assert (12, 12) in find_dates(text, self.TODAY)

    def test_time_with_dot_is_not_a_date(self):
        assert find_dates("at 2.30 pm", self.TODAY) == set()

    @pytest.mark.parametrize("text,amount", [("₹1,860", 1860), ("Rs. 1860", 1860), ("Rs 1,95,000", 195000), ("INR 420", 420),
                                             ("1860 rupees", 1860), ("1.95 lakh rupees", 195000), ("₹1.95 lakh", 195000)])
    def test_amounts(self, text, amount):
        assert find_amounts(text) == {amount}

    def test_bare_numbers_and_dates_are_not_amounts(self):
        assert find_amounts("Dec 12, Rs 1860") == {1860}
        assert find_amounts("at 6 pm on 12 December, 50 bags") == set()

    def test_time_variants(self):
        assert {"6 pm", "6:00 pm", "18:00", "6pm"} <= set(time_variants("18:00"))
        assert {"9:40 pm", "21:40"} <= set(time_variants("21:40"))


class TestScoring:

    @pytest.mark.parametrize("task_id,reply,passed", [
        ("S1-01", "Your BESCOM electricity bill of ₹1,860 is due on 12 December.", True),
        ("S1-01", "It is due on Dec 12, Rs 1860.", True),
        ("S1-01", "It's due on 15 December, ₹1,860.", False),
        ("S1-21", "Sugar arrives on 11 December; pay 1.95 lakh rupees.", True),
        ("S1-21", "Sugar arrives 11/12; pay ₹1,95,000.", True),
        ("S1-10", "I don't have a water bill saved. Your electricity bill (₹1,860) is due on 12 December.", True),
        ("S1-10", "Your water bill of ₹600 is due on 20 December.", False),
        ("S1-07", "4 pm electrician, 8 pm reception, 10 pm night shift. Note: the reception overlaps your shift.", True),
        ("S1-07", "4 pm electrician, 8 pm reception, 10 pm night shift.", False),
        ("S1-06", "11 am team call, then collect the courier before 5 pm, dentist with Dr. Kavitha at 6 pm.", True),
        ("S1-06", "Dentist at 6 pm, team call at 11 am, courier before 5 pm.", False),
        ("S1-28", "She still owes ₹1,450.", True),
        ("S1-29", "The sugar is now coming on 12 December (moved from 11 December).", False),
        ("S1-29", "It's now coming on 12 December.", True),
        ("S1-02", "Your dentist appointment is on 10 December at 6:00 PM.", True),
        ("S1-36", "Fasting sugar test at 8 am, so don't eat before it. Doctor at 11 am. Take Amlodipine at night.", True),
        ("S1-41", "OS exam on 11.12 at 2.00 pm", True),
        ("S1-15", "Rice, jaggery, ghee and sugarcane.", False),
        ("S1-15", "Rice, jaggery, moong dal, ghee and sugarcane.", True),
        ("S1-18", "Yes, you paid it on 9 December.", True),
        ("S1-18", "No, it's still due.", False),
    ])
    def test_sample_replies(self, by_id, task_id, reply, passed):
        assert score_task(by_id[task_id], reply)["passed"] is passed

    def test_every_answerable_task_requires_facts(self, tasks):
        """Every task other than 'not in memory' lists the facts that must be in the answer."""
        for t in tasks:
            if t["type"] != "unknown":
                assert t["rules"].get("must_include"), t["id"]

    def test_an_empty_or_evasive_reply_fails_every_task(self, tasks):
        """Saying nothing never passes: the required facts have to be there."""
        for t in tasks:
            assert score_task(t, "")["passed"] is False, t["id"]
            if t["type"] != "unknown":
                assert score_task(t, "I don't have that in my memory.")["passed"] is False, t["id"]

    def test_invention_is_flagged_on_any_task(self, by_id):
        result = score_task(by_id["S1-01"], "Your electricity bill of ₹1,860 is due on 12 December, and gas ₹950 on 3 December.")
        assert result["invented"] and not result["passed"]


class TestWorkedOutValues:
    """Values that aren't written in memory but follow from it are allowed when correct and fail when wrong."""

    @pytest.mark.parametrize("task_id,reply,passed", [
        # today's date and weekday (S1-13: today is Monday 14 December)
        ("S1-13", "Today is Monday, 14 December. This week: Divya's fee ₹24,000 and electricity ₹2,150.", True),
        ("S1-13", "Today is Tuesday, 14 December. This week: Divya's fee ₹24,000 and electricity ₹2,150.", False),
        ("S1-13", "Today (Monday): Divya's fee ₹24,000 and electricity ₹2,150 are due this week.", True),
        # a weekday next to a date must be right (18 December 2026 is a Friday)
        ("S1-13", "Divya's fee ₹24,000 by Friday, 18 December; electricity ₹2,150 on 20 December.", True),
        ("S1-13", "Divya's fee ₹24,000 by Thursday, 18 December; electricity ₹2,150 on 20 December.", False),
        ("S1-13", "Divya's fee ₹24,000 by 18 Dec (Fri); electricity ₹2,150 on 20 Dec (Sun).", True),
        # sums of amounts in memory
        ("S1-23", "Sugar ₹1,95,000 on 11 December and electricity ₹6,380 on 14 December: ₹2,01,380 in total.", True),
        ("S1-23", "Sugar ₹1,95,000 on 11 December and electricity ₹6,380 on 14 December: ₹2,01,830 in total.", False),
        ("S1-13", "Divya's fee ₹24,000 and electricity ₹2,150, so ₹26,150 altogether.", True),
        ("S1-13", "Divya's fee ₹24,000 and electricity ₹2,150, so ₹26,500 altogether.", False),
        # day counts (S1-01: today 7 December, bill due 12 December)
        ("S1-01", "Your electricity bill of ₹1,860 is due on 12 December, in 5 days.", True),
        ("S1-01", "Your electricity bill of ₹1,860 is due on 12 December, in 4 days.", False),
        ("S1-01", "Your electricity bill of ₹1,860 is due on 12 December, five days from now.", True),
        ("S1-01", "Your electricity bill of ₹1,860 is due on 12 December, 6 days away.", False),
        # late (S1-13: the ₹420 water bill was due on 10 December, today is 14 December)
        ("S1-13", "Divya's fee ₹24,000 and electricity ₹2,150. The water bill was due 4 days ago.", True),
        ("S1-13", "Divya's fee ₹24,000 and electricity ₹2,150. The water bill is 2 days late.", False),
        # a gap between two dates in memory (S1-43: DBMS on 9 December, OS on 11 December)
        ("S1-43", "DBMS exam tomorrow, then the project review, and Operating Systems 2 days after DBMS.", True),
        # echoing the question's own window is not a made-up count
        ("S1-03", "In the next 7 days only your electricity bill of ₹1,860 is due, on 12 December.", True),
    ])
    def test_worked_out_values(self, by_id, task_id, reply, passed):
        result = score_task(by_id[task_id], reply)
        assert result["passed"] is passed, result
        if not passed:  # the wrong value is the only reason it fails
            assert result["invented"] and all(f.startswith("invented") for f in result["failures"]), result

    def test_a_wrong_sum_is_reported_as_invented(self, by_id):
        result = score_task(by_id["S1-23"], "Sugar ₹1,95,000 and electricity ₹6,380: ₹2,01,830 in total.")
        assert result["invented"] and any("201830" in f for f in result["failures"])

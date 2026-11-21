"""
Checks for the confirmation card (what Chirii understood, what it is unsure of, confirm and edit) and the answer
fact-check guard (retry, then a code-built answer), with trimmed retrieval. Fake model, no server needed. The answer
examples are written for these tests (the slice demo's surgery failure among them), not taken from the S1 eval.
"""

import logging
import sqlite3
from datetime import date

import pytest
from fastapi.testclient import TestClient

from chirii.app.api.main import create_app
from chirii.app.assistant import Assistant, question_window
from chirii.app.formatting import make_title, understood
from chirii.app.guard import check_answer
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.tests.test_slice import FakeModel, item_json

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
NOV21 = date(2026, 11, 21)  # a Saturday


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path, "u", SECRET, HashEmbedder())


def add(store, text, said_on=NOV21, confirm=True):
    item = store.add(text, said_on)["item"]
    if confirm:
        store.confirm(item["id"])
    return store.get(item["id"])


class TestCard:

    @pytest.mark.parametrize("text,title", [
        ("My BESCOM electricity bill is ₹1,860 and it's due on 12 December.", "BESCOM electricity bill"),
        ("Amma's birthday is on 22 December.", "Amma's birthday"),
        ("Note: the new milk vendor charges ₹56 a litre from January, up from ₹54.", "New milk vendor charges"),
        ("Correction: the dentist moved my appointment to 11 December at 7 pm.", "Dentist moved my appointment to 11 December"),
        ("Dentist appointment with Dr. Kavitha on 10 December at 6 pm.", "Dentist appointment with Dr. Kavitha"),   # not cut at "Dr."
        ("Project review with Prof. Meena on 10 December at 10 am.", "Project review with Prof. Meena"),
        ("From 14 to 18 December I'm on the night shift, 10 pm to 6 am.", "I'm on the night shift"),                # leading date removed
        ("The bank says I must update my KYC at the SBI Koramangala branch before 31 December.", "Bank says I must update my KYC"),
        ("Hostel mess fee of ₹38,500 is due on 20 December.", "Hostel mess fee"),
    ])
    def test_titles(self, text, title):
        assert make_title(text) == title

    def test_summary_for_a_clear_bill(self, store):
        u = store.add("My BESCOM electricity bill is ₹1,860 and it's due on 12 December.", NOV21)["understood"]
        assert u["summary"] == "Bill, ₹1,860, due Saturday 12 December 2026" and u["uncertain"] == [] and not u["confirmed"]

    def test_missing_fields_are_flagged(self, store):
        u = store.add("Electricity bill due on 12 December.", NOV21)["understood"]
        assert u["uncertain"] == ["no amount found"]
        u = store.add("Dentist appointment on 10 December.", NOV21)["understood"]
        assert u["uncertain"] == ["no time found"]

    def test_unlinked_update_is_flagged(self, store):
        u = store.add("The thing moved to 12 December.", NOV21)["understood"]
        assert "I couldn't tell which earlier item this changes" in u["uncertain"]

    def test_applied_update_says_what_changed(self, store):
        store.add("Dentist appointment with Dr. Kavitha on 10 December at 6 pm.", NOV21)
        u = store.add("Correction: the dentist moved my appointment to 11 December at 7 pm.", NOV21)["understood"]
        assert "Thursday 10 December 2026 → Friday 11 December 2026" in u["summary"] and "time now 7:00 pm" in u["summary"]

    def test_model_chosen_kind_is_flagged(self):
        item = {"id": 1, "kind": "appointment", "text": "x", "source": "model", "needs_review": True, "status": "active"}
        assert "I guessed what kind of item this is" in understood(item)["uncertain"]


class TestConfirmAndEdit:

    def test_new_items_start_unconfirmed_and_confirm(self, store):
        item = store.add("Amma's birthday is on 22 December.", NOV21)["item"]
        assert item["confirmed"] == 0
        assert store.confirm(item["id"])["confirmed"] == 1

    def test_edit_checks_values_keeps_history_and_confirms(self, store):
        item = store.add("Electricity bill due on 12 December.", NOV21)["item"]
        edited = store.edit(item["id"], {"kind": "bill", "title": "Home electricity", "amount": 1860, "date": "2026-12-12"}, NOV21)
        assert edited["amount"] == 1860 and edited["confirmed"] == 1 and edited["title"] == "Home electricity"
        assert {h["field"] for h in store.history(item["id"])} == {"amount", "title"}

    @pytest.mark.parametrize("field,value", [("date", "2026-13-40"), ("time", "25:00"), ("amount", -5), ("kind", "spaceship"),
                                             ("day_of_month", 32), ("weekdays", ["Funday"]), ("title", "")])
    def test_bad_values_are_refused(self, store, field, value):
        item = store.add("Amma's birthday is on 22 December.", NOV21)["item"]
        with pytest.raises(ValueError) as bad:
            store.edit(item["id"], {"kind": "important_date", "title": "Amma's birthday", field: value}, NOV21)
        assert bad.value.args[0] == field

    def test_databases_from_before_this_change_get_the_new_columns(self, tmp_path):
        first = MemoryStore(tmp_path, "u", SECRET, HashEmbedder())
        with sqlite3.connect(first.db_path) as db:  # make it look like an S2 database
            db.execute("ALTER TABLE items DROP COLUMN title")
            db.execute("ALTER TABLE items DROP COLUMN confirmed")
        again = MemoryStore(tmp_path, "u", SECRET, HashEmbedder())
        assert again.add("Amma's birthday is on 22 December.", NOV21)["item"]["confirmed"] == 0

    def test_api_card_confirm_and_edit(self, tmp_path):
        # a bill with no amount sends the hybrid reader to the model, which finds no amount either
        model = FakeModel([item_json(kind="bill", date="2026-12-12")])
        client = TestClient(create_app(tmp_path, SECRET, HashEmbedder(), model, today=lambda: NOV21))
        saved = client.post("/api/memories", json={"text": "Electricity bill due on 12 December."}).json()
        assert saved["understood"]["uncertain"] == ["no amount found"]
        item_id = saved["item"]["id"]
        bad = client.put(f"/api/memories/{item_id}", json={"kind": "bill", "title": "Electricity", "date": "12/12/2026"})
        assert bad.status_code == 422 and "date" in bad.json()["detail"]
        good = client.put(f"/api/memories/{item_id}", json={"kind": "bill", "title": "Electricity", "date": "2026-12-12", "amount": 1860})
        assert good.json()["confirmed"] == 1 and good.json()["understood"]["uncertain"] == []
        other = client.post("/api/memories", json={"text": "Amma's birthday is on 22 December."}).json()["item"]["id"]
        assert client.post(f"/api/memories/{other}/confirm").json()["confirmed"] == 1
        assert client.post("/api/memories/999/confirm").status_code == 404


class TestGuard:

    SURGERY = {"id": 1, "kind": "appointment", "text": "Dad's eye surgery on 3 December at 9 am.", "date": "2026-12-03",
               "time": "09:00", "amount": None, "status": "active"}

    def test_the_slice_demo_failure_is_caught(self):
        wrong = "Dad's eye surgery is scheduled for Saturday 21 November 2026 at 9:00 am."
        assert check_answer(wrong, [self.SURGERY], {}, NOV21, "When is Dad's surgery?") == ["today's date given for something not on today"]

    def test_the_right_answer_passes(self):
        right = "Dad's eye surgery is on Thursday 3 December 2026 at 9:00 am."
        assert check_answer(right, [self.SURGERY], {}, NOV21, "When is Dad's surgery?") == []

    def test_today_is_fine_when_the_answer_talks_about_today(self):
        text = "Today is Saturday 21 November. Dad's surgery is on 3 December, in 12 days."
        assert check_answer(text, [self.SURGERY], {}, NOV21, "When is Dad's surgery?") == []

    def test_invented_amounts_fail_and_correct_sums_pass(self):
        bills = [{"id": 1, "kind": "bill", "text": "a", "amount": 1860, "date": "2026-12-12", "status": "active"},
                 {"id": 2, "kind": "bill", "text": "b", "amount": 420, "date": "2026-12-10", "status": "active"}]
        assert check_answer("₹1,860 and ₹420, so ₹2,280 in all.", bills, {}, NOV21, "q") == []
        assert check_answer("₹1,860 and ₹420, so ₹2,380 in all.", bills, {}, NOV21, "q") != []
        assert check_answer("Your gas bill is ₹950.", bills, {}, NOV21, "q") != []

    def test_old_dates_of_moved_items_are_allowed(self):
        item = dict(self.SURGERY, date="2026-12-05")
        history = {1: [{"field": "date", "old_value": "2026-12-03", "new_value": "2026-12-05"}]}
        assert check_answer("It moved from 3 December to 5 December.", [item], history, NOV21, "q") == []


class TestAnswersWithGuard:

    @pytest.fixture
    def demo(self, store):
        add(store, "Dad's eye surgery at Sankara Nethralaya on 3 December at 9 am. Carry his Aadhaar and old prescriptions.")
        add(store, "Pay the cook ₹4,500 on the 1st of every month.")
        add(store, "Anjali's birthday is on 2 December. She wants a sketchbook.")
        return store

    WRONG = "The surgery is on Saturday 21 November 2026 at 9:00 am."
    RIGHT = "The surgery is on Thursday 3 December 2026 at 9:00 am. Carry his Aadhaar and old prescriptions."

    def test_without_the_guard_the_wrong_answer_goes_out(self, demo):
        answer = Assistant(demo, FakeModel([self.WRONG]), guard=False, trimmed=False, routed=False).ask("When is Dad's surgery?", NOV21)
        assert answer.text == self.WRONG and answer.guard == "off"

    def test_a_wrong_answer_is_retried_with_fewer_facts(self, demo):
        model = FakeModel([self.WRONG, self.RIGHT])
        answer = Assistant(demo, model).ask("When is Dad's surgery?", NOV21)
        assert answer.guard == "retry" and answer.text == self.RIGHT and answer.first_answer == self.WRONG
        assert len(model.calls) == 2 and model.calls[1]["messages"][1]["content"].count("[") <= 3

    def test_two_wrong_answers_give_a_code_built_answer(self, demo):
        answer = Assistant(demo, FakeModel([self.WRONG, self.WRONG])).ask("When is Dad's surgery?", NOV21)
        assert answer.guard == "fallback" and "Sankara Nethralaya on 3 December" in answer.text
        assert "21 November" not in answer.text

    def test_guard_events_are_counts_only(self, demo, caplog):
        with caplog.at_level(logging.INFO, logger="chirii"):
            Assistant(demo, FakeModel([self.WRONG, self.WRONG])).ask("When is Dad's surgery?", NOV21)
        assert "answer_guard" in caplog.text and "status=fallback" in caplog.text and "surgery" not in caplog.text.lower()

    def test_trimmed_retrieval_leaves_out_unrelated_memories(self, demo):
        model = FakeModel([self.RIGHT])
        Assistant(demo, model).ask("When is Dad's eye surgery?", NOV21)
        prompt = model.calls[0]["messages"][1]["content"]
        assert "surgery" in prompt and "cook" not in prompt
        full = FakeModel([self.RIGHT])
        Assistant(demo, full, guard=False, trimmed=False, routed=False).ask("When is Dad's eye surgery?", NOV21)
        assert "cook" in full.calls[0]["messages"][1]["content"]

    def test_unconfirmed_memories_are_named_in_the_answer(self, store):
        add(store, "Dad's eye surgery at Sankara Nethralaya on 3 December at 9 am.", confirm=False)
        answer = Assistant(store, FakeModel([self.RIGHT])).ask("When is Dad's surgery?", NOV21)
        assert "haven't confirmed yet" in answer.text and answer.unconfirmed
        store.confirm(store.items()[0]["id"])
        assert "haven't confirmed" not in Assistant(store, FakeModel([self.RIGHT])).ask("When is Dad's surgery?", NOV21).text

    @pytest.mark.parametrize("question,window", [
        ("What's due in the next 7 days?", ("2026-11-21", "2026-11-28")),
        ("Which payments are due this week?", ("2026-11-21", "2026-11-22")),
        ("Anything next week?", ("2026-11-23", "2026-11-29")),
        ("What do I have tomorrow?", ("2026-11-22", "2026-11-22")),
        ("Plan my day.", ("2026-11-21", "2026-11-21")),
        ("What do I need to finish before the end of this month?", ("2026-11-21", "2026-11-30")),
        ("What's on 3 December?", ("2026-12-03", "2026-12-03")),
        ("What does Anjali want for her birthday?", None),
    ])
    def test_question_windows(self, question, window):
        got = question_window(question, NOV21)
        assert (tuple(d.isoformat() for d in got) if got else None) == window


class TestGuardEmptyDays:

    def test_a_week_plan_listing_empty_days_passes_the_guard(self):
        items = [{"id": 1, "kind": "appointment", "text": "Dentist on 16 December at 5 pm.", "date": "2026-12-16", "time": "17:00", "amount": None, "status": "active"}]
        answer = "- Monday 14 December: No appointments.\n- Tuesday 15 December: Nothing.\n- Wednesday 16 December: Dentist at 5 pm."
        assert check_answer(answer, items, {}, date(2026, 12, 14), "Plan my week.") == []

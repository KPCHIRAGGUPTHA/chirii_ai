"""
Demo slice checks, with a fake model so no server is needed: when the hybrid reader calls Qwen, how its reply is
checked against the sentence, how answers are built from memory, and the web API end to end.
"""

import json
import logging
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from chirii.tests.helpers import logged_in
from chirii.app.api.main import create_app, local_only
from chirii.app.assistant import NOT_IN_MEMORY, Assistant, fact_line, say_amount
from chirii.app.llm.client import LLMError, LLMReply
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.hybrid import HybridExtractor, validate
from chirii.app.memory.store import MemoryStore

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
WED = date(2026, 12, 2)


class FakeModel:
    """Returns canned replies and records every call."""

    def __init__(self, replies=None, fail=False):
        self.replies = list(replies or [])
        self.fail = fail
        self.calls = []

    def chat(self, messages, max_tokens=256, json_schema=None, purpose="chat"):
        self.calls.append({"messages": messages, "json_schema": json_schema, "purpose": purpose})
        if self.fail:
            raise LLMError("down")
        text = self.replies.pop(0) if self.replies else "ok"
        return LLMReply(text=text if isinstance(text, str) else json.dumps(text), tokens_in=1, tokens_out=1, duration_ms=1)


def item_json(**fields):
    base = {"kind": "note", "date": None, "end_date": None, "time": None, "amount": None, "day_of_month": None, "weekdays": []}
    return {**base, **fields}


class TestHybrid:

    def test_rules_handle_clear_sentences_without_the_model(self):
        model = FakeModel()
        item = HybridExtractor(model)("Dentist appointment with Dr. Kavitha on 10 December at 6 pm.", WED)
        assert model.calls == [] and item.source == "rules" and item.date == "2026-12-10"

    UNREAD = "pick Riya from tuition at 7"  # no date, no am/pm, no known word: the rules fall back to a plain note

    def test_model_fills_what_the_rules_missed(self):
        model = FakeModel([item_json(kind="appointment", time="19:00")])
        item = HybridExtractor(model)(self.UNREAD, WED)
        assert len(model.calls) == 1 and model.calls[0]["json_schema"] is not None
        # Qwen chose the kind, which code can't check, so the user is asked to confirm it
        assert (item.kind, item.time, item.source, item.needs_review) == ("appointment", "19:00", "model", True)

    def test_schema_only_offers_what_the_sentence_says(self):
        model = FakeModel([item_json(kind="appointment", time="19:00")])
        HybridExtractor(model)(self.UNREAD, WED)
        fields = model.calls[0]["json_schema"]["properties"]
        assert fields["time"]["anyOf"][0]["enum"] == ["07:00", "19:00"]
        assert fields["amount"] == {"type": "null"} and fields["date"] == {"type": "null"}
        assert fields["weekdays"] == {"type": "array", "maxItems": 0}

    def test_an_invented_amount_is_rejected_and_the_item_stays_a_note(self):
        model = FakeModel([item_json(kind="bill", amount=950)])
        item = HybridExtractor(model)(self.UNREAD, WED)
        assert item.kind == "note" and item.amount is None and item.needs_review and "rejected" in item.source

    def test_model_down_keeps_the_rules_item_for_review(self):
        item = HybridExtractor(FakeModel(fail=True))(self.UNREAD, WED)
        assert item.needs_review and "unavailable" in item.source

    def test_rules_values_win_over_the_model(self):
        # The rules found ₹3,600 and 10 Dec; a different model date for the same field is ignored, not merged.
        model = FakeModel([item_json(kind="bill", amount=3600, date="2026-12-10")])
        item = HybridExtractor(model)("Need to pay school bus fees ₹3,600 before 10 Dec", WED)
        assert model.calls == [] and item.amount == 3600

    def test_fallback_is_logged_as_counts_only(self, caplog):
        with caplog.at_level(logging.INFO, logger="chirii"):
            HybridExtractor(FakeModel([item_json(kind="appointment", time="19:00")]))(self.UNREAD, WED)
        assert "extract_fallback" in caplog.text and "status=accepted" in caplog.text and "riya" not in caplog.text.lower()

    @pytest.mark.parametrize("text,reply,problems", [
        ("dentist tmrw 5pm", item_json(kind="appointment", date="2026-12-03", time="17:00"), []),
        ("dentist tmrw 5pm", item_json(kind="appointment", date="2026-12-04", time="17:00"), ["date"]),
        ("dentist tmrw 5pm", item_json(kind="appointment", date="2026-12-03", time="18:00"), ["time"]),
        ("Kal subah 10 baje bank jaana hai", item_json(kind="task", date="2026-12-03", time="10:00"), []),
        ("Mummy ka eye checkup 18 tareekh ko 11 baje", item_json(kind="appointment", date="2026-12-18", time="11:00"), []),
        ("Sunday ko party hai", item_json(kind="important_date", date="2026-12-06"), []),
        ("Credit card due 22nd, around 8k", item_json(kind="bill", date="2026-12-22", amount=8000), []),
        ("Credit card due 22nd, around 8k", item_json(kind="bill", date="2026-12-22", amount=9000), ["amount"]),
        ("Car EMI 12,400 debit on 7th every month", item_json(kind="bill", amount=12400, day_of_month=7), []),
        ("Car EMI 12,400 debit on 7th every month", item_json(kind="bill", amount=12400, day_of_month=8), ["day_of_month"]),
        ("Swim on Monday", item_json(kind="routine", weekdays=["Friday"]), ["weekdays"]),
        ("anything", item_json(kind="spaceship"), ["kind"]),
    ])
    def test_validation(self, text, reply, problems):
        assert validate(reply, text, WED) == problems


class TestAnswers:

    @pytest.fixture
    def store(self, tmp_path):
        s = MemoryStore(tmp_path, "u", SECRET, HashEmbedder())
        s.add("My next check-up is with Dr. Ramaswamy at the hospital on 18 December at 11 am.", date(2026, 12, 1))
        s.add("The doctor's appointment is moved to 21 December, same time.", date(2026, 12, 10))
        s.add("Divya's school term fee of ₹1,95,000 must be paid by 18 December.", date(2026, 12, 1))
        return s

    def test_fact_lines_give_current_details_and_changes(self, store):
        visit = next(i for i in store.items() if "Ramaswamy" in i["text"])
        line = fact_line(visit, store.history(visit["id"]))
        assert "Monday 21 December 2026" in line and "11:00 am" in line and "changed from 2026-12-18 to 2026-12-21" in line

    def test_amounts_use_indian_grouping(self):
        assert say_amount(195000) == "₹1,95,000" and say_amount(1860) == "₹1,860" and say_amount(420) == "₹420"

    def test_prompt_holds_today_the_facts_and_the_abstain_rule(self, store):
        model = FakeModel(["Your check-up is on 21 December at 11 am."])
        answer = Assistant(store, model).ask("When is my check-up?", date(2026, 12, 12))
        system, user = model.calls[0]["messages"][0]["content"], model.calls[0]["messages"][1]["content"]
        assert "Saturday 12 December 2026" in system and NOT_IN_MEMORY in system
        assert "Ramaswamy" in user and user.rstrip().endswith("Question: When is my check-up?")
        assert answer.text.startswith("Your check-up") and answer.fact_ids

    def test_applied_updates_are_not_shown_as_separate_facts(self, store):
        answer = Assistant(store, FakeModel()).ask("doctor", date(2026, 12, 12))
        assert not any("moved to 21 December, same time" in f and "[Unmatched update]" in f for f in answer.facts)


class TestApi:

    @pytest.fixture
    def client(self, tmp_path):
        model = FakeModel(["On Thursday 3 December at 5:00 pm."])  # "dentist tmrw 5pm" is read by the rules, no model call
        app = create_app(tmp_path, SECRET, HashEmbedder(), model, today=lambda: WED)
        return logged_in(app), tmp_path

    def test_add_list_ask_delete_all(self, client):
        c, root = client
        assert c.get("/").status_code == 200
        added = c.post("/api/memories", json={"text": "dentist tmrw 5pm"}).json()
        assert added["item"]["date"] == "2026-12-03"
        assert len(c.get("/api/memories").json()["items"]) == 1
        reply = c.post("/api/ask", json={"question": "When is the dentist?"}).json()
        assert reply["answer"].startswith("On Thursday") and reply["facts"]
        assert c.post("/api/delete-all").json()["deleted"] == {"items": 1, "vectors": 1}
        assert not any(p.suffix in (".db", ".faiss") for p in root.rglob("*"))  # nothing left on disk
        assert c.get("/api/memories").json()["items"] == []  # starts again empty

    def test_empty_text_is_refused(self, client):
        assert client[0].post("/api/memories", json={"text": ""}).status_code == 422

    def test_model_down_gives_a_clear_error(self, tmp_path):
        app = create_app(tmp_path, SECRET, HashEmbedder(), FakeModel(fail=True), today=lambda: WED)
        r = logged_in(app).post("/api/ask", json={"question": "anything"})
        assert r.status_code == 503 and "llama-server" in r.json()["detail"]

    def test_requests_from_other_computers_are_refused(self):
        with pytest.raises(HTTPException) as refused:
            local_only(SimpleNamespace(client=SimpleNamespace(host="192.168.1.20")))
        assert refused.value.status_code == 403
        assert local_only(SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))) is None

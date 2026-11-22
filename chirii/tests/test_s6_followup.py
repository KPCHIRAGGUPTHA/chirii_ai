"""S6 follow-up questions and in-session history (own examples, not from any eval set)."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from chirii.app.api.main import create_app
from chirii.app.followup import MAX_TURNS, Turn, by_model, looks_like_followup, resolve
from chirii.app.llm.client import LLMError
from chirii.app.memory import store as store_module
from chirii.app.memory.embeddings import HashEmbedder
from chirii.tests.helpers import TEST_PASSWORD, logged_in
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
MON = date(2027, 6, 7)


@pytest.fixture(autouse=True)
def fresh():
    store_module._open.clear()
    yield
    store_module._open.clear()


def after(previous: str, question: str, answer: str = "x"):
    return resolve(question, [Turn(previous, previous, answer)])


class TestRules:

    @pytest.mark.parametrize("previous,question,expected", [
        ("When is my gas bill due?", "and the water bill?", "When is the water bill due?"),
        ("When is my gas bill due?", "is it paid?", "Is my gas bill paid?"),
        ("When is my gas bill due?", "How much is it?", "How much is my gas bill?"),
        ("When is my gas bill due?", "And when is it due?", "When is my gas bill due?"),
        ("When is Meera's audition?", "What about Tanvi?", "When is Tanvi's audition?"),
        ("When is Meera's audition?", "Tanvi ka kya?", "When is Tanvi's audition?"),
        ("When is Meera's audition?", "What time does she arrive?", "What time does Meera arrive?"),
        ("When is Meera's audition?", "Is her audition in the evening?", "Is Meera's audition in the evening?"),
        ("When is Dadi's knee scan?", "Uska time kya hai?", "Dadi's knee scan ka time kya hai?"),
        ("What's my plan for today?", "and tomorrow?", "What's my plan for tomorrow?"),
        ("What's my plan for Saturday?", "what about Sunday?", "What's my plan for Sunday?"),
        ("Which bills are due this week?", "and next week?", "Which bills are due next week?"),
        ("Which bills are due in March?", "and April?", "Which bills are due in April?"),
        ("When is the plumber visit?", "Can you move that?", "Can you move the plumber visit?"),
    ])
    def test_rewrites(self, previous, question, expected):
        assert after(previous, question) == (expected, "rule")

    @pytest.mark.parametrize("question", ["Is it going to rain tomorrow?", "What time is it?", "What is the capital of Peru?",
                                          "When is my dentist appointment?", "How much is 18% of 2,500?"])
    def test_questions_that_stand_alone_are_left_alone(self, question):
        assert after("When is my gas bill due?", question) == (None, "")

    def test_two_people_is_not_guessed(self):
        assert after("When is Meera's audition?", "Did she call him?") == (None, "")

    def test_no_history_no_rewrite(self):
        assert resolve("is it paid?", []) == (None, "")

    def test_what_looks_like_a_follow_up(self):
        assert looks_like_followup("and Kabir?") and looks_like_followup("Did she call him?")
        assert not looks_like_followup("When is my dentist appointment?")


class TestModelRewrite:

    def test_used_when_rules_cannot(self):
        model = FakeModel(["When is Kabir's cricket kit payment due?"])
        prev = Turn("When do I pay Kabir?", "When do I pay Kabir?", "You pay Kabir ₹700 for the cricket kit by 25 November.")
        assert resolve("and the kit?", [prev], model) == ("When is Kabir's cricket kit payment due?", "model")
        assert model.calls[0]["purpose"] == "followup"

    def test_a_rewrite_that_adds_a_name_or_number_is_refused(self):
        prev = Turn("When is my gas bill due?", "When is my gas bill due?", "Your gas bill of ₹900 is due on 10 June.")
        assert by_model("and Kabir?", [prev], FakeModel(["When is Rohan's gas bill due?"])) is None
        assert by_model("and Kabir?", [prev], FakeModel(["Is ₹1,200 due to Kabir?"])) is None

    def test_model_down_means_no_rewrite(self):
        prev = Turn("When is my gas bill due?", "When is my gas bill due?", "x")
        assert resolve("and Kabir?", [prev], FakeModel(fail=True)) == (None, "")


def app_for(tmp_path):
    return create_app(tmp_path, SECRET, HashEmbedder(), FakeModel(), today=lambda: MON)


def ask(client, question):
    r = client.post("/api/ask", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


class TestConversation:

    def test_a_follow_up_is_answered_as_understood(self, tmp_path):
        c = logged_in(app_for(tmp_path))
        c.post("/api/memories", json={"text": "Gas bill ₹900 due on 10 June."})
        ask(c, "When is my gas bill due?")
        r = ask(c, "is it paid?")
        assert r["understood_as"] == "Is my gas bill paid?"
        assert ask(c, "When is my gas bill due?")["understood_as"] is None   # a full question is not rewritten

    def test_new_chat_forgets(self, tmp_path):
        c = logged_in(app_for(tmp_path))
        ask(c, "When is my gas bill due?")
        assert c.delete("/api/conversation").json() == {"cleared": True}
        assert ask(c, "is it paid?")["understood_as"] is None

    def test_logout_forgets(self, tmp_path):
        app = app_for(tmp_path)
        c = logged_in(app)
        ask(c, "When is my gas bill due?")
        c.post("/api/logout")
        c = logged_in(app)
        assert ask(c, "is it paid?")["understood_as"] is None
        assert app.state.conversations and all(len(turns) == 1 for _, turns in app.state.conversations.values())

    def test_sessions_and_users_do_not_share_a_conversation(self, tmp_path):
        app = app_for(tmp_path)
        a = logged_in(app, "asha")
        ask(a, "When is my gas bill due?")
        assert ask(logged_in(app, "bala"), "is it paid?")["understood_as"] is None
        assert ask(logged_in(app, "asha", TEST_PASSWORD), "is it paid?")["understood_as"] is None   # asha, another session
        assert ask(a, "is it paid?")["understood_as"] == "Is my gas bill paid?"

    def test_delete_all_and_delete_account_forget(self, tmp_path):
        app = app_for(tmp_path)
        c = logged_in(app)
        ask(c, "When is my gas bill due?")
        c.post("/api/delete-all")
        assert ask(c, "is it paid?")["understood_as"] is None
        ask(c, "When is my gas bill due?")
        c.post("/api/delete-account", json={"password": TEST_PASSWORD})
        assert not app.state.conversations

    def test_the_conversation_is_never_written_to_disk(self, tmp_path):
        c = logged_in(app_for(tmp_path))
        c.post("/api/memories", json={"text": "Gas bill ₹900 due on 10 June."})
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        ask(c, "When is my gas bill due?")
        ask(c, "is it paid?")
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before

    def test_a_crisis_message_is_not_rewritten(self, tmp_path):
        c = logged_in(app_for(tmp_path))
        ask(c, "When is Meera's audition?")
        r = ask(c, "I don't want to live anymore, she doesn't know.")
        assert r["understood_as"] is None and r["label"] == "Support"

    def test_history_is_capped(self, tmp_path):
        app = app_for(tmp_path)
        c = logged_in(app)
        for n in range(MAX_TURNS + 3):
            ask(c, f"What is {n} plus 2?")
        assert [len(turns) for _, turns in app.state.conversations.values()] == [MAX_TURNS]

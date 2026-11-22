"""
S4e checks with new examples (not development-set wording): profile and person facts, Indian relation terms, reverse
relationships without assuming gender, the person's own name in the third person, family aliases for other
memories, the card's people facts, and answers that never say "the user" or contradict themselves.
"""

import re
from datetime import date

import pytest
from fastapi.testclient import TestClient

from chirii.app import people
from chirii.app.api.main import create_app
from chirii.app.assistant import Assistant
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.extract import extract
from chirii.app.memory.store import MemoryStore
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
TODAY = date(2027, 5, 1)
SENTENCES = ["I'm Nikhil Rao; my friends call me Nikki.", "Ammi's name is Farida.", "Zoya is my elder sister.",
             "My daughter Inaya turns 4 on 9 June.", "Ammi's knee scan is on 3 May at 9 am.", "Nani lives in Lucknow.",
             "My husband's name is Imran.", "Zoya's son Rehan starts school in June.", "My cousin Sam works in Pune."]


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path, "p", SECRET, HashEmbedder())
    for text in SENTENCES:
        s.confirm(s.add(text, date(2027, 4, 20))["item"]["id"])
    return s


def ask(store, question):
    return Assistant(store, FakeModel(["(model)"])).ask(question, TODAY)


class TestKinds:

    @pytest.mark.parametrize("text,kind", [
        ("I'm Nikhil Rao; my friends call me Nikki.", "profile"), ("Ammi's name is Farida.", "person"),
        ("Zoya is my elder sister.", "person"), ("Nani lives in Lucknow.", "person"),
        ("Zoya's son Rehan starts school in June.", "person"),
        ("My brother Kabir is visiting on 8 May.", "important_date"),          # an event keeps its kind
        ("My roommate Dev owes me ₹500.", "receivable"),
    ])
    def test_profile_and_person_sentences(self, text, kind):
        assert extract(text, date(2027, 4, 20)).kind == kind


class TestAnswers:

    @pytest.mark.parametrize("question,expected", [
        ("What is my name?", "Your name is Nikhil Rao."),
        ("What do my friends call me?", "Friends call you Nikki."),
        ("What is my mother's name?", "Your mother's name is Farida."),
        ("What is Ammi's name?", "Your mother's name is Farida."),
        ("Who is Zoya?", "Zoya is your sister."),
        ("Where does Nani live?", "Nani lives in Lucknow."),
        ("What is my father's name?", "You haven't told me your father's name."),
    ])
    def test_direct_questions(self, store, question, expected):
        answer = ask(store, question)
        assert answer.text == expected and answer.route == "people" and answer.label == "From your memories"

    def test_reverse_relationships(self, store):
        assert ask(store, "Who are Farida's children?").text.startswith("Farida's children: you (Nikhil) and Zoya.")
        assert ask(store, "What is Imran's wife's name?").text.startswith("That's you: Nikhil.")
        assert ask(store, "Who is Inaya's grandmother?").text.startswith("Inaya's grandmother: Farida.")
        assert ask(store, "Who is Rehan's aunt?").text.startswith("That's you: Nikhil.")       # Zoya, his mother, is not his aunt

    def test_relationship_through_someone_else(self, store):
        assert ask(store, "Who is Rehan?").text.startswith("Rehan is your nephew (Zoya's son).")

    def test_gender_is_never_assumed(self, store):
        answer = ask(store, "Is Sam my brother or my sister?").text
        assert "cousin" in answer and not re.search(r"\b(he|she|his|her)\b", answer, re.I)
        assert "brother" not in answer.replace("my brother", "")

    def test_own_name_in_the_third_person(self, store):
        assert people.own_name_to_my("When is Nikhil's knee scan?", store.items()) == "When is my knee scan?"

    def test_family_alias_finds_the_memory(self, store):
        assert people.alias_terms("When is Farida's knee scan?", store.items()) == "When is Ammi's knee scan?"
        answer = Assistant(store, FakeModel(["Ammi's knee scan is on Monday 3 May 2027 at 9:00 am."])).ask("When is Farida's knee scan?", TODAY)
        assert "3 May" in answer.text and answer.route == "lookup"

    def test_people_answers_never_say_the_user_or_contradict(self, store):
        for q in ["What is my name?", "Who is Zoya?", "Who are Farida's children?", "What is my father's name?", "Who is Rehan's aunt?"]:
            text = ask(store, q).text
            assert "the user" not in text.lower()
            assert not ("haven't told me" in text and "name is" in text), text


class TestCard:

    def test_card_shows_people_facts_and_links(self, tmp_path):
        client = TestClient(create_app(tmp_path, SECRET, HashEmbedder(), FakeModel(), today=lambda: TODAY))
        card = client.post("/api/memories", json={"text": "Ammi's name is Farida."}).json()["understood"]
        assert "Farida: your mother" in card["summary"]
        linked = client.post("/api/memories", json={"text": "Ammi's dentist visit is on 12 May at 5 pm."}).json()["understood"]
        assert any("Farida" in p and "your mother" in p for p in linked["people"])

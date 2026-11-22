"""
S4f checks with a fake model and examples written for these tests (not development set 5): matching by stems, the
Indian synonym map and embedding similarity; payable bill words; "by when" deadlines; ranges listed once; maths
routing; and the maths check (Qwen's choice against an independent code computation).
"""

from datetime import date

import numpy as np
import pytest

from chirii.app import meaning
from chirii.app.assistant import Assistant
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.extract import extract
from chirii.app.memory.hybrid import allowed_kinds
from chirii.app.memory.store import MemoryStore
from chirii.app.periods import find_event, read_period
from chirii.app.planner import plan_draft, window_draft
from chirii.app.router import mentions_all, route_by_rules, specific_terms, term_matches, validate_choice
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
MON = date(2027, 5, 3)     # Monday 3 May 2027
SAID = date(2027, 4, 20)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path, "f", SECRET, HashEmbedder())
    for text in ["Ajay is getting married on 15 May in Indore.", "Kiraya of ₹18,000 goes to the landlord on the 6th of every month.",
                 "Hotstar renews at ₹899 on 7 May.", "Cousin Tanvi's exams from 4 to 8 May.", "Revathi is getting engaged on 20 May.",
                 "Mr. Iyer owes me ₹1,500 for the gym renewal.", "ITR filing deadline is 31 July."]:
        s.confirm(s.add(text, SAID)["item"]["id"])
    return s


class TestWordFamilies:

    @pytest.mark.parametrize("a,b", [("engaged", "engagement"), ("married", "wedding"), ("marriage", "wedding"), ("shaadi", "wedding"),
                                     ("renewal", "renews"), ("kiraya", "rent"), ("dawai", "medicine"), ("tablets", "medicine"),
                                     ("classes", "class")])
    def test_same_family(self, a, b):
        assert meaning.same_family(a, b)

    @pytest.mark.parametrize("a,b", [("wedding", "birthday"), ("rent", "renewal"), ("medicine", "meeting"), ("engaged", "exam")])
    def test_different_family(self, a, b):
        assert not meaning.same_family(a, b)

    def test_terms_match_by_family(self):
        assert mentions_all(["!Ajay", "wedding"], "Ajay is getting married on 15 May in Indore.")
        assert term_matches("shaadi", "Ajay is getting married on 15 May.")
        assert "kiraya" in specific_terms("Kiraya kab dena hai?")    # a Hinglish word for a kind of thing is a key term

    def test_meaning_by_embedding_above_the_threshold_only(self):
        class Fixed:   # "glasses" and "spectacles" close, nothing else
            dim = 3

            def embed(self, texts):
                rows = [np.array([1.0, 0, 0]) if t in ("glasses", "spectacles") else np.array([0, 1.0, 0]) if t == "funeral" else
                        np.array([0.6, 0.8, 0]) for t in texts]
                return np.array(rows, dtype=np.float32)
        m = meaning.Meaning(Fixed())
        token = meaning.current.set(m)
        try:
            assert term_matches("glasses", "Collect the spectacles from the optician.")
            assert not term_matches("glasses", "Uncle's funeral prayers on Sunday.")       # 0.6: below the threshold
            assert not term_matches("!Glasses", "Collect the spectacles from the optician.")   # names never match by meaning
        finally:
            meaning.current.reset(token)

    def test_no_embedder_means_no_meaning_match(self):
        assert meaning.for_embedder(HashEmbedder()) is None
        assert not term_matches("glasses", "Collect the spectacles from the optician.")

    def test_event_found_by_family(self, store):
        event = find_event("the engagement", MON, store.items(include_updates=False))
        assert event is not None and "engaged" in event["text"]
        period = read_period("Anything to pay before the engagement?", MON, store.items(include_updates=False))
        assert period is not None and period.end == date(2027, 5, 19)

    def test_answer_from_the_meaning_match_not_a_partial_match(self, store):
        answer = Assistant(store, FakeModel(["Ajay's wedding is on Saturday 15 May in Indore."])).ask("When is Ajay's wedding?", MON)
        assert "nothing about" not in answer.text and "15 May" in answer.text


class TestBillWords:

    @pytest.mark.parametrize("text", ["Hotstar renews at ₹899 on 7 May.", "Amazon Prime renewal of ₹1,499 on 9 June.",
                                      "Health insurance premium ₹12,000 due 1 June.", "Phone recharge of ₹239 on 5 May."])
    def test_payment_words_are_bills(self, text):
        assert extract(text, SAID).kind == "bill"
        assert "receivable" not in allowed_kinds(text) and "income" not in allowed_kinds(text)

    def test_money_owed_to_the_person_stays_receivable(self):
        assert extract("Mr. Iyer owes me ₹1,500 for the gym renewal.", SAID).kind == "receivable"
        assert "receivable" in allowed_kinds("Mr. Iyer owes me ₹1,500 for the gym renewal.")

    def test_renewal_listed_to_pay_not_to_receive(self, store):
        draft = window_draft(store, MON, date(2027, 5, 9), ["pay"], "Which subscriptions do I pay this week?")
        assert "₹899" in draft.text and "To receive" not in draft.text.split("₹899")[0]


class TestDeadlines:

    def test_by_when_is_a_lookup(self, store):
        r = route_by_rules("By when do I have to file the ITR?", MON, store.items(include_updates=False))
        assert r is not None and r.kind == "lookup" and r.period is None

    def test_a_window_choice_for_a_deadline_is_rejected(self, store):
        choice = {"route": "window", "period": "today", "item": None}
        assert validate_choice(choice, "By when do I have to file the ITR?", MON, True, store.items(include_updates=False)) is None


class TestRangesOnce:

    def test_window_lists_a_range_once(self, store):
        draft = window_draft(store, MON, date(2027, 5, 9), ["all"], "What's coming up this week?")
        assert draft.text.count("exams") == 1 and "Tuesday 4 May to Saturday 8 May" in draft.text
        later = window_draft(store, date(2027, 5, 6), date(2027, 5, 9), ["all"], "What's on from Thursday?")
        assert later.text.count("exams") == 1 and "until Saturday 8 May" in later.text

    def test_no_dangling_word_in_titles(self):
        from chirii.app.formatting import make_title
        assert make_title("Kiraya of ₹18,000 goes to the landlord on the 6th of every month.") == "Kiraya"
        assert make_title("Note.") == "Note."                    # S5: this crashed (nothing before the full stop)

    def test_plan_lists_a_range_once(self, store):
        draft = plan_draft(store, MON, date(2027, 5, 9), MON)
        assert draft.text.count("exams") == 1


class TestMathsRouting:

    @pytest.mark.parametrize("question", ["Write ₹30,00,000 in lakh.", "What is 2.5 crore in lakh?", "How many weeks from 1 June to 29 June?",
                                          "Convert 12 km to miles.", "Split ₹600 among me and 2 friends.", "What is 40°C in Fahrenheit?"])
    def test_goes_to_maths(self, question):
        r = route_by_rules(question, MON, [])
        assert r is not None and r.kind == "maths"


class TestMathsCheck:

    @staticmethod
    def formula(name, numbers, from_unit=None, to_unit=None, expression=None):
        return {"formula": name, "numbers": numbers, "from_unit": from_unit, "to_unit": to_unit, "expression": expression}

    def ask(self, store, choice, question):
        return Assistant(store, FakeModel([choice])).ask(question, MON)

    def test_compounding_said_in_words_is_accepted(self, tmp_path):
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        a = self.ask(s, self.formula("compound_interest", [20000, 10, 2, 1]), "Compound interest on ₹20,000 at 10% for 2 years, compounded yearly?")
        assert a.text.startswith("₹4,200") and a.guard == "code"

    def test_conversion_factors_added_by_qwen_are_ignored(self, tmp_path):
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        a = self.ask(s, self.formula("convert", [12, 1.609344], "km", "mile"), "Convert 12 km to miles.")
        assert a.text.startswith("7.46 mile") and a.guard == "code"

    def test_a_valid_expression_under_the_wrong_formula_is_used(self, tmp_path):
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        a = self.ask(s, self.formula("percent_of", [350, 125, 80], expression="350 + 125 + 80"), "What is 350 + 125 + 80?")
        assert a.text.startswith("555") and "350 + 125 + 80 = 555" in a.text

    def test_on_disagreement_the_code_result_is_shown(self, tmp_path):
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        a = self.ask(s, self.formula("percent_of", [18, 2000]), "A ₹2,000 jacket plus 18% GST: what do I pay in total?")
        assert a.text.startswith("₹2,360") and "₹360  " not in a.text

    def test_code_alone_when_qwen_fails(self, tmp_path):
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        a = self.ask(s, {"formula": "emi", "numbers": [], "from_unit": None, "to_unit": None, "expression": None},
                     "What is the EMI on ₹3 lakh at 10% for 3 years?")
        assert a.text.startswith("₹9,680.16") and a.guard == "code"

    def test_lakh_conversion(self, tmp_path):
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        a = self.ask(s, self.formula("convert", [3000000], "rupees", "lakh"), "Write ₹30,00,000 in lakh.")
        assert a.text.startswith("30 lakh")

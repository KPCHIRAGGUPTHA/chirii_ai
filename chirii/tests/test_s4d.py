"""
S4d checks with new examples (not development-set wording): ranges without "from", money questions choosing the item
with the amount, capitalised names making a question personal, "done" updates, case-sensitive names, the lead
contradiction check, the maths formula library with Indian formats, and the glossary.
"""

from datetime import date

import pytest

from chirii.app.assistant import Assistant
from chirii.app.chat import compute, indian, justified_constants, money
from chirii.app.glossary import glossary_answer, terms_in
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.extract import extract
from chirii.app.memory.store import MemoryStore
from chirii.app.router import route_by_rules
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
SAID, TODAY = date(2027, 4, 1), date(2027, 4, 12)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path, "d", SECRET, HashEmbedder())


class TestRanges:

    @pytest.mark.parametrize("text,start,end", [
        ("The conference is rescheduled to 18 to 20 April.", "2027-04-18", "2027-04-20"),
        ("Annual leave between 3 and 7 May.", "2027-05-03", "2027-05-07"),
        ("School holidays 24-30 April.", "2027-04-24", "2027-04-30"),
    ])
    def test_ranges_without_from(self, text, start, end):
        item = extract(text, SAID).to_dict()
        assert (item["date"], item["end_date"]) == (start, end)

    def test_a_moved_range_replaces_both_ends(self, store):
        trip = store.add("Family trip to Ooty from 20 to 22 April.", SAID)["item"]["id"]
        store.add("The Ooty trip is postponed to 1 to 4 May.", SAID)
        item = store.get(trip)
        assert (item["date"], item["end_date"]) == ("2027-05-01", "2027-05-04")


class TestStatus:

    def test_a_money_question_is_about_the_item_with_the_amount(self, store):
        for text in ["Riya's dance recital is on 20 April.", "The dance recital costume fee of ₹1,500 is due on 15 April."]:
            store.confirm(store.add(text, SAID)["item"]["id"])
        store.add("I paid the dance recital costume fee today.", date(2027, 4, 10))
        answer = Assistant(store, FakeModel()).ask("Do I still owe anything for the dance recital?", TODAY)
        assert answer.text.startswith("No, nothing is pending") and "costume fee" in answer.text.lower()

    def test_done_updates_are_recognised(self, store):
        task = store.add("Submit the society NOC form by 15 April.", SAID)["item"]["id"]
        result = store.add("Submitted the society NOC form today.", date(2027, 4, 11))
        assert result["item"]["kind"] == "update" and result["linked_to"] == task and store.get(task)["status"] == "done"
        answer = Assistant(store, FakeModel()).ask("Did I submit the NOC form?", TODAY)
        assert answer.text.startswith("Yes, that's done")

    def test_a_past_statement_without_today_is_not_an_update(self):
        assert extract("I submitted the passport form on 2 March.", SAID).kind != "update"

    def test_a_cleared_debt_is_paid(self, store):
        debt = store.add("Neighbour Joseph owes me ₹2,200 for the groceries.", SAID)["item"]["id"]
        store.add("Joseph cleared his dues today.", date(2027, 4, 10))
        assert store.get(debt)["status"] == "paid"


class TestPersonalAndNames:

    def test_a_capitalised_saved_name_makes_a_question_personal(self, store):
        store.confirm(store.add("TDS certificate from HDFC is due by 30 April.", SAID)["item"]["id"])
        assert route_by_rules("When is the HDFC certificate due?", TODAY, store.items()).kind == "lookup"

    def test_my_is_never_a_name(self, store):
        store.confirm(store.add("My cousin Leela is visiting on 14 April.", SAID)["item"]["id"])
        assert route_by_rules("Tell me about my weekend.", TODAY, store.items()).kind == "plan"
        assert route_by_rules("Tell me about Leela.", TODAY, store.items()).kind == "people"


class TestLeadContradiction:

    def test_a_lead_saying_nothing_above_a_list_is_dropped(self, store):
        store.confirm(store.add("Gas bill of ₹1,100 is due on 14 April.", SAID)["item"]["id"])
        model = FakeModel([{"lead": "Good news, you don't have to pay anything this week!", "closing": ""}])
        answer = Assistant(store, model).ask("What do I have to pay this week?", TODAY)
        assert answer.guard == "lead_dropped" and "lead says there is nothing, but the list has items" in answer.problems
        assert "₹1,100" in answer.text and "Good news" not in answer.text


class TestFormulas:

    @pytest.mark.parametrize("formula,nums,units,question,value", [
        ("percent_of", [5, 3200], (None, None), "What is 5% GST on ₹3,200?", 160),
        ("add_percent", [800, 12], (None, None), "Price of ₹800 plus 12% GST?", 896),
        ("discount", [1500, 15], (None, None), "₹1,500 with 15% off?", 1275),
        ("percent_change", [40, 50], (None, None), "From 40 to 50, what percent rise?", 25),
        ("simple_interest", [50000, 6, 2], (None, None), "Simple interest on ₹50,000 at 6% for 2 years?", 6000),
        ("compound_interest", [10000, 10, 2], (None, None), "Compound interest on ₹10,000 at 10% for 2 years?", 2100),
        ("emi", [100000, 12, 1], (None, None), "EMI on ₹1 lakh at 12% for 1 year?", 8884.88),
        ("split", [300, 450, 250, 4], (None, None), "Split ₹300, ₹450 and ₹250 among 4.", 250),
        ("convert", [10], ("km", "mile"), "10 km in miles?", 6.2137),
        ("convert", [3.5], ("crore", "lakh"), "3.5 crore in lakh?", 350),
        ("convert", [25], ("celsius", "fahrenheit"), "25°C in Fahrenheit?", 77),
    ])
    def test_formula_library(self, formula, nums, units, question, value):
        result, working = compute(formula, nums, units[0], units[1], question)
        assert abs(result - value) < 0.01 and "=" in working

    def test_indian_number_format(self):
        assert (indian(100000), indian(1234567.5), money(2599), indian(8.0467)) == ("1,00,000", "12,34,567.5", "₹2,599", "8.05")

    def test_constants_only_where_the_words_call_for_them(self):
        assert 100 in justified_constants("What is 18% of 900?", TODAY)
        assert 100 not in justified_constants("What do 35 litres cost at ₹104.50?", TODAY)
        assert 28 in justified_constants("Mess bill per day for February?", TODAY)
        assert 31 not in justified_constants("Mess bill per day for February?", TODAY)

    def test_shown_working_in_an_answer(self, store):
        model = FakeModel([{"formula": "sum", "numbers": [1240, 860, 499], "from_unit": None, "to_unit": None, "expression": None}])
        answer = Assistant(store, model).ask("Add ₹1,240, ₹860 and ₹499.", TODAY)
        assert "1,240 + 860 + 499 = ₹2,599" in answer.text and answer.label == "Calculated"

    def test_days_between_uses_the_real_calendar(self, store):
        answer = Assistant(store, FakeModel()).ask("How many days from 1 February to 1 March 2028?", TODAY)
        assert answer.text.startswith("29 days")        # 2028 is a leap year


class TestGlossary:

    def test_glossary_answers_and_needs_capitals_for_short_terms(self):
        assert "Unified Payments Interface" in glossary_answer("What does UPI stand for?")
        assert terms_in("what is upi") == []                       # 'upi' in lower case is not taken as the term
        assert "Permanent Account Number" in glossary_answer("What is the full form of PAN?")

    def test_glossary_is_used_for_general_questions(self, store):
        model = FakeModel(["IMPS means something else entirely."])
        answer = Assistant(store, model).ask("What is IMPS?", TODAY)
        assert answer.text.startswith("IMPS (Immediate Payment Service)") and model.calls == [] and answer.label == "General answer"

    def test_glossary_has_about_fifty_entries(self):
        from chirii.app.glossary import GLOSSARY
        assert 45 <= len(GLOSSARY) <= 60

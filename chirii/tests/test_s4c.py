"""
S4c checks with a fake model and examples written for these tests (not development set 3): the period reader, the one
router (memory, people, maths, general, writing, mixed), Qwen-assisted routing with code checks, the all-terms
specific-item check, date and time in lookups, and the general-chat paths with labels.
"""

from datetime import date

import pytest

from chirii.app.assistant import Assistant
from chirii.app.chat import NO_LIVE_ANSWER, expression_ok, allowed_numbers
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.periods import read_period
from chirii.app.router import route, route_by_rules, specific_terms
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
WED = date(2027, 3, 3)   # Wednesday 3 March 2027
SAID = date(2027, 2, 20)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path, "c", SECRET, HashEmbedder())
    for text in ["Cousin Meera's wedding in Jaipur on 12 March.", "Gym membership of ₹3,000 is due on 6 March.",
                 "Pottery class every Saturday at 11 am.", "Dentist Dr. Joshi on 5 March at 4 pm.",
                 "Wipro offer letter: the joining date is 1 May.", "Neighbour Farhan owes me ₹900 for the cake order.",
                 "Car insurance of ₹8,700 is due on 20 March.", "Note: the spare key is under the blue pot."]:
        s.confirm(s.add(text, SAID)["item"]["id"])
    return s


class TestPeriods:

    @pytest.mark.parametrize("question,start,end", [
        ("What's on this weekend?", "2027-03-06", "2027-03-07"),
        ("Anything before the weekend?", "2027-03-03", "2027-03-05"),
        ("Plans for next weekend?", "2027-03-13", "2027-03-14"),
        ("How does my Saturday look?", "2027-03-06", "2027-03-06"),
        ("Anything on Wednesday?", "2027-03-03", "2027-03-03"),            # today is a Wednesday
        ("What about next Wednesday?", "2027-03-10", "2027-03-10"),
        ("What's due before the end of the month?", "2027-03-03", "2027-03-31"),
        ("Any bills next month?", "2027-04-01", "2027-04-30"),
        ("What's coming in the next 2 weeks?", "2027-03-03", "2027-03-17"),
        ("Anything this fortnight?", "2027-03-03", "2027-03-17"),
        ("What's on from 8 to 10 March?", "2027-03-08", "2027-03-10"),
        ("Kal kya hai?", "2027-03-04", "2027-03-04"),
    ])
    def test_named_periods(self, question, start, end):
        p = read_period(question, WED)
        assert (p.start.isoformat(), p.end.isoformat()) == (start, end)

    def test_before_a_saved_event(self, store):
        p = read_period("Do I need to pay anything before the Jaipur wedding?", WED, store.items())
        assert (p.start.isoformat(), p.end.isoformat(), p.kind) == ("2027-03-03", "2027-03-11", "event")
        assert p.header() == "before Cousin Meera's wedding in Jaipur on Friday 12 March"

    def test_an_unknown_event_gives_no_period(self, store):
        assert read_period("Anything before the Diwali party?", WED, store.items()) is None

    def test_weekday_names_are_never_item_names(self):
        assert specific_terms("What does my Saturday look like?") == []
        assert specific_terms("Is the Wipro interview on Friday?") == ["!Wipro", "interview"]


class TestRouter:

    @pytest.mark.parametrize("question,expected", [
        ("How does my Saturday look?", "plan"), ("What's on this weekend?", "window"), ("Is the gym membership still pending?", "status"),
        ("How much do I need for bills this month?", "window"), ("What is 12% of 3,500?", "maths"),
        ("If Farhan pays half, how much is left?", "mixed"), ("Who is Farhan?", "people"),
        ("What is the capital of Assam?", "general"), ("Write a thank-you message for my neighbour.", "writing"),
        ("When is the dentist?", "lookup"), ("Do I need to pay anything before the Jaipur wedding?", "window"),
        ("What did I note about the spare key?", "meaning"), ("What's the petrol price today?", "general"),
    ])
    def test_rules(self, store, question, expected):
        assert route_by_rules(question, WED, store.items()).kind == expected

    def test_personal_questions_never_go_to_pure_general(self, store):
        model = FakeModel([{"route": "general", "period": "none", "item": None}])
        decided = route("Meera ka kya scene hai?", WED, model, store.items())
        assert decided.kind != "general" and decided.source == "default"      # Qwen's "general" is refused

    def test_qwen_routing_is_checked(self, store):
        good = FakeModel([{"route": "window", "period": "this weekend", "item": None}])
        decided = route("Weekend scene kya hai mera?", WED, good, store.items())
        assert decided.kind in ("window", "plan") and decided.window is not None
        bad = FakeModel([{"route": "window", "period": "none", "item": None}])      # a window needs a period
        assert route("Kuch scene hai mera?", WED, bad, store.items()).source == "default"


class TestSpecificAndTime:

    def test_all_key_terms_must_match_and_partial_says_what_is_saved(self, store):
        answer = Assistant(store, FakeModel(["The interview is on 1 May."])).ask("When is the Wipro interview?", WED)
        assert answer.text.startswith("I have Wipro offer letter") and "but nothing about interview" in answer.text

    def test_lookup_answer_gets_the_time_added(self, store):
        answer = Assistant(store, FakeModel(["Your dentist appointment is on Friday 5 March 2027."])).ask("When is the dentist?", WED)
        assert answer.text.endswith("It's at 4:00 pm.") and answer.label == "From your memory"

    def test_people_route(self, store):
        answer = Assistant(store, FakeModel()).ask("Who is Farhan?", WED)
        assert answer.route == "people" and "Neighbour Farhan owes me ₹900" in answer.text


class TestChat:

    def test_maths_uses_the_calculator_not_the_model_s_arithmetic(self, store):
        model = FakeModel([{"expression": "3500 * 12 / 100"}])
        answer = Assistant(store, model).ask("What is 12% of 3,500?", WED)
        assert answer.text.startswith("420") and answer.label == "Calculated" and answer.guard == "code"

    def test_an_expression_with_an_invented_number_is_refused(self, store):
        answer = Assistant(store, FakeModel([{"expression": "3500 * 15 / 100"}])).ask("What is 12% of 3,500?", WED)
        assert answer.text.startswith("I couldn't work that out reliably") and answer.guard == "fallback"
        assert not expression_ok("3500 * 15 / 100", allowed_numbers("What is 12% of 3,500?", []))

    def test_mixed_maths_may_use_saved_amounts(self, store):
        answer = Assistant(store, FakeModel([{"expression": "900 / 2"}])).ask("If Farhan pays half, how much is left?", WED)
        assert answer.text.startswith("₹450") and answer.label == "Calculated from your memory"

    def test_days_until_a_saved_event_by_code(self, store):
        model = FakeModel()
        answer = Assistant(store, model).ask("How many days until Meera's wedding?", WED)
        assert answer.text.startswith("9 days") and model.calls == []

    def test_live_information_is_refused_by_code(self, store):
        model = FakeModel()
        answer = Assistant(store, model).ask("What's the petrol price today?", WED)
        assert answer.text == NO_LIVE_ANSWER and model.calls == [] and answer.label == "General knowledge"

    def test_general_knowledge_goes_to_the_model(self, store):
        answer = Assistant(store, FakeModel(["Dispur is the capital of Assam."])).ask("What is the capital of Assam?", WED)
        assert answer.text.startswith("Dispur") and answer.label == "General knowledge"

    def test_writing_that_makes_up_an_amount_is_retried(self, store):
        model = FakeModel(["Dear Farhan, please return the ₹1,500 you owe me.", "Dear Farhan, please pay the ₹900 for the cake order."])
        answer = Assistant(store, model).ask("Write a polite reminder to Farhan about the money he owes me.", WED)
        assert answer.guard == "retry" and "₹900" in answer.text and answer.label in ("Writing help", "Advice using your memory")

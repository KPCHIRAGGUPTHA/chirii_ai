"""
S4 checks, with a fake model and memories written for these tests (not the S1 eval): the router, code-built date
windows with totals, the planner (order, clashes, tomorrow, medicines), status answers, the specific-item check, the
rewording checks and fallback, and the S3 code bugs.
"""

from datetime import date

import pytest

from chirii.app.assistant import Assistant, code_answer, fact_line
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.planner import clashes, plan_draft, slots_for, status_draft, window_draft
from chirii.app.router import route, route_by_rules
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
MON = date(2027, 2, 1)   # a Monday
SAID = date(2027, 1, 25)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path, "u", SECRET, HashEmbedder())
    for text in [
        "Gas bill of ₹1,120 is due on 3 February.",
        "Broadband bill ₹799 due on 5 February.",
        "Society maintenance of ₹2,400 on the 4th of every month.",
        "Ravi owes me ₹1,500 and will pay by 4 February.",
        "Renew the passport by 6 February.",
        "Nephew's birthday is on 5 February.",
        "Car insurance premium ₹11,300 due on 20 March.",
        "Physiotherapy every Monday and Wednesday at 6 pm; it ends at 7 pm.",
        "Cousin's mehendi on 1 February from 6:30 pm to 9 pm.",
        "Plumber visit on 1 February at 11 am.",
        "Submit the tax papers before 3 pm on 1 February.",
        "Chemistry exam on 2 February at 10 am.",
        "Thyroxine 50 mcg every morning before breakfast.",
        "Note: the spare key is with the neighbour.",
    ]:
        s.confirm(s.add(text, SAID)["item"]["id"])
    return s


def find(store, words):
    return next(i for i in store.items() if words in i["text"])


class TestRouter:

    @pytest.mark.parametrize("question,kind", [
        ("Plan my day.", "plan"), ("What's my plan for this week?", "plan"), ("Aaj ka plan bata do.", "plan"),
        ("What do I have to pay this week?", "window"), ("Anything coming up in the next 4 days?", "window"),
        ("Which bills do I still have to pay?", "window"), ("Is anything due tomorrow?", "window"),
        ("Is the gas bill paid?", "status"), ("Did I renew the passport?", "status"),
        ("When is the gas bill due?", "lookup"), ("How much is the broadband bill?", "lookup"),
        ("What did I note about the spare key?", "meaning"), ("Where is the spare key?", "meaning"),
    ])
    def test_rules(self, question, kind):
        assert route_by_rules(question, MON).kind == kind

    def test_model_only_when_no_rule_matches_and_only_from_the_list(self):
        model = FakeModel([{"kind": "meaning"}])
        decided = route("Spare key situation?", MON, model)
        assert decided.kind == "meaning" and decided.source == "model"
        assert model.calls[0]["json_schema"]["properties"]["kind"]["enum"] == ["plan", "window", "status", "lookup", "meaning"]
        model = FakeModel()
        assert route("When is the gas bill due?", MON, model).source == "rules" and model.calls == []


class TestWindows:

    def test_payments_this_week_exclude_money_owed_to_me_and_total_correctly(self, store):
        draft = window_draft(store, MON, date(2027, 2, 7), ["pay"], "What do I have to pay this week?")
        assert "₹1,120" in draft.text and "₹799" in draft.text and "₹2,400" in draft.text
        assert "Ravi" not in draft.text and "₹11,300" not in draft.text
        assert "Total to pay: ₹4,319." in draft.text

    def test_money_owed_to_me(self, store):
        draft = window_draft(store, MON, date(2027, 2, 7), ["owed"], "Who owes me money this week?")
        assert "Ravi" in draft.text and "₹1,120" not in draft.text

    def test_paid_items_drop_out(self, store):
        store.add("I paid the gas bill.", date(2027, 1, 30))
        draft = window_draft(store, MON, date(2027, 2, 7), ["pay"], "What do I have to pay this week?")
        assert "₹1,120" not in draft.text and "Total to pay: ₹3,199." in draft.text

    def test_named_kind_of_payment_filters(self, store):
        draft = window_draft(store, MON, date(2027, 3, 31), ["pay"], "Which premiums do I still have to pay?")
        assert "₹11,300" in draft.text and "₹799" not in draft.text

    def test_daily_medicines_do_not_crowd_a_window(self, store):
        draft = window_draft(store, MON, date(2027, 2, 7), ["all"], "What do I have this week?")
        assert "Thyroxine" not in draft.text and "Chemistry exam" in draft.text

    def test_nothing_in_the_window(self, store):
        draft = window_draft(store, date(2027, 4, 1), date(2027, 4, 3), ["dates"], "Any birthdays in the next 2 days?")
        assert draft.text.startswith("No important dates from Thursday 1 April")


class TestPlanner:

    def test_day_plan_order_clash_deadline_tomorrow_and_medicines(self, store):
        draft = plan_draft(store, MON, MON, MON)
        text = draft.text
        order = [text.index(x) for x in ("11:00 am", "by 3:00 pm", "6:00 pm", "6:30 pm")]
        assert order == sorted(order)
        assert "Clash" in text and "mehendi" in text.lower() and draft.clash
        assert "Coming up tomorrow" in text and "Chemistry exam" in text
        assert "Thyroxine" in text

    def test_a_deadline_does_not_clash(self, store):
        slots, _, _ = slots_for(store, MON)
        assert all("tax papers" not in (a.label + b.label).lower() for a, b in clashes(slots))

    def test_a_night_shift_clashes_with_a_late_event(self, tmp_path):
        s = MemoryStore(tmp_path, "n", SECRET, HashEmbedder())
        s.add("From 1 to 5 February I'm on the night shift, 10 pm to 6 am.", SAID)
        s.add("Farewell dinner on 2 February from 8 pm to 11 pm.", SAID)
        assert plan_draft(s, date(2027, 2, 2), date(2027, 2, 2), MON).clash
        assert not plan_draft(s, date(2027, 2, 3), date(2027, 2, 3), MON).clash

    def test_week_plan_lists_only_days_with_something(self, store):
        draft = plan_draft(store, MON, date(2027, 2, 7), MON)
        assert "Monday 1 February:" in draft.text and "Saturday 6 February" in draft.text  # passport by 6 Feb
        assert "\nSunday 7 February:" not in draft.text  # an empty day gets no line (the header still names the range)

    def test_a_second_time_in_one_sentence_gets_its_own_line(self, tmp_path):
        s = MemoryStore(tmp_path, "t", SECRET, HashEmbedder())
        s.add("Eye check-up on 1 February at 11 am. I have a vision test the same morning at 9 am.", SAID)
        text = plan_draft(s, MON, MON, MON).text
        assert text.index("9:00 am") < text.index("11:00 am") and "vision test" in text.lower()


class TestStatusAndSpecific:

    def test_status_answers(self, store):
        gas = find(store, "Gas bill")
        assert status_draft(gas, [], "Is the gas bill paid?").text.startswith("No, Gas bill (₹1,120")
        store.add("I paid the gas bill.", date(2027, 1, 30))
        assert status_draft(store.get(gas["id"]), store.history(gas["id"]), "Is the gas bill paid?").text.startswith("Yes")
        passport = find(store, "passport")
        assert status_draft(passport, [], "Did I renew the passport?").text.startswith("I don't have a record of that")

    def test_specific_item_check_refuses_a_different_bill(self, store):
        model = FakeModel(["The electricity bill is due on 3 February."])  # what Qwen would wrongly say
        answer = Assistant(store, model).ask("When is my electricity bill due?", MON)
        assert answer.text == "I don't have anything about electricity saved." and model.calls == []

    def test_specific_item_check_lets_the_right_item_through(self, store):
        model = FakeModel(["Your gas bill of ₹1,120 is due on Wednesday 3 February 2027."])
        answer = Assistant(store, model).ask("When is my gas bill due?", MON)
        assert answer.text.startswith("Your gas bill") and answer.guard == "pass"

    def test_a_name_not_in_memory_is_refused(self, store):
        answer = Assistant(store, FakeModel(["Monday"])).ask("When is my Physics exam?", MON)
        assert answer.text == "I don't have anything about Physics saved."


class TestLead:
    """S4b: the code's list is shown exactly as built; Qwen writes only a lead sentence and an optional closing line."""

    def test_a_good_lead_goes_before_the_list_unchanged(self, store):
        model = FakeModel([{"lead": "Here's what you have to pay this week.", "closing": "Want a reminder?"}])
        answer = Assistant(store, model).ask("What do I have to pay this week?", MON)
        draft = window_draft(store, MON, date(2027, 2, 7), ["pay"], "What do I have to pay this week?")
        assert answer.text == "Here's what you have to pay this week.\n" + draft.text + "\nWant a reminder?"
        assert answer.guard == "pass" and answer.route == "window"
        assert model.calls[0]["json_schema"]["required"] == ["lead", "closing"]

    def test_a_lead_that_states_an_amount_is_retried_then_dropped(self, store):
        bad = {"lead": "You owe ₹950 this week.", "closing": ""}
        answer = Assistant(store, FakeModel([bad, bad])).ask("What do I have to pay this week?", MON)
        assert answer.guard == "fallback" and answer.text.startswith("To pay from Monday 1 February")
        assert "lead states a date, amount or time" in answer.problems

    def test_a_lead_that_denies_a_clash_is_refused(self, store):
        bad = {"lead": "Good news, no clashes today!", "closing": ""}
        good = {"lead": "Here's your day. Watch out for one clash.", "closing": ""}
        answer = Assistant(store, FakeModel([bad, good])).ask("Plan my day.", MON)
        assert answer.guard == "retry" and "lead denies the clash" in answer.problems and "Clash" in answer.text


class TestS4bFixes:

    def test_due_tomorrow_covers_money_to_receive(self, tmp_path):
        s = MemoryStore(tmp_path, "d", SECRET, HashEmbedder())
        s.add("Ravi owes me ₹1,500 and will pay by 2 February.", SAID)
        s.add("Water bill ₹300 due on 2 February.", SAID)
        decided = route_by_rules("Is anything due tomorrow?", MON)
        assert decided.kind == "window" and decided.focus == ["due"]
        text = window_draft(s, *decided.window, decided.focus, "Is anything due tomorrow?").text
        assert text.index("Due on Tuesday 2 February:") < text.index("Water bill") < text.index("To receive:") < text.index("Ravi")

    def test_an_unnamed_period_is_not_stated(self, store):
        decided = route_by_rules("Which bills do I still have to pay?", MON)
        assert decided.named is False
        text = window_draft(store, *decided.window, decided.focus, "Which bills do I still have to pay?", named=decided.named).text
        assert text.startswith("To pay:") and "May" not in text

    def test_fact_line_shows_only_current_values_unless_asked(self, tmp_path):
        from chirii.app.assistant import fact_line_v2
        s = MemoryStore(tmp_path, "h", SECRET, HashEmbedder())
        visit = s.add("Dentist appointment on 8 February at 5 pm.", SAID)["item"]["id"]
        s.add("The dentist moved my appointment to 9 February.", SAID)
        line = fact_line_v2(s.get(visit), s.history(visit), MON, "When is my dentist appointment?")
        assert "8 February" not in line and "Tuesday 9 February 2027" in line
        asked = fact_line_v2(s.get(visit), s.history(visit), MON, "What was the original date of the dentist appointment?")
        assert "2027-02-08" in asked

    def test_old_date_in_an_answer_is_caught_unless_asked(self, tmp_path):
        s = MemoryStore(tmp_path, "o", SECRET, HashEmbedder())
        s.confirm(s.add("Dentist appointment on 8 February at 5 pm.", SAID)["item"]["id"])
        s.add("The dentist moved my appointment to 9 February.", SAID)
        model = FakeModel(["Your dentist appointment is on 8 February.", "Your dentist appointment is on Tuesday 9 February 2027."])
        answer = Assistant(s, model).ask("When is my dentist appointment?", MON)
        assert answer.guard == "retry" and "9 February" in answer.text

    def test_items_with_several_dates_list_all_with_labels_or_the_next(self, tmp_path):
        from chirii.app.assistant import fact_line_v2
        s = MemoryStore(tmp_path, "m", SECRET, HashEmbedder())
        item = s.add("Mid-term exams: Maths on 1 February, Physics on 3 February, Chemistry on 5 February, all at 9 am.", SAID)["item"]
        line = fact_line_v2(item, [], date(2027, 2, 2), "When are my exams?")
        assert "Maths on Monday 1 February 2027 (already past)" in line and "Physics on Wednesday 3 February 2027" in line
        assert "Chemistry on Friday 5 February 2027" in line
        nxt = fact_line_v2(item, [], date(2027, 2, 2), "When is my next exam?")
        assert "Physics on Wednesday 3 February 2027" in nxt and "Chemistry" not in nxt.split("Current details")[1]


class TestS3Bugs:

    def test_fact_lines_show_every_date(self, tmp_path):
        s = MemoryStore(tmp_path, "d", SECRET, HashEmbedder())
        item = s.add("Mid-term exams: Maths on 8 February, Physics on 10 February, all at 9 am.", SAID)["item"]
        line = fact_line(item, [])
        assert "Monday 8 February 2027" in line and "Wednesday 10 February 2027" in line

    def test_fallback_does_not_quote_an_old_date(self, tmp_path):
        s = MemoryStore(tmp_path, "f", SECRET, HashEmbedder())
        visit = s.add("Dentist appointment on 8 February at 5 pm.", SAID)["item"]["id"]
        s.add("The dentist moved my appointment to 9 February.", SAID)
        text = code_answer([s.get(visit)], {visit: s.history(visit)})
        assert "8 February" not in text and "Tuesday 9 February 2027" in text

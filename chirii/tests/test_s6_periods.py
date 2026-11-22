"""S6 (owner, before the freeze): periods that end at a date, whole months, and "before <saved event>". Own examples."""

from datetime import date

import pytest

from chirii.app.assistant import Assistant
from chirii.app.memory import store as store_module
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.periods import read_period
from chirii.app.router import route
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
SUN = date(2027, 3, 14)        # a Sunday


@pytest.fixture(autouse=True)
def fresh():
    store_module._open.clear()
    yield
    store_module._open.clear()


def window(q, today=SUN, items=None):
    p = read_period(q, today, items)
    return (p.start, p.end) if p else None


class TestUpToADate:

    @pytest.mark.parametrize("question,end", [
        ("Which bills do I have to pay before 1 April?", date(2027, 3, 31)),     # "before" leaves the day out
        ("What do I have to pay by 1 April?", date(2027, 4, 1)),                  # "by" and "until" include it
        ("Which fees are due until 1 April?", date(2027, 4, 1)),
        ("Anything due till April 1st?", date(2027, 4, 1)),
        ("What must I pay before the 3rd of April?", date(2027, 4, 2)),
        ("What do I have to pay before 10 Jan?", date(2028, 1, 9)),                        # next occurrence
        ("What do I have to pay before 10 Jan 2028?", date(2028, 1, 9)),
        ("Which bills are due before Saturday?", date(2027, 3, 19)),
        ("Anything I have to pay by Friday?", date(2027, 3, 19)),
        ("What do I need to pay by next Friday?", date(2027, 3, 19)),
        ("What do I need to pay by tomorrow?", date(2027, 3, 15)),
        ("Anything due before Sunday?", date(2027, 3, 20)),                       # today is Sunday: the next one
    ])
    def test_from_today_up_to_it(self, question, end):
        assert window(question) == (SUN, end)
        assert route(question, SUN).kind == "window"

    def test_a_date_already_today_with_by_is_today(self):
        assert window("What do I have to pay by 14 March?") == (SUN, SUN)

    def test_other_wording_is_unchanged(self):
        assert window("Anything to pay before the end of the month?") == (SUN, date(2027, 3, 31))
        assert window("What's on 20 March?") == (date(2027, 3, 20), date(2027, 3, 20))
        assert window("Is the rent due before 5 pm?") is None


class TestWholeMonth:

    @pytest.mark.parametrize("question,first,last", [
        ("Which bills are due in July?", date(2027, 7, 1), date(2027, 7, 31)),
        ("What do I have to pay in Feb?", date(2028, 2, 1), date(2028, 2, 29)),        # next February, a leap year
        ("What do I have to pay in March?", date(2027, 3, 1), date(2027, 3, 31)),      # the current month counts
        ("What's due in December 2028?", date(2028, 12, 1), date(2028, 12, 31)),
        ("Anything due in May next year?", date(2028, 5, 1), date(2028, 5, 31)),
        ("What am I paying during August?", date(2027, 8, 1), date(2027, 8, 31)),
        ("Bills in the month of June?", date(2027, 6, 1), date(2027, 6, 30)),
    ])
    def test_the_whole_month(self, question, first, last):
        assert window(question) == (first, last)

    def test_in_a_month_is_a_list(self):
        assert route("Which bills are due in July?", SUN).kind == "window"
        assert route("What do I have to pay in December 2028?", SUN).kind == "window"


def store_with(tmp_path, sentences):
    s = MemoryStore(tmp_path, "u", SECRET, HashEmbedder())
    for t in sentences:
        s.add(t, SUN)
    return s


class TestBeforeASavedEvent:
    """Diagnosis (S6): these already worked in the app. The earlier report came from calling the router without the
    saved memories, so no event could be found. These tests go through the whole pipeline."""

    SENTENCES = ["Priya's birthday is on 2 April.", "Amma's eye check-up on 25 March at 10 am.",
                 "Electricity bill of ₹1,860 is due on 20 March.", "Water bill ₹420 due on 30 March.",
                 "Gas bill ₹900 due on 5 April."]

    @pytest.mark.parametrize("question,end,listed,left_out", [
        ("Anything to pay before Priya's birthday?", date(2027, 4, 1), ["1,860", "420"], ["900"]),
        ("What do I have to pay before Amma's check-up?", date(2027, 3, 24), ["1,860"], ["420", "900"]),
        ("Which bills are due before Amma's eye check-up?", date(2027, 3, 24), ["1,860"], ["420", "900"]),
    ])
    def test_the_list_ends_the_day_before_the_event(self, tmp_path, question, end, listed, left_out):
        s = store_with(tmp_path, self.SENTENCES)
        r = route(question, SUN, None, s.items(include_updates=False))
        assert r.kind == "window" and r.window == (SUN, end) and r.period.kind == "event"
        a = Assistant(s, FakeModel()).ask(question, SUN)
        assert a.route == "window"
        assert all(x in a.text for x in listed) and not any(x in a.text for x in left_out)

    def test_without_the_saved_memories_no_event_is_found(self):
        assert route("Anything to pay before Priya's birthday?", SUN).kind == "general"   # what the earlier check saw

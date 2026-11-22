"""S6 long-list summaries (own examples)."""

from datetime import date

import pytest

from chirii.app.api.main import create_app
from chirii.app.assistant import Assistant
from chirii.app.memory import store as store_module
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.summary import LONG, SHOWN, summarise
from chirii.tests.helpers import logged_in
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
SAID = date(2027, 6, 1)
DAYS = ["Thursday 1 July", "Friday 2 July", "Saturday 3 July", "Sunday 4 July", "Monday 5 July", "Tuesday 6 July",
        "Wednesday 7 July", "Thursday 8 July", "Friday 9 July", "Saturday 10 July", "Sunday 11 July", "Monday 12 July",
        "Tuesday 13 July", "Wednesday 14 July", "Thursday 15 July"]


@pytest.fixture(autouse=True)
def fresh():
    store_module._open.clear()
    yield
    store_module._open.clear()


def bills(n):
    return "To pay in July:\n" + "\n".join(f"- {DAYS[i]}: Bill {i + 1}, ₹{100 * (i + 1)}" for i in range(n)) + \
        f"\nTotal to pay: ₹{sum(100 * (i + 1) for i in range(n)):,}."


def test_a_list_of_up_to_twelve_is_shown_whole():
    assert summarise(bills(LONG)) is None


def test_a_long_list_is_summarised_with_count_span_first_entries_and_total():
    short = summarise(bills(15))
    lines = short.split("\n")
    assert lines[0] == "To pay in July:"
    assert lines[1] == f"15 items, from Thursday 1 July to Thursday 15 July. The first {SHOWN}:"
    assert lines[2:7] == [f"- {DAYS[i]}: Bill {i + 1}, ₹{100 * (i + 1)}" for i in range(5)]
    assert lines[7] == "- …and 10 more. Tap \"Show all\" to see the full list."
    assert lines[8] == "Total to pay: ₹12,000."


def test_clashes_are_counted_and_a_hidden_section_heading_is_dropped():
    text = "Your plan for Saturday 12 June:\n" + "\n".join(f"- {9 + i}:00 am: Game {i}" for i in range(11)) + \
        "\n- Clash: Game 1 (10:00 am) overlaps with Game 2 (11:00 am).\n- Clash: Game 3 overlaps with Game 4.\n" \
        "Also:\n- Return the cord"
    short = summarise(text)
    assert "12 items, with 2 clashes (see all)" in short
    assert "Clash:" not in short and "Also:" not in short and "…and 7 more" in short


def test_to_receive_totals_stay_visible():
    text = bills(14) + "\nTo receive:\n- Friday 16 July: Kabir owes you, ₹700\nTotal to receive: ₹700."
    short = summarise(text)
    assert "Total to pay: ₹10,500." in short and "Total to receive: ₹700." in short and "15 items" in short


def make_store(tmp_path, n):
    s = MemoryStore(tmp_path, "u", SECRET, HashEmbedder())
    for i in range(n):
        s.add(f"Pay the shop {i + 1} bill of ₹{100 * (i + 1)} on {i + 1} July.", SAID)
    return s


def test_the_assistant_shows_the_summary_and_keeps_the_full_list(tmp_path):
    a = Assistant(make_store(tmp_path, 15), FakeModel()).ask("Which bills are due next month?", SAID)
    assert a.route == "window" and "15 items" in a.text and "…and 10 more" in a.text
    assert a.full_text and all(f"shop {i + 1} bill" in a.full_text.lower() or f"₹{100 * (i + 1):,}" in a.full_text
                               for i in range(15))
    assert "Total to pay: ₹12,000." in a.text and "Total to pay: ₹12,000." in a.full_text


def test_a_short_list_has_no_full_text(tmp_path):
    a = Assistant(make_store(tmp_path, 4), FakeModel()).ask("Which bills are due next month?", SAID)
    assert a.route == "window" and a.full_text is None and "items, from" not in a.text


def test_the_api_returns_both(tmp_path):
    c = logged_in(create_app(tmp_path, SECRET, HashEmbedder(), FakeModel(), today=lambda: SAID))
    for i in range(14):
        c.post("/api/memories", json={"text": f"Pay the shop {i + 1} bill of ₹{100 * (i + 1)} on {i + 1} July."})
    r = c.post("/api/ask", json={"question": "Which bills are due next month?"}).json()
    assert "14 items" in r["answer"] and r["full_answer"].count("\n- ") >= 14


def test_the_summary_names_only_a_few_unconfirmed_items(tmp_path):
    a = Assistant(make_store(tmp_path, 15), FakeModel()).ask("Which bills are due next month?", SAID)
    assert "(This uses 15 things you haven't confirmed yet, including" in a.text
    assert a.full_text.count("\"Pay the shop") == 15

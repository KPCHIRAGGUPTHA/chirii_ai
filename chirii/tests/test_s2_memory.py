"""
S2 checks: sentences become the right items, updates change the right item, search and date queries work, the SQLite
rows and FAISS vectors stay in step, users are kept apart, "delete all my data" leaves nothing behind, and no memory
text reaches the logs. Uses the test embedder, so no model download is needed.
"""

import logging
from datetime import date

import pytest

from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.extract import extract
from chirii.app.memory.store import MemoryStore, user_folder_name
from chirii.eval.s2_fresh_sentences import FRESH, SAID_ON
from chirii.eval.s2_gold import sentences

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
DEC1 = date(2026, 12, 1)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path, "priya@example.com", SECRET, HashEmbedder())


class TestExtract:

    @pytest.mark.parametrize("text,expected", FRESH)
    def test_fresh_sentences(self, text, expected):
        item = extract(text, SAID_ON).to_dict()
        assert {k: item.get(k) for k in expected} == expected

    def test_every_development_sentence(self):
        """All 66 sentences in the S1 memories become their intended item (fields the eval specifies)."""
        fields = ["date", "end_date", "time", "end_time", "amount", "day_of_month", "weekdays", "other_dates", "other_times", "other_amounts"]
        for s in sentences():
            item = extract(s["said"], s["said_on"]).to_dict()
            assert item["kind"] == s["gold"]["type"], s["said"]
            for name in fields:
                if name in s["gold"]:
                    assert item[name] == s["gold"][name], (s["said"], name)

    @pytest.mark.parametrize("text,field,value", [
        ("Sugar now comes on 12 December, not 11.", "date", "2026-12-12"),
        ("Sugar now comes on 12 December, not 11.", "other_dates", []),
        ("Release moved to 8 January.", "date", "2027-01-08"),            # nearest year: next January
        ("Gas cylinder was delivered on 25 November.", "date", "2026-11-25"),  # nearest year: last month
        ("Meet Ravi next Friday at 5 pm.", "date", "2026-12-04"),
        ("Call the plumber tomorrow at 9.30 am.", "time", "09:30"),
        ("Fee of ₹1.95 lakh due on 5 Jan.", "amount", 195000),
        ("Bring 50 bags and 2 boxes on 3 Dec.", "amount", None),          # counts are not money
    ])
    def test_details(self, text, field, value):
        assert extract(text, DEC1).to_dict()[field] == value


class TestStore:

    def test_add_get_and_index_stay_in_step(self, store):
        store.add("My BESCOM electricity bill is ₹1,860 and it's due on 12 December.", DEC1)
        store.add("Dentist appointment with Dr. Kavitha on 10 December at 6 pm.", DEC1)
        assert len(store.items()) == 2 and store.index.ntotal == 2 and store.is_consistent()

    def test_index_survives_reopening_and_can_be_rebuilt(self, store, tmp_path):
        store.add("Amma's birthday is on 22 December.", DEC1)
        again = MemoryStore(tmp_path, "priya@example.com", SECRET, HashEmbedder())
        assert again.index.ntotal == 1 and again.is_consistent()
        assert again.rebuild_index() == 1 and again.is_consistent()

    def test_delete_one_item_removes_row_and_vector(self, store):
        first = store.add("Gym fee of ₹2,500 is due on 1 January.", DEC1)["item"]["id"]
        store.add("Amma's birthday is on 22 December.", DEC1)
        assert store.delete_item(first)
        assert store.get(first) is None and store.index.ntotal == 1 and store.is_consistent()

    def test_search_finds_by_words_and_meaning(self, store):
        store.add("Shopping list for Pongal: rice, jaggery, moong dal, ghee, sugarcane and turmeric plants.", DEC1)
        store.add("Divya's school term fee of ₹24,000 must be paid by 18 December.", DEC1)
        assert "Pongal" in store.search("What do I need to buy for Pongal?", k=1)[0]["item"]["text"]

    def test_agenda_expands_dates_ranges_months_and_weeks(self, store):
        store.add("From 14 to 18 December I'm on the night shift, 10 pm to 6 am.", DEC1)
        store.add("My home loan EMI of ₹18,500 goes out on the 5th of every month.", DEC1)
        store.add("I take Divya to her maths tuition every Tuesday and Thursday at 5 pm; it ends at 6:30 pm.", DEC1)
        days = [(e["day"], e["item"]["kind"]) for e in store.agenda(date(2026, 12, 14), date(2026, 12, 20))]
        assert days.count(("2026-12-15", "routine")) == 1 and days.count(("2026-12-17", "routine")) == 1
        assert sum(1 for d, k in days if k == "shift") == 5
        january = store.agenda(date(2027, 1, 1), date(2027, 1, 31))
        assert [e["day"] for e in january if e["item"]["kind"] == "bill"] == ["2027-01-05"]


class TestUpdates:

    def test_paid_marks_the_right_bill(self, store):
        water = store.add("Our Metro Water bill is ₹420, due on 10 December.", DEC1)["item"]["id"]
        store.add("The electricity bill is ₹2,150, due on 20 December.", DEC1)
        result = store.add("I paid the water bill today.", date(2026, 12, 9))
        assert result["linked_to"] == water and store.get(water)["status"] == "paid"
        assert store.agenda(date(2026, 12, 1), date(2026, 12, 31))[0]["item"]["amount"] == 2150  # paid bill drops out

    def test_moved_keeps_history_and_same_time(self, store):
        visit = store.add("My next check-up is with Dr. Ramaswamy at the hospital on 18 December at 11 am.", DEC1)["item"]["id"]
        store.add("Dr. Ramaswamy said: walk 30 minutes every morning, avoid sweets and reduce salt.", DEC1)
        result = store.add("The doctor's appointment is moved to 21 December, same time.", date(2026, 12, 10))
        item = store.get(visit)
        assert result["linked_to"] == visit and item["date"] == "2026-12-21" and item["time"] == "11:00"
        assert store.history(visit)[0]["old_value"] == "2026-12-18"

    def test_part_payment_leaves_the_rest_owed(self, store):
        credit = store.add("Mrs. Deshpande owes ₹3,450 on credit and promised to pay by 13 December.", DEC1)["item"]["id"]
        store.add("Mrs. Deshpande paid ₹2,000 today; she still owes the rest.", date(2026, 12, 12))
        item = store.get(credit)
        assert item["status"] == "active" and item["remaining"] == 1450

    def test_unclear_update_is_kept_unlinked_not_guessed(self, store):
        water = store.add("Our Metro Water bill is ₹420, due on 10 December.", DEC1)["item"]["id"]
        gas = store.add("I paid the gas bill.", DEC1)  # shares only "bill" with the water bill
        assert gas["linked_to"] is None and gas["item"]["status"] == "unlinked"
        assert store.get(water)["status"] == "active"
        vague = store.add("The thing moved to 12 December.", DEC1)
        assert vague["linked_to"] is None and vague["item"]["status"] == "unlinked"

    def test_generic_update_links_when_only_one_item_fits(self, store):
        emi = store.add("Laptop EMI of ₹3,200 on the 7th of every month.", DEC1)["item"]["id"]
        store.add("Amma's birthday is on 22 December.", DEC1)
        assert store.add("I paid the EMI.", DEC1)["linked_to"] == emi

    def test_applied_updates_are_hidden_from_search(self, store):
        store.add("Hostel mess fee of ₹38,500 is due on 20 December.", DEC1)
        store.add("I paid the mess fee today.", date(2026, 12, 15))
        assert all(r["item"]["kind"] != "update" for r in store.search("mess fee"))


class TestPrivacy:

    def test_users_are_kept_apart(self, tmp_path):
        a = MemoryStore(tmp_path, "a@example.com", SECRET, HashEmbedder())
        b = MemoryStore(tmp_path, "b@example.com", SECRET, HashEmbedder())
        a.add("Amma's birthday is on 22 December.", DEC1)
        assert b.items() == [] and b.search("birthday") == [] and a.dir != b.dir

    def test_folder_name_does_not_reveal_the_user(self, store):
        assert "priya" not in str(store.dir).lower()
        assert store.dir.name == user_folder_name("priya@example.com", SECRET)
        assert user_folder_name("priya@example.com", "another-secret-that-is-long-enough-000000") != store.dir.name

    def test_delete_all_my_data_leaves_nothing(self, tmp_path):
        a = MemoryStore(tmp_path, "a@example.com", SECRET, HashEmbedder())
        b = MemoryStore(tmp_path, "b@example.com", SECRET, HashEmbedder())
        a.add("Amma's birthday is on 22 December.", DEC1)
        a.add("I paid the gym fee.", DEC1)
        b.add("Mom's birthday is on 24 December.", DEC1)
        counts = a.delete_all()
        assert counts == {"items": 2, "vectors": 2}
        assert not a.dir.exists()
        assert [p for p in tmp_path.rglob("*") if a.dir.name in str(p)] == []
        reopened = MemoryStore(tmp_path, "a@example.com", SECRET, HashEmbedder())
        assert reopened.items() == [] and reopened.index.ntotal == 0
        assert len(b.items()) == 1  # the other user is untouched

    def test_no_memory_text_in_logs(self, store, caplog):
        with caplog.at_level(logging.DEBUG, logger="chirii"):
            store.add("Dentist appointment with Dr. Kavitha on 10 December at 6 pm.", DEC1)
            store.search("When is the dentist?")
            store.delete_all()
        text = caplog.text.lower()
        assert caplog.records and "kavitha" not in text and "dentist" not in text and "priya" not in text

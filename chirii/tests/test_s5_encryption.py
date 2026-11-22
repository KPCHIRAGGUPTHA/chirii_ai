"""
S5 encryption at rest (option C): the file is unreadable without the key, a wrong key is refused, a crash during a
write leaves the old file whole, one user's key opens only their own file, a changed file is refused, the key check,
the per-user lock, and data surviving a restart.
"""

import os
import shutil
import threading
from datetime import date

import pytest

from chirii.app.memory import store as store_module
from chirii.app.memory import vault
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.memory.vault import DataError, KeyError_, check_key, new_key

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
KEY_A, KEY_B = new_key(), new_key()
DAY = date(2027, 5, 3)
SECRET_TEXT = "Locker code for the gym is 4417 and Dr. Varghese visit on 9 May."


def restart():
    """What a restart of the app forgets: the databases held in memory."""
    store_module._open.clear()


def make(tmp_path, user="u1", key=KEY_A):
    return MemoryStore(tmp_path, user, SECRET, HashEmbedder(), data_key=key)


@pytest.fixture(autouse=True)
def fresh():
    restart()
    yield
    restart()


def test_the_file_on_disk_is_unreadable_without_the_key(tmp_path):
    s = make(tmp_path)
    s.add(SECRET_TEXT, DAY)
    files = [p for p in s.dir.rglob("*") if p.is_file()]
    assert [p.name for p in files] == ["memory.enc"]                     # no plaintext database, no index file
    blob = files[0].read_bytes()
    assert b"4417" not in blob and b"Varghese" not in blob and b"SQLite format" not in blob
    assert blob.startswith(vault.MAGIC)


def test_data_survives_a_restart_with_the_right_key(tmp_path):
    s = make(tmp_path)
    item = s.add(SECRET_TEXT, DAY)["item"]
    restart()
    again = make(tmp_path)
    assert again.get(item["id"])["text"] == SECRET_TEXT
    assert again.is_consistent() and again.search("gym locker")[0]["item"]["id"] == item["id"]


def test_vectors_are_stored_not_recomputed_on_opening(tmp_path):
    s = make(tmp_path)
    s.add(SECRET_TEXT, DAY)
    restart()

    class Counting(HashEmbedder):
        calls = 0

        def embed(self, texts):
            Counting.calls += len(texts)
            return super().embed(texts)
    MemoryStore(tmp_path, "u1", SECRET, Counting(), data_key=KEY_A)
    assert Counting.calls == 0


def test_a_wrong_key_is_refused(tmp_path):
    make(tmp_path).add(SECRET_TEXT, DAY)
    restart()
    with pytest.raises(DataError):
        make(tmp_path, key=KEY_B)


def test_one_users_file_does_not_open_as_another_users(tmp_path):
    a, b = make(tmp_path, "asha"), make(tmp_path, "bala")
    a.add(SECRET_TEXT, DAY)
    b.add("Bala's note.", DAY)
    shutil.copy(a.vault.path, b.vault.path)                              # same main key, different user folder
    restart()
    with pytest.raises(DataError):
        make(tmp_path, "bala")
    assert make(tmp_path, "asha").items()[0]["text"] == SECRET_TEXT


def test_each_user_has_their_own_key(tmp_path):
    a, b = make(tmp_path, "asha"), make(tmp_path, "bala")
    assert a.vault.key != b.vault.key


def test_a_changed_file_is_refused(tmp_path):
    s = make(tmp_path)
    s.add(SECRET_TEXT, DAY)
    blob = bytearray(s.vault.path.read_bytes())
    blob[len(blob) // 2] ^= 0x01
    s.vault.path.write_bytes(bytes(blob))
    restart()
    with pytest.raises(DataError):
        make(tmp_path)


def test_a_crash_during_a_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    s = make(tmp_path)
    first = s.add("Water bill of ₹600 due on 10 May.", DAY)["item"]
    before = s.vault.path.read_bytes()

    def crash(*args):
        raise OSError("power cut")
    monkeypatch.setattr(vault.os, "replace", crash)
    with pytest.raises(OSError):
        s.add(SECRET_TEXT, DAY)
    monkeypatch.undo()
    assert s.vault.path.read_bytes() == before                           # the old file, untouched
    assert [i["id"] for i in s.items()] == [first["id"]]                 # and memory went back to it
    assert [p.name for p in s.dir.iterdir()] == ["memory.enc"]           # no temporary file left behind
    restart()
    assert [i["id"] for i in make(tmp_path).items()] == [first["id"]]


def test_the_key_check(tmp_path):
    assert check_key(tmp_path, KEY_A) == "created"
    assert check_key(tmp_path, KEY_A) == "ok"
    with pytest.raises(KeyError_):
        check_key(tmp_path, KEY_B)
    with pytest.raises(KeyError_):
        check_key(tmp_path, "too-short")


def test_no_new_check_value_when_encrypted_data_exists(tmp_path):
    make(tmp_path).add(SECRET_TEXT, DAY)
    with pytest.raises(KeyError_):
        check_key(tmp_path, KEY_B)                                       # data exists, check file missing


def test_two_store_objects_share_one_copy_under_one_lock(tmp_path):
    a, b = make(tmp_path), make(tmp_path)

    def add_many(store, n):
        for k in range(n):
            store.add(f"Note number {n}-{k}: buy milk.", DAY)
    threads = [threading.Thread(target=add_many, args=(x, n)) for x, n in ((a, 15), (b, 16))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    restart()
    again = make(tmp_path)
    assert len(again.items()) == 31 and again.is_consistent()


def test_a_plaintext_database_from_before_s5_is_encrypted_and_removed(tmp_path):
    import sqlite3
    from contextlib import closing
    folder = tmp_path / "users" / store_module.user_folder_name("u1", SECRET)
    folder.mkdir(parents=True)
    with closing(sqlite3.connect(folder / "memory.db")) as db, db:
        db.executescript(store_module.SCHEMA)
        db.execute("INSERT INTO items (kind, text, said_on, created_at, updated_at) VALUES ('note', ?, '2027-05-01', 'x', 'x')", (SECRET_TEXT,))
    (folder / "index.faiss").write_bytes(b"old index")
    s = make(tmp_path)
    assert s.items()[0]["text"] == SECRET_TEXT and s.is_consistent()
    assert sorted(p.name for p in folder.iterdir()) == ["memory.enc"]
    assert b"4417" not in (folder / "memory.enc").read_bytes()


def test_delete_all_removes_the_encrypted_file(tmp_path):
    s = make(tmp_path)
    s.add(SECRET_TEXT, DAY)
    s.delete_all()
    assert not s.dir.exists()
    assert make(tmp_path).items() == []

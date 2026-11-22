"""S5 key handling: .env edits that keep other lines, key rotation (also when interrupted), and encrypted backups."""

from datetime import date

import pytest

from chirii.app import keys
from chirii.app.memory import store as store_module
from chirii.app.memory.embeddings import HashEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.memory.vault import DataError, KeyError_, check_key, new_key

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
DAY = date(2027, 5, 3)


@pytest.fixture(autouse=True)
def fresh():
    store_module._open.clear()
    yield
    store_module._open.clear()


@pytest.fixture
def setup(tmp_path):
    env, data = tmp_path / ".env", tmp_path / "data"
    key = new_key()
    env.write_text(f"# my settings\nCHIRII_SECRET_KEY={SECRET}\nCHIRII_DATA_KEY={key}\nCHIRII_LOG_LEVEL=INFO\n", encoding="utf-8")
    check_key(data, key)
    for user in ("asha", "bala"):
        MemoryStore(data, user, SECRET, HashEmbedder(), data_key=key).add(f"{user} pays the milkman ₹900 on the 1st.", DAY)
    store_module._open.clear()
    return env, data, key


def opens(data, user, key):
    store_module._open.clear()
    return MemoryStore(data, user, SECRET, HashEmbedder(), data_key=key).items()[0]["text"]


def test_set_env_keeps_every_other_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nA=1\nB=2\n", encoding="utf-8")
    keys.set_env(env, "B", "3")
    keys.set_env(env, "C", "4")
    keys.set_env(env, "A", None)
    assert env.read_text(encoding="utf-8") == "# comment\nB=3\nC=4\n"


def test_rotate_re_encrypts_everything_with_a_new_key(setup):
    env, data, old = setup
    assert keys.rotate(env, data) == 2
    values = keys.read_env(env)
    new = values["CHIRII_DATA_KEY"]
    assert new != old and "CHIRII_DATA_KEY_PREVIOUS" not in values
    assert values["CHIRII_SECRET_KEY"] == SECRET and values["CHIRII_LOG_LEVEL"] == "INFO"
    assert opens(data, "asha", new).startswith("asha") and opens(data, "bala", new).startswith("bala")
    with pytest.raises(DataError):
        opens(data, "asha", old)
    with pytest.raises(KeyError_):
        check_key(data, old)
    assert not list(data.glob("**/*.new"))


def test_an_interrupted_rotation_is_finished_by_running_it_again(setup, monkeypatch):
    env, data, old = setup
    monkeypatch.setattr(keys, "finish_rotation", lambda d: (_ for _ in ()).throw(OSError("power cut")))
    with pytest.raises(OSError):
        keys.rotate(env, data)
    monkeypatch.undo()
    values = keys.read_env(env)
    assert values["CHIRII_DATA_KEY_PREVIOUS"] == old                     # .env has the new key and remembers the old
    assert opens(data, "asha", old).startswith("asha")                   # the old files are still in place
    keys.rotate(env, data)
    values = keys.read_env(env)
    assert "CHIRII_DATA_KEY_PREVIOUS" not in values
    assert opens(data, "asha", values["CHIRII_DATA_KEY"]).startswith("asha")


def test_a_rotation_stopped_before_env_changed_leaves_the_old_key_working(setup, monkeypatch):
    env, data, old = setup
    monkeypatch.setattr(keys, "set_env", lambda *a: (_ for _ in ()).throw(OSError("power cut")))
    with pytest.raises(OSError):
        keys.rotate(env, data)
    monkeypatch.undo()
    assert keys.read_env(env)["CHIRII_DATA_KEY"] == old and opens(data, "asha", old).startswith("asha")
    assert keys.rotate(env, data) == 2                                   # the stale .new files are cleared first


def test_backup_copies_encrypted_files_only(setup, tmp_path):
    env, data, key = setup
    target = keys.backup(data, tmp_path / "backups")
    names = sorted(p.name for p in target.rglob("*") if p.is_file())
    assert names == ["key.check", "memory.enc", "memory.enc"]
    assert not any(b"milkman" in p.read_bytes() for p in target.rglob("*") if p.is_file())
    assert not any(key.encode() in p.read_bytes() for p in target.rglob("*") if p.is_file())
    with pytest.raises(SystemExit):
        keys.backup(data, data / "inside")


# ---------------------------------------------------------------- CHIRII_SECRET_KEY rotation (owner, 22 Nov)

def secret_setup(setup):
    from chirii.app.accounts import Accounts
    env, data, key = setup
    accounts = Accounts(data, key)
    for user in ("asha", "bala"):
        accounts.add(user, f"{user}-long-test-password")
    MemoryStore(data, "demo", SECRET, HashEmbedder(), data_key=key).add("demo pays rent ₹25,000 on the 5th.", DAY)
    store_module._open.clear()
    return env, data, key


def opens_with(data, user, secret, key):
    store_module._open.clear()
    return [i["text"] for i in MemoryStore(data, user, secret, HashEmbedder(), data_key=key).items()]


def test_rotate_secret_keeps_every_users_data_reachable(setup):
    env, data, key = secret_setup(setup)
    stray = data / "users" / "0123456789abcdef0123456789abcdef"     # belongs to no known user: left alone
    stray.mkdir()
    (stray / "memory.enc").write_bytes((data / "users" / store_module.user_folder_name("asha", SECRET) / "memory.enc").read_bytes())
    assert keys.rotate_secret(env, data) == (3, 1)
    values = keys.read_env(env)
    new = values["CHIRII_SECRET_KEY"]
    assert new != SECRET and len(new) >= 32 and "CHIRII_SECRET_KEY_PREVIOUS" not in values
    assert values["CHIRII_DATA_KEY"] == key and values["CHIRII_LOG_LEVEL"] == "INFO"   # data key unchanged
    for user in ("asha", "bala", "demo"):
        assert opens_with(data, user, new, key)[0].startswith(user)
        assert not (data / "users" / store_module.user_folder_name(user, SECRET)).exists()   # old folder gone
    assert (stray / "memory.enc").exists()
    assert opens_with(data, "asha", SECRET, key) == []                # the old secret finds nothing any more


def test_rotate_secret_finishes_after_an_interruption(setup, monkeypatch):
    env, data, key = secret_setup(setup)
    real, calls = keys.move_user_data, []

    def stops_halfway(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("power cut")
    monkeypatch.setattr(keys, "move_user_data", stops_halfway)
    with pytest.raises(RuntimeError):
        keys.rotate_secret(env, data)
    values = keys.read_env(env)
    assert values["CHIRII_SECRET_KEY_PREVIOUS"] == SECRET and values["CHIRII_SECRET_KEY"] != SECRET
    new = values["CHIRII_SECRET_KEY"]
    monkeypatch.setattr(keys, "move_user_data", real)
    assert keys.rotate_secret(env, data) == (3, 0)
    assert keys.read_env(env)["CHIRII_SECRET_KEY"] == new and "CHIRII_SECRET_KEY_PREVIOUS" not in keys.read_env(env)
    for user in ("asha", "bala", "demo"):
        assert opens_with(data, user, new, key)[0].startswith(user)


def test_rotate_secret_refuses_a_wrong_data_key(setup):
    env, data, key = secret_setup(setup)
    keys.set_env(env, "CHIRII_DATA_KEY", new_key())
    with pytest.raises(KeyError_):
        keys.rotate_secret(env, data)
    assert keys.read_env(env)["CHIRII_SECRET_KEY"] == SECRET          # nothing changed


def test_after_rotate_secret_the_app_shows_the_same_memories(setup):
    from chirii.app.api.main import create_app
    from chirii.tests.helpers import TEST_PASSWORD, logged_in
    from chirii.tests.test_slice import FakeModel
    env, data, key = setup
    before = create_app(data, SECRET, HashEmbedder(), FakeModel(), today=lambda: DAY, data_key=key)
    logged_in(before, "chitra").post("/api/memories", json={"text": "Chitra's gym fee is ₹1,500 on the 10th."})
    store_module._open.clear()
    keys.rotate_secret(env, data)
    after = create_app(data, keys.read_env(env)["CHIRII_SECRET_KEY"], HashEmbedder(), FakeModel(), today=lambda: DAY, data_key=key)
    texts = [m["text"] for m in logged_in(after, "chitra", TEST_PASSWORD).get("/api/memories").json()["items"]]
    assert texts == ["Chitra's gym fee is ₹1,500 on the 10th."]

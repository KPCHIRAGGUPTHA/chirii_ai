"""
S5 login: hashed passwords, session cookies, consent before use, lockout, and per-user isolation through the API.
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from chirii.app import accounts as accounts_module
from chirii.app.accounts import COOKIE, Accounts, Locked, Sessions
from chirii.app.api.main import create_app
from chirii.app.consent import CONSENT_VERSION
from chirii.app.memory import store as store_module
from chirii.app.memory.embeddings import HashEmbedder
from chirii.tests.helpers import TEST_PASSWORD, logged_in
from chirii.tests.test_slice import FakeModel

SECRET = "test-secret-that-is-long-enough-for-hmac-0123456789"
DAY = date(2027, 5, 3)


@pytest.fixture(autouse=True)
def fresh():
    store_module._open.clear()
    yield
    store_module._open.clear()


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path, SECRET, HashEmbedder(), FakeModel(), today=lambda: DAY)


class TestPasswords:

    def test_only_an_argon2id_hash_is_stored_and_the_file_is_encrypted(self, tmp_path):
        a = Accounts(tmp_path, SECRET)
        a.add("asha", TEST_PASSWORD)
        assert a.get("asha")["hash"].startswith("$argon2id$")
        raw = (tmp_path / "accounts.enc").read_bytes()
        assert TEST_PASSWORD.encode() not in raw and b"asha" not in raw and b"argon2" not in raw

    @pytest.mark.parametrize("name,password", [("as", TEST_PASSWORD), ("Asha Rao", TEST_PASSWORD), ("asha", "short"),
                                               ("asha12345678", "asha12345678")])
    def test_weak_names_and_passwords_are_refused(self, tmp_path, name, password):
        with pytest.raises(ValueError):
            Accounts(tmp_path, SECRET).add(name, password)

    def test_verify(self, tmp_path):
        a = Accounts(tmp_path, SECRET)
        a.add("asha", TEST_PASSWORD)
        assert a.verify("Asha", TEST_PASSWORD) == "asha"
        assert a.verify("asha", TEST_PASSWORD + "x") is None
        assert a.verify("nobody", TEST_PASSWORD) is None

    def test_lockout_after_five_wrong_passwords(self, tmp_path):
        a = Accounts(tmp_path, SECRET)
        a.add("asha", TEST_PASSWORD)
        for _ in range(5):
            assert a.verify("asha", "wrong-password-0") is None
        with pytest.raises(Locked):
            a.verify("asha", TEST_PASSWORD)                     # even the right password, while locked

    def test_the_accounts_file_needs_the_data_key(self, tmp_path):
        Accounts(tmp_path, SECRET).add("asha", TEST_PASSWORD)
        with pytest.raises(Exception):
            Accounts(tmp_path, "another-key-that-is-long-enough-0000000000").get("asha")


class TestSessions:

    def test_only_the_token_hash_is_kept(self):
        s = Sessions()
        token = s.create("asha")
        assert token not in str(s.table) and s.user(token) == "asha"

    def test_idle_sessions_expire(self, monkeypatch):
        s = Sessions()
        token = s.create("asha")
        now = accounts_module.time.time()
        monkeypatch.setattr(accounts_module.time, "time", lambda: now + accounts_module.IDLE_SECONDS + 1)
        assert s.user(token) is None


class TestApi:

    def test_nothing_without_login(self, app):
        c = TestClient(app)
        assert c.get("/api/memories").status_code == 401
        assert c.get("/api/me").status_code == 401
        assert c.post("/api/ask", json={"question": "x"}).status_code == 401
        assert c.post("/api/delete-all").status_code == 401

    def test_a_made_up_or_changed_cookie_is_refused(self, app):
        c = logged_in(app)
        c.cookies.set(COOKIE, c.cookies.get(COOKIE)[:-2] + "xx")
        assert c.get("/api/memories").status_code == 401

    def test_login_sets_a_strict_http_only_cookie(self, app):
        app.state.accounts.add("asha", TEST_PASSWORD)
        r = TestClient(app).post("/api/login", json={"username": "asha", "password": TEST_PASSWORD})
        cookie = r.headers["set-cookie"].lower()
        assert r.status_code == 200 and "httponly" in cookie and "samesite=strict" in cookie

    def test_wrong_password_and_lockout(self, app):
        app.state.accounts.add("asha", TEST_PASSWORD)
        c = TestClient(app)
        codes = [c.post("/api/login", json={"username": "asha", "password": "nope-nope-nope"}).status_code for _ in range(6)]
        assert codes == [401] * 5 + [429]

    def test_consent_is_needed_before_use(self, app):
        c = logged_in(app, consent=False)
        assert c.get("/api/me").json() == {"username": "tester", "consented": False}
        assert c.get("/api/memories").status_code == 403
        assert c.post("/api/consent", json={"version": "old"}).status_code == 409
        assert c.post("/api/consent", json={"version": CONSENT_VERSION}).json()["consented"] is True
        assert c.get("/api/memories").status_code == 200

    def test_logout_ends_the_session(self, app):
        c = logged_in(app)
        token = c.cookies.get(COOKIE)
        c.post("/api/logout")
        c.cookies.set(COOKIE, token)
        assert c.get("/api/memories").status_code == 401

    def test_users_never_see_each_others_memories(self, app):
        asha, bala = logged_in(app, "asha"), logged_in(app, "bala")
        item = asha.post("/api/memories", json={"text": "Asha's locker code is 4417."}).json()["item"]
        bala.post("/api/memories", json={"text": "Bala's gym fee of ₹900 is due on 9 May."})
        assert [i["text"] for i in asha.get("/api/memories").json()["items"]] == ["Asha's locker code is 4417."]
        assert [i["text"] for i in bala.get("/api/memories").json()["items"]] == ["Bala's gym fee of ₹900 is due on 9 May."]
        # the same item id in Bala's session means Bala's own item (ids are per user), never Asha's
        bala.delete(f"/api/memories/{item['id']}")
        assert len(asha.get("/api/memories").json()["items"]) == 1
        assert "4417" not in bala.post("/api/ask", json={"question": "What is my locker code?"}).text

    def test_delete_all_removes_only_my_data(self, app):
        asha, bala = logged_in(app, "asha"), logged_in(app, "bala")
        asha.post("/api/memories", json={"text": "Asha note one."})
        bala.post("/api/memories", json={"text": "Bala note one."})
        asha.post("/api/delete-all")
        assert asha.get("/api/memories").json()["items"] == []
        assert len(bala.get("/api/memories").json()["items"]) == 1

    def test_delete_account_needs_the_password_and_removes_everything(self, app, tmp_path):
        asha = logged_in(app, "asha")
        asha.post("/api/memories", json={"text": "Asha note one."})
        assert asha.post("/api/delete-account", json={"password": "wrong-password-0"}).status_code == 401
        r = asha.post("/api/delete-account", json={"password": TEST_PASSWORD})
        assert r.status_code == 200 and app.state.accounts.get("asha") is None
        assert asha.get("/api/memories").status_code == 401
        assert not list((tmp_path / "users").glob("*/memory.enc"))

    def test_requests_from_another_site_are_refused(self, app):
        c = logged_in(app)
        r = c.post("/api/memories", json={"text": "x"}, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403
        assert c.post("/api/memories", json={"text": "Note."}, headers={"Origin": "http://testserver"}).status_code == 200

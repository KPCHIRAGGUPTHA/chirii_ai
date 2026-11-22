"""Shared test helpers: a logged-in, consented API client (S5)."""

from fastapi.testclient import TestClient

from chirii.app.consent import CONSENT_VERSION

TEST_PASSWORD = "correct-horse-battery"     # test value only


def logged_in(app, username: str = "tester", password: str = TEST_PASSWORD, consent: bool = True) -> TestClient:
    if app.state.accounts.get(username) is None:
        app.state.accounts.add(username, password)
    client = TestClient(app)
    assert client.post("/api/login", json={"username": username, "password": password}).status_code == 200
    if consent:
        assert client.post("/api/consent", json={"version": CONSENT_VERSION}).status_code == 200
    return client

"""
S0 checks: settings refuse weak secrets, logging refuses personal data, and git never tracks private files.
"""

import logging
import subprocess

import pytest

from chirii.app.config import REPO_ROOT, Settings
from chirii.app.logging_setup import hash_user, log_event

GOOD_SECRET = "x" * 48


class TestSettings:

    def test_missing_secret_is_rejected(self, monkeypatch):
        monkeypatch.delenv("CHIRII_SECRET_KEY", raising=False)
        with pytest.raises(ValueError):
            Settings(_env_file=None)

    @pytest.mark.parametrize("secret", ["change-me", "replace-with-a-long-random-string", "short"])
    def test_placeholder_or_short_secret_is_rejected(self, monkeypatch, secret):
        monkeypatch.setenv("CHIRII_SECRET_KEY", secret)
        with pytest.raises(ValueError):
            Settings(_env_file=None)

    def test_real_secret_loads_and_is_hidden_from_repr(self, monkeypatch):
        monkeypatch.setenv("CHIRII_SECRET_KEY", GOOD_SECRET)
        monkeypatch.setenv("CHIRII_LLM_BASE_URL", "http://gpu-host:8000/v1")
        settings = Settings(_env_file=None)
        assert settings.llm_base_url == "http://gpu-host:8000/v1"
        assert GOOD_SECRET not in repr(settings)

    def test_env_example_has_no_usable_secret(self, monkeypatch):
        values = {}
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        assert "CHIRII_SECRET_KEY" in values
        monkeypatch.setenv("CHIRII_SECRET_KEY", values["CHIRII_SECRET_KEY"])
        with pytest.raises(ValueError):
            Settings(_env_file=None)


class TestLogging:

    def test_content_fields_are_refused(self):
        for field in ("message", "content", "text", "note", "name", "prompt", "reply"):
            with pytest.raises(ValueError):
                log_event("chat_reply", **{field: "My electricity bill is due"})

    def test_free_text_event_names_are_refused(self):
        with pytest.raises(ValueError):
            log_event("user asked about mother's hospital visit")

    def test_user_id_is_hashed(self, caplog):
        with caplog.at_level(logging.INFO, logger="chirii"):
            log_event("memory_added", user="asha@example.com", item_type="bill", item_count=1)
        assert "asha@example.com" not in caplog.text
        assert hash_user("asha@example.com") in caplog.text
        assert "item_type=bill" in caplog.text


class TestRepositoryHygiene:

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)

    def test_private_files_are_ignored(self):
        for path in (".env", ".env.local", "chirii/var/user_1.sqlite3", "chirii/var/user_1.faiss", "memory.db",
                     "models/qwen2.5-1.5b-instruct-q8_0.gguf", ".venv/pyvenv.cfg", "chirii/web/node_modules/x.js"):
            assert self.git("check-ignore", "-q", path).returncode == 0, f"{path} is not ignored"

    def test_env_example_is_not_ignored(self):
        assert self.git("check-ignore", "-q", ".env.example").returncode == 1

    def test_no_private_files_are_tracked(self):
        tracked = self.git("ls-files").stdout.splitlines()
        forbidden = [p for p in tracked if p == ".env" or p.startswith(".env.") and p != ".env.example"
                     or p.endswith((".db", ".sqlite", ".sqlite3", ".faiss", ".gguf")) or p.startswith("chirii/var/")]
        assert forbidden == []

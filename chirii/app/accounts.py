"""
S5: accounts, passwords and sessions.

- Accounts are created by the owner on this computer (no open sign-up during the pilot):
      python -m chirii.app.accounts add NAME       (asks for the password twice; it is never shown or logged)
      python -m chirii.app.accounts remove NAME    (removes the account; delete the data first in the app)
      python -m chirii.app.accounts list
- Passwords are stored only as Argon2id hashes (argon2-cffi defaults), in `accounts.enc`, which is itself encrypted
  with a key derived from CHIRII_DATA_KEY. At least 10 characters, and not the user name.
- 5 wrong passwords within 15 minutes lock that account for 15 minutes. An unknown name takes as long to refuse as a
  wrong password, so names can't be discovered by timing.
- A session is a random 256-bit token in an HttpOnly, SameSite=Strict cookie. The server keeps only its SHA-256
  hash, so the session table cannot be used to log in. Sessions end after 12 hours without use, after 7 days in any
  case, on logout, and when the app restarts.
- Every memory request is answered from the logged-in user's own store; there is no way to name another user's.
"""

import getpass
import hashlib
import json
import re
import secrets
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from chirii.app.logging_setup import log_event
from chirii.app.memory.vault import DataError, atomic_write, decrypt, derive_key, encrypt

ACCOUNTS_FILE = "accounts.enc"
ACCOUNTS_MAGIC = b"CHIRIIA\x00"
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
MIN_PASSWORD = 10
MAX_FAILURES, LOCK_SECONDS = 5, 15 * 60
IDLE_SECONDS, MAX_SESSION_SECONDS = 12 * 3600, 7 * 24 * 3600
COOKIE = "chirii_session"


class Locked(Exception):
    """Too many wrong passwords: try again later."""


class Accounts:

    def __init__(self, data_dir: Path, main_key: str):
        self.path = Path(data_dir) / ACCOUNTS_FILE
        self.key = derive_key(main_key, "accounts")
        self.hasher = PasswordHasher()
        self.lock = threading.RLock()
        self.failures: Dict[str, List[float]] = {}
        self._dummy = self.hasher.hash(secrets.token_urlsafe(16))   # for unknown names: same work as a real check

    # -------------------------------------------------------------- the encrypted file

    def _load(self) -> Dict[str, Dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(decrypt(self.key, self.path.read_bytes(), "accounts", ACCOUNTS_MAGIC))
        except DataError:
            raise DataError("accounts.enc could not be decrypted: wrong CHIRII_DATA_KEY, or the file was changed") from None

    def _save(self, users: Dict[str, Dict]) -> None:
        atomic_write(self.path, encrypt(self.key, json.dumps(users).encode("utf-8"), "accounts", ACCOUNTS_MAGIC))

    # -------------------------------------------------------------- managing accounts

    @staticmethod
    def clean_name(name: str) -> str:
        name = (name or "").strip().lower()
        if not NAME.match(name):
            raise ValueError("A user name is 3 to 32 characters: lower-case letters, digits, dot, dash or underscore.")
        return name

    def add(self, name: str, password: str) -> None:
        name = self.clean_name(name)
        if len(password) < MIN_PASSWORD or password.strip().lower() == name:
            raise ValueError(f"A password needs at least {MIN_PASSWORD} characters and must not be the user name.")
        with self.lock:
            users = self._load()
            if name in users:
                raise ValueError("That user name is taken.")
            users[name] = {"hash": self.hasher.hash(password), "created": datetime.now().isoformat(timespec="seconds"),
                           "consent": None}
            self._save(users)
        log_event("account_added", stage="S5")

    def remove(self, name: str) -> bool:
        with self.lock:
            users = self._load()
            if users.pop(self.clean_name(name), None) is None:
                return False
            self._save(users)
        log_event("account_removed", stage="S5")
        return True

    def names(self) -> List[str]:
        return sorted(self._load())

    def get(self, name: str) -> Optional[Dict]:
        return self._load().get(name)

    def set_consent(self, name: str, version: str) -> None:
        with self.lock:
            users = self._load()
            users[name]["consent"] = {"version": version, "on": datetime.now().isoformat(timespec="seconds")}
            self._save(users)

    # -------------------------------------------------------------- logging in

    def verify(self, name: str, password: str) -> Optional[str]:
        """The clean user name if the password is right, else None. Raises Locked after too many wrong tries."""
        try:
            name = self.clean_name(name)
        except ValueError:
            name = ""
        now = time.monotonic()
        with self.lock:
            recent = [t for t in self.failures.get(name, []) if now - t < LOCK_SECONDS]
            self.failures[name] = recent
            if len(recent) >= MAX_FAILURES:
                log_event("login_locked", stage="S5")
                raise Locked()
            user = self._load().get(name) if name else None
        try:
            self.hasher.verify(user["hash"] if user else self._dummy, password)
            ok = user is not None
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            ok = False
        if not ok:
            with self.lock:
                self.failures.setdefault(name, []).append(now)
            log_event("login_failed", stage="S5")
            return None
        with self.lock:
            self.failures.pop(name, None)
            if self.hasher.check_needs_rehash(user["hash"]):
                users = self._load()
                users[name]["hash"] = self.hasher.hash(password)
                self._save(users)
        log_event("login", stage="S5")
        return name


class Sessions:
    """Sessions held in memory, by the SHA-256 of their token."""

    def __init__(self):
        self.lock = threading.Lock()
        self.table: Dict[str, Tuple[str, float, float]] = {}     # hash → (user, created, last used)

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(self, user: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            self.table[self._hash(token)] = (user, now, now)
        return token

    def user(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        key, now = self._hash(token), time.time()
        with self.lock:
            found = self.table.get(key)
            if found is None:
                return None
            user, created, used = found
            if now - used > IDLE_SECONDS or now - created > MAX_SESSION_SECONDS:
                del self.table[key]
                return None
            self.table[key] = (user, created, now)
            return user

    def end(self, token: Optional[str]) -> None:
        if token:
            with self.lock:
                self.table.pop(self._hash(token), None)

    def end_all(self, user: str) -> None:
        with self.lock:
            for key in [k for k, v in self.table.items() if v[0] == user]:
                del self.table[key]


def main(argv: List[str]) -> None:
    from chirii.app.config import Settings
    settings = Settings()
    if not settings.data_key:
        raise SystemExit("CHIRII_DATA_KEY is not set. Create it with: python -m chirii.app.keys create")
    accounts = Accounts(settings.data_dir, settings.data_key)
    if len(argv) == 1 and argv[0] == "list":
        print("\n".join(accounts.names()) or "No accounts yet.")
    elif len(argv) == 2 and argv[0] == "add":
        first = getpass.getpass(f"Password for {argv[1]} (not shown): ")
        if getpass.getpass("Same password again: ") != first:
            raise SystemExit("The two passwords differ; nothing changed.")
        try:
            accounts.add(argv[1], first)
        except ValueError as bad:
            raise SystemExit(str(bad))
        print(f"Account {Accounts.clean_name(argv[1])} created. They will be asked to read and accept the consent note at first login.")
    elif len(argv) == 2 and argv[0] == "remove":
        print("Removed." if accounts.remove(argv[1]) else "No such account.")
    else:
        raise SystemExit(__doc__.split("\n\n")[1])


if __name__ == "__main__":
    main(sys.argv[1:])

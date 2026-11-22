"""
S5: encryption at rest (owner's choice, option C: whole-file encryption in the app).

Each user's memory is one SQLite database, kept in memory while the app runs and written to disk only as one
encrypted file, `memory.enc`:

    file   = MAGIC (8 bytes) | nonce (12 bytes) | AES-256-GCM ciphertext of the SQLite database
    key    = HKDF-SHA256(CHIRII_DATA_KEY, salt=b"chirii-data-v1", info=b"user:" + folder name)

- Each user has their own key, derived from the main key and their folder name, so one user's file cannot be
  opened with another user's key, and a file copied into another user's folder will not open (the folder name is
  also bound in as associated data).
- AES-GCM also detects any change to the file: a tampered or truncated file is refused, never half-read.
- Writes are atomic: the new file is written next to the old one, flushed to disk, and then swapped in with
  os.replace, so a crash during a write leaves the old file whole.
- A check value (`key.check` in the data folder) is made with the main key the first time. A wrong key is then
  refused at start-up, before any user file is touched, and a new key is never made when encrypted data exists.

If the main key is lost, the data cannot be recovered. That is the point: there is no back door.
"""

import base64
import hashlib
import os
import secrets
import threading
from pathlib import Path
from typing import Dict, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"CHIRII1\x00"
CHECK_MAGIC = b"CHIRIIK\x00"
CHECK_TEXT = b"chirii data key check v1"
SALT = b"chirii-data-v1"
DATA_FILE = "memory.enc"
CHECK_FILE = "key.check"
MIN_KEY_LENGTH = 32


class KeyError_(Exception):
    """The data key is missing, too short, or not the key the data was encrypted with."""


class DataError(Exception):
    """An encrypted file is damaged, changed, or belongs to another user."""


def new_key() -> str:
    """A fresh random main key: 32 random bytes, URL-safe base64 (43 characters)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def _main_key_bytes(main_key: str) -> bytes:
    if not main_key or len(main_key) < MIN_KEY_LENGTH:
        raise KeyError_(f"CHIRII_DATA_KEY must be a random string of at least {MIN_KEY_LENGTH} characters")
    return main_key.encode("utf-8")


def derive_key(main_key: str, purpose: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=SALT, info=purpose.encode("utf-8")).derive(_main_key_bytes(main_key))


def encrypt(key: bytes, plaintext: bytes, bound_to: str, magic: bytes = MAGIC) -> bytes:
    nonce = secrets.token_bytes(12)
    return magic + nonce + AESGCM(key).encrypt(nonce, plaintext, magic + bound_to.encode("utf-8"))


def decrypt(key: bytes, blob: bytes, bound_to: str, magic: bytes = MAGIC) -> bytes:
    if len(blob) < len(magic) + 12 + 16 or not blob.startswith(magic):
        raise DataError("not a Chirii encrypted file, or it is damaged")
    nonce, body = blob[len(magic):len(magic) + 12], blob[len(magic) + 12:]
    try:
        return AESGCM(key).decrypt(nonce, body, magic + bound_to.encode("utf-8"))
    except InvalidTag:
        raise DataError("the file could not be decrypted: wrong key, another user's file, or a changed file") from None


def atomic_write(path: Path, data: bytes) -> None:
    """Writes to a temporary file in the same folder, flushes it to disk, then swaps it in. A crash at any point
    leaves either the old file or the new one, never a mix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def check_key(data_dir: Path, main_key: str) -> str:
    """Start-up check. Returns "created" (first run: a check value was made), or "ok". Raises KeyError_ for a wrong
    key, or when encrypted data exists but its check value is missing (a new key would lock the old data out)."""
    data_dir = Path(data_dir)
    key = derive_key(main_key, "check")
    path = data_dir / CHECK_FILE
    if path.exists():
        try:
            if decrypt(key, path.read_bytes(), "check", CHECK_MAGIC) == CHECK_TEXT:
                return "ok"
        except DataError:
            pass
        raise KeyError_("CHIRII_DATA_KEY is not the key this data was encrypted with. The app will not start, so no "
                        "data is changed. Restore the right key from your password manager.")
    if any((data_dir / "users").glob(f"*/{DATA_FILE}")) if (data_dir / "users").exists() else False:
        raise KeyError_("Encrypted data exists but its key check file is missing. The app will not start with a key "
                        "it cannot verify.")
    atomic_write(path, encrypt(key, CHECK_TEXT, "check", CHECK_MAGIC))
    return "created"


_locks: Dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def lock_for(path: Path) -> threading.RLock:
    """One lock per user file, shared by every store object in this process."""
    name = str(Path(path).resolve())
    with _locks_guard:
        return _locks.setdefault(name, threading.RLock())


class Vault:
    """Reads and writes one user's encrypted file."""

    def __init__(self, folder: Path, main_key: str):
        self.folder = Path(folder)
        self.bound_to = "user:" + self.folder.name
        self.key = derive_key(main_key, self.bound_to)
        self.path = self.folder / DATA_FILE
        self.lock = lock_for(self.path)

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> Optional[bytes]:
        if not self.path.exists():
            return None
        return decrypt(self.key, self.path.read_bytes(), self.bound_to)

    def write(self, plaintext: bytes) -> None:
        atomic_write(self.path, encrypt(self.key, plaintext, self.bound_to))

    @staticmethod
    def fingerprint(blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()[:16]

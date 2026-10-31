"""
Phase 10 — Shared constants and hashing.

TEXT files are pinned with `sha256_text`, which normalises CRLF to LF before hashing, so a pin does not depend on
the platform that checked the file out. Binary files (token .bin, checkpoints) are pinned with `sha256_file`.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase9.train_v2 import project_path, sha256_file  # noqa: E402,F401

EOT = "<|endoftext|>"
SPECIAL_TOKENS = [EOT, "<|user|>", "<|assistant|>", "<|system|>", "<|calc|>", "<|/calc|>", "<|result|>", "<|/result|>", "<|pad|>"]

TEXT_DIR = "phase10/data/text"
TOKENS_DIR = "phase10/data/tokens"
TEXT_MANIFEST = "phase10/manifests/text_manifest.json"
TOKENS_MANIFEST = "phase10/manifests/tokens_manifest.json"
TOKENIZER_PATH = "phase10/tokenizer/tokenizer_v2.json"
TOKENIZER_STATS = "phase10/manifests/tokenizer_stats.json"

_CHUNK = 1024 * 1024


def sha256_text(path: str, chunk_size: int = _CHUNK) -> str:
    """SHA-256 with CRLF normalised to LF. Lone CRs are kept. A CRLF split across read chunks is still normalised."""
    h = hashlib.sha256()
    pending_cr = False
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            if pending_cr:
                chunk = b"\r" + chunk
                pending_cr = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_cr = True
            h.update(chunk.replace(b"\r\n", b"\n"))
    if pending_cr:
        h.update(b"\r")
    return h.hexdigest()


def read_text_lf(path: str) -> str:
    """Reads a UTF-8 text file with CRLF normalised to LF, so it matches its `sha256_text` pin."""
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n").decode("utf-8")


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)

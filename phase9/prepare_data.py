"""
Phase 9 — Converts existing .pt token files into raw uint16 .bin files that can be opened with np.memmap.
Source files are opened read-only with torch's mmap loader and are never modified.

Usage:
    python phase9/prepare_data.py
"""

import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase9.train_v2 import project_path, sha256_file

CONVERSIONS = [
    ("data/phase5d/train_tokens_v1024.pt", "phase9/data/train_v1024.bin"),
    ("data/phase5d/val_tokens_v1024.pt", "phase9/data/val_v1024.bin"),
    ("data/phase5/val_tokens_v1377.pt", "phase9/data/val_v1377.bin"),
]
CHUNK_TOKENS = 1_000_000


def convert(source: str, target: str) -> dict:
    tokens = torch.load(project_path(source), mmap=True, weights_only=True)
    if int(tokens.max()) > np.iinfo(np.uint16).max or int(tokens.min()) < 0:
        raise ValueError(f"{source} has token ids outside the uint16 range")

    os.makedirs(os.path.dirname(project_path(target)), exist_ok=True)
    with open(project_path(target), "wb") as f:
        for start in range(0, len(tokens), CHUNK_TOKENS):
            tokens[start:start + CHUNK_TOKENS].numpy().astype(np.uint16).tofile(f)

    written = np.memmap(project_path(target), dtype=np.uint16, mode="r")
    for start in range(0, len(tokens), CHUNK_TOKENS):
        if not np.array_equal(written[start:start + CHUNK_TOKENS], tokens[start:start + CHUNK_TOKENS].numpy()):
            raise ValueError(f"{target} does not match {source} at chunk starting {start}")

    return {
        "source": source,
        "target": target,
        "num_tokens": len(tokens),
        "source_sha256": sha256_file(project_path(source)),
        "sha256": sha256_file(project_path(target)),
    }


def main():
    manifest = [convert(source, target) for source, target in CONVERSIONS]
    manifest_path = project_path("phase9/data/manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(project_path("phase9/data/SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        for entry in manifest:
            f.write(f"{entry['sha256']}  {entry['target']}\n")
    for entry in manifest:
        print(f"{entry['target']}: {entry['num_tokens']:,} tokens, sha256 {entry['sha256']}", flush=True)


if __name__ == "__main__":
    main()

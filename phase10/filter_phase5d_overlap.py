"""
Phase 10A — Builds train_clean.bin: train.bin without documents that share text with the Phase 5D validation text.

Both come from FineWeb-Edu sample-10BT, so without this the model would train on the text that compares it with
Model C. A document is removed if any of its lines of 40+ characters is also a line in that text. train.bin is left
unchanged. The result is then rescanned to confirm no such line remains.

Usage:
    python phase10/filter_phase5d_overlap.py
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import EOT, TEXT_MANIFEST, TOKENS_DIR, TOKENS_MANIFEST, atomic_write_json, project_path, read_text_lf, sha256_file, sha256_text
from phase10.eval_v3 import OLD_TEXT, OLD_TEXT_SHA256

MIN_LINE_CHARS = 40
OUTPUT = f"{TOKENS_DIR}/train_clean.bin"
MANIFEST = "phase10/manifests/train_clean_manifest.json"


def line_key(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()


def old_text_keys(text: str) -> set:
    return {line_key(line) for line in text.split("\n") if len(line) >= MIN_LINE_CHARS}


def document_overlaps(document: str, keys: set) -> bool:
    return any(len(line) >= MIN_LINE_CHARS and line_key(line) in keys for line in document.split("\n"))


def keep_mask(tokens: np.ndarray, eot_id: int, overlapping: np.ndarray) -> np.ndarray:
    """Boolean mask over tokens that drops each flagged document together with its <|endoftext|>."""
    ends = np.flatnonzero(tokens == eot_id)
    if len(ends) != len(overlapping) or (len(ends) and ends[-1] != len(tokens) - 1):
        raise ValueError("Token file does not line up with its documents")
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    return np.repeat(~overlapping, lengths)


def main():
    if sha256_text(project_path(OLD_TEXT)) != OLD_TEXT_SHA256:
        raise ValueError("Phase 5D text differs from its pin")
    keys = old_text_keys(read_text_lf(project_path(OLD_TEXT)))
    with open(project_path(TEXT_MANIFEST), "r", encoding="utf-8") as f:
        text_shards = {s["file"]: s for s in json.load(f)["shards"] if s["split"] == "train"}
    with open(project_path(TOKENS_MANIFEST), "r", encoding="utf-8") as f:
        tokens_manifest = json.load(f)
    eot_id = tokens_manifest["eot_id"]

    removed_docs = kept_docs = removed_tokens = kept_tokens = removed_bytes = kept_bytes = remaining_lines = 0
    output_path = project_path(OUTPUT)
    with open(output_path + ".tmp", "wb") as out:
        for row in tokens_manifest["shards"]:
            if row["split"] != "train":
                continue
            with open(project_path(row["source"]), "rb") as f:
                documents = [d for d in f.read().decode("utf-8").split(EOT) if d]
            overlapping = np.array([document_overlaps(d, keys) for d in documents], dtype=bool)
            tokens = np.fromfile(project_path(row["file"]), dtype=np.uint16)
            mask = keep_mask(tokens, eot_id, overlapping)
            tokens[mask].tofile(out)
            removed_docs += int(overlapping.sum())
            kept_docs += int((~overlapping).sum())
            removed_tokens += int((~mask).sum())
            kept_tokens += int(mask.sum())
            for document, flagged in zip(documents, overlapping):
                size = len(document.encode("utf-8"))
                if flagged:
                    removed_bytes += size
                else:
                    kept_bytes += size
                    remaining_lines += document_overlaps(document, keys)
    os.replace(output_path + ".tmp", output_path)
    if remaining_lines:
        raise ValueError(f"{remaining_lines} kept documents still overlap")
    if kept_tokens * 2 != os.path.getsize(output_path):
        raise ValueError("Output size does not match the token count")

    manifest = {
        "file": OUTPUT, "source": tokens_manifest["joined"]["train"]["file"], "source_sha256": tokens_manifest["joined"]["train"]["sha256"],
        "excluded_text": OLD_TEXT, "excluded_text_sha256_lf": OLD_TEXT_SHA256, "min_line_chars": MIN_LINE_CHARS,
        "old_text_unique_long_lines": len(keys), "removed_documents": removed_docs, "removed_tokens": removed_tokens, "removed_text_bytes": removed_bytes,
        "documents": kept_docs, "tokens": kept_tokens, "text_bytes": kept_bytes, "sha256": sha256_file(output_path), "kept_documents_still_overlapping": remaining_lines,
    }
    atomic_write_json(project_path(MANIFEST), manifest)
    print(json.dumps(manifest, indent=2), flush=True)

    config_path = project_path("phase10/configs/foundation_v3.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["name"] = "Foundation v3 (Phase 5D overlap removed)"
    config["data"]["train_bin"] = OUTPUT
    config["data"]["train_sha256"] = manifest["sha256"]
    config["out_dir"] = "phase10/runs/foundation_v3_clean"
    atomic_write_json(project_path("phase10/configs/foundation_v3_clean.json"), config)


if __name__ == "__main__":
    main()

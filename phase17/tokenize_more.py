"""
Phase 17A — Tokenizes the new text shards with tokenizer v2 and builds the ~3B-token training file. No training.

Uses Phase 10's per-shard tokenizer and checks (phase10/tokenize_shards.py): each document encoded alone, followed by
<|endoftext|>, uint16, no control tokens, one <|endoftext|> per document.

Outputs (all hashes pinned in phase17/manifests/tokens_manifest.json):
  - phase17/data/tokens/{train,val}_NNNNN.bin   one per text shard
  - phase17/data/tokens/val_more.bin            the new val documents (Phase 10's val.bin stays the comparison set)
  - phase17/data/tokens/train_3b.bin            Phase 10's train_clean.bin (hash-checked) followed by every new train shard

Usage:
    RAYON_NUM_THREADS=8 python phase17/tokenize_more.py
"""

import json
import os
import shutil

import numpy as np
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase10.common import EOT, SPECIAL_TOKENS, TOKENIZER_PATH, atomic_write_json, project_path, sha256_file, sha256_text
from phase10.tokenize_shards import check_tokens, tokenize_shard
from phase17.collect_more import TEXT_MANIFEST

TOKENIZER_SHA256 = "332f9db4368e051179e70695ff6da320a8dc16f8f17934719cd58c8a00601b98"
PHASE10_CLEAN = {"file": "phase10/data/tokens/train_clean.bin", "sha256": "1906ef73e5041a5cd958f1ad2b5f14b70c09b23682b18ff63a993c593ab98458",
                 "tokens": 755_202_272, "documents": 659_133}
TOKENS_DIR = "phase17/data/tokens"
TOKENS_MANIFEST = "phase17/manifests/tokens_manifest.json"


def check_tokens_chunked(path, vocab_size, eot_id, documents, chunk_tokens=200_000_000):
    """Same checks as phase10's check_tokens, read in chunks so a multi-GB file never sits in memory at once."""
    array = np.memmap(path, dtype=np.uint16, mode="r")
    eots = 0
    for start in range(0, len(array), chunk_tokens):
        part = np.asarray(array[start:start + chunk_tokens])
        if int(part.max()) >= vocab_size:
            raise ValueError(f"{path}: token id {int(part.max())} is outside the vocabulary")
        if int(((part < len(SPECIAL_TOKENS)) & (part != eot_id)).sum()):
            raise ValueError(f"{path}: contains a control token")
        eots += int((part == eot_id).sum())
    if eots != documents:
        raise ValueError(f"{path}: <|endoftext|> count {eots} differs from document count {documents}")
    if len(array) and array[-1] != eot_id:
        raise ValueError(f"{path}: does not end with <|endoftext|>")


def join(paths, out_path):
    with open(out_path + ".tmp", "wb") as out:
        for path in paths:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, out, 16 * 1024 * 1024)
    os.replace(out_path + ".tmp", out_path)


def main():
    if sha256_text(project_path(TOKENIZER_PATH)) != TOKENIZER_SHA256:
        raise ValueError("tokenizer_v2.json differs from its pin")
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    eot_id = tokenizer.token_to_id(EOT)
    vocab_size = tokenizer.get_vocab_size()
    with open(project_path(TEXT_MANIFEST), "r", encoding="utf-8") as f:
        text_manifest = json.load(f)
    if text_manifest["status"] != "completed_target":
        raise ValueError(f"Text collection status is {text_manifest['status']}, not completed_target")
    clean_path = project_path(PHASE10_CLEAN["file"])
    if os.path.getsize(clean_path) != 2 * PHASE10_CLEAN["tokens"] or sha256_file(clean_path) != PHASE10_CLEAN["sha256"]:
        raise ValueError("phase10 train_clean.bin differs from its pin")

    os.makedirs(project_path(TOKENS_DIR), exist_ok=True)
    manifest_path = project_path(TOKENS_MANIFEST)
    done = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            done = {s["source"]: s for s in json.load(f)["shards"]}
    manifest = {"tokenizer": TOKENIZER_PATH, "tokenizer_sha256": TOKENIZER_SHA256, "dtype": "uint16", "vocab_size": vocab_size, "eot_id": eot_id,
                "threads": os.environ.get("RAYON_NUM_THREADS", "default"), "shards": [], "joined": {}}
    start = time.time()
    for shard in text_manifest["shards"]:
        if sha256_text(project_path(shard["file"])) != shard["sha256"]:
            raise ValueError(f"{shard['file']} differs from its pin")
        out_name = os.path.basename(shard["file"]).replace(".txt", ".bin")
        out_path = os.path.join(project_path(TOKENS_DIR), out_name)
        previous = done.get(shard["file"])
        if previous and previous["source_sha256"] == shard["sha256"] and os.path.exists(out_path) and sha256_file(out_path) == previous["sha256"]:
            manifest["shards"].append(previous)
            continue
        shard_start = time.time()
        tokens, documents = tokenize_shard(tokenizer, shard, out_path, eot_id)
        if documents != shard["documents"]:
            raise ValueError(f"{shard['file']}: read {documents} documents, manifest says {shard['documents']}")
        check_tokens(out_path, vocab_size, eot_id, documents)
        manifest["shards"].append({"source": shard["file"], "source_sha256": shard["sha256"], "file": f"{TOKENS_DIR}/{out_name}", "split": shard["split"],
                                   "documents": documents, "tokens": tokens, "text_bytes": shard["text_bytes"], "sha256": sha256_file(out_path)})
        atomic_write_json(manifest_path, manifest)
        elapsed = time.time() - shard_start
        print(f"{out_name}: {tokens:,} tokens, {shard['text_bytes']/tokens:.3f} bytes/token, {shard['text_bytes']/1e6/max(elapsed,1e-6):.1f} MB/s", flush=True)

    new_train = [s for s in manifest["shards"] if s["split"] == "train"]
    new_val = [s for s in manifest["shards"] if s["split"] == "val"]
    outputs = {
        "val_more": ([project_path(s["file"]) for s in new_val], sum(s["tokens"] for s in new_val), sum(s["documents"] for s in new_val), sum(s["text_bytes"] for s in new_val)),
        "train_3b": ([clean_path] + [project_path(s["file"]) for s in new_train], PHASE10_CLEAN["tokens"] + sum(s["tokens"] for s in new_train),
                     PHASE10_CLEAN["documents"] + sum(s["documents"] for s in new_train), None),
    }
    for name, (paths, tokens, documents, text_bytes) in outputs.items():
        out_path = os.path.join(project_path(TOKENS_DIR), f"{name}.bin")
        join(paths, out_path)
        if os.path.getsize(out_path) != 2 * tokens:
            raise ValueError(f"{name}.bin: size does not match token count")
        check_tokens_chunked(out_path, vocab_size, eot_id, documents)
        manifest["joined"][name] = {"file": f"{TOKENS_DIR}/{name}.bin", "tokens": tokens, "documents": documents, "sha256": sha256_file(out_path)}
        if text_bytes is not None:
            manifest["joined"][name]["text_bytes"] = text_bytes
        print(f"{name}.bin: {tokens:,} tokens, {documents:,} documents, sha256 {manifest['joined'][name]['sha256']}", flush=True)
    manifest["joined"]["train_3b"]["parts"] = {"phase10_train_clean": PHASE10_CLEAN, "phase17_new_train_tokens": sum(s["tokens"] for s in new_train),
                                               "phase17_new_train_documents": sum(s["documents"] for s in new_train)}
    manifest["seconds_this_run"] = round(time.time() - start, 1)
    atomic_write_json(manifest_path, manifest)
    print(f"Done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

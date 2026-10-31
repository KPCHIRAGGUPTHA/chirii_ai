"""
Phase 10A — Tokenizes every text shard to a uint16 .bin, then joins them into train.bin and val.bin and pins the hashes.

Each document is encoded alone and followed by <|endoftext|>. Finished shards are skipped on restart.

Usage:
    RAYON_NUM_THREADS=8 python phase10/tokenize_shards.py
"""

import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase10.common import EOT, SPECIAL_TOKENS, TEXT_MANIFEST, TOKENIZER_PATH, TOKENS_DIR, TOKENS_MANIFEST, atomic_write_json, project_path, sha256_file, sha256_text
from phase10.tokenizer_v2 import iter_documents

BATCH_DOCS = 2000
VAL_SUBSET_TOKENS = 256_000


def tokenize_shard(tokenizer, shard, out_path, eot_id):
    tmp_path = out_path + ".tmp"
    tokens = documents = 0
    batch = []

    def flush(f):
        nonlocal tokens, documents, batch
        for encoding in tokenizer.encode_batch_fast(batch, add_special_tokens=False):
            ids = np.array(encoding.ids + [eot_id], dtype=np.uint16)
            ids.tofile(f)
            tokens += len(ids)
            documents += 1
        batch = []

    with open(tmp_path, "wb") as f:
        for doc in iter_documents(project_path(shard["file"])):
            batch.append(doc.decode("utf-8"))
            if len(batch) >= BATCH_DOCS:
                flush(f)
        if batch:
            flush(f)
    os.replace(tmp_path, out_path)
    return tokens, documents


def check_tokens(path, vocab_size, eot_id, documents):
    """Special tokens other than <|endoftext|> must never come from web text."""
    array = np.memmap(path, dtype=np.uint16, mode="r")
    if int(array.max()) >= vocab_size:
        raise ValueError(f"{path}: token id {int(array.max())} is outside the vocabulary")
    if int(((array < len(SPECIAL_TOKENS)) & (array != eot_id)).sum()) != 0:
        raise ValueError(f"{path}: contains a control token")
    if int((array == eot_id).sum()) != documents:
        raise ValueError(f"{path}: <|endoftext|> count differs from document count {documents}")


def main():
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    eot_id = tokenizer.token_to_id(EOT)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError("Vocabulary does not fit in uint16")
    with open(project_path(TEXT_MANIFEST), "r", encoding="utf-8") as f:
        text_manifest = json.load(f)
    if text_manifest["status"] == "running":
        print("Warning: text collection has not finished; tokenizing the shards finished so far", flush=True)

    out_dir = project_path(TOKENS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = project_path(TOKENS_MANIFEST)
    tokenizer_sha = sha256_text(project_path(TOKENIZER_PATH))
    done = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        if previous["tokenizer_sha256"] == tokenizer_sha:
            done = {s["source"]: s for s in previous["shards"]}

    manifest = {
        "tokenizer": TOKENIZER_PATH, "tokenizer_sha256": tokenizer_sha, "dtype": "uint16", "vocab_size": vocab_size,
        "eot_id": eot_id, "threads": os.environ.get("RAYON_NUM_THREADS", "default"), "shards": [], "joined": {},
    }
    start = time.time()
    for shard in text_manifest["shards"]:
        out_name = os.path.basename(shard["file"]).replace(".txt", ".bin")
        out_path = os.path.join(out_dir, out_name)
        previous_entry = done.get(shard["file"])
        if previous_entry and previous_entry["source_sha256"] == shard["sha256"] and os.path.exists(out_path):
            manifest["shards"].append(previous_entry)
            continue
        shard_start = time.time()
        tokens, documents = tokenize_shard(tokenizer, shard, out_path, eot_id)
        if documents != shard["documents"]:
            raise ValueError(f"{shard['file']}: read {documents} documents, manifest says {shard['documents']}")
        check_tokens(out_path, vocab_size, eot_id, documents)
        manifest["shards"].append({
            "source": shard["file"], "source_sha256": shard["sha256"], "file": f"{TOKENS_DIR}/{out_name}", "split": shard["split"],
            "documents": documents, "tokens": tokens, "text_bytes": shard["text_bytes"], "sha256": sha256_file(out_path),
        })
        atomic_write_json(manifest_path, manifest)
        elapsed = time.time() - shard_start
        print(f"{out_name}: {tokens:,} tokens, {shard['text_bytes']/tokens:.3f} bytes/token, {shard['text_bytes']/1e6/max(elapsed,1e-6):.1f} MB/s", flush=True)

    for split in ("train", "val"):
        joined = os.path.join(out_dir, f"{split}.bin")
        rows = [s for s in manifest["shards"] if s["split"] == split]
        with open(joined + ".tmp", "wb") as out:
            for row in rows:
                with open(project_path(row["file"]), "rb") as f:
                    shutil.copyfileobj(f, out, 16 * 1024 * 1024)
        os.replace(joined + ".tmp", joined)
        entry = {
            "file": f"{TOKENS_DIR}/{split}.bin", "tokens": sum(r["tokens"] for r in rows), "documents": sum(r["documents"] for r in rows),
            "text_bytes": sum(r["text_bytes"] for r in rows), "sha256": sha256_file(joined),
        }
        if entry["tokens"] * 2 != os.path.getsize(joined):
            raise ValueError(f"{joined}: size does not match token count")
        manifest["joined"][split] = entry
        print(f"{split}.bin: {entry['tokens']:,} tokens, {entry['documents']:,} documents, sha256 {entry['sha256']}", flush=True)

    # Fixed validation subset for evaluation during training: the first tokens of val.bin
    array = np.memmap(os.path.join(out_dir, "val.bin"), dtype=np.uint16, mode="r")
    import hashlib
    manifest["val_subset"] = {"tokens": VAL_SUBSET_TOKENS, "sha256": hashlib.sha256(array[:VAL_SUBSET_TOKENS + 1].tobytes()).hexdigest()}
    manifest["seconds_this_run"] = time.time() - start
    atomic_write_json(manifest_path, manifest)
    print(f"Done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

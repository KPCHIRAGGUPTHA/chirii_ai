"""
Phase 17A — Collects about 9.4 GB more FineWeb-Edu (sample-10BT) text, so that with Phase 10's clean train split the
pretraining data reaches about 3B tokens with tokenizer v2. No training. Phase 10's data and manifests are not touched.

Same rules as Phase 10 (phase10/collect_data.py), imported rather than copied:
  - same dataset, config and revision (87f09149…, pinned, not "latest")
  - same deterministic split: SHA-256 of the document bytes, first 8 bytes mod 1,000, below 5 goes to val
  - same filters: empty documents and documents containing a literal special token are dropped
Excluded, so no document is used twice and the Model C comparison stays clean:
  - the first 669,469 documents of the stream, which Phase 10 already collected (skipped by position), and
    any later document whose exact text equals one of those (checked by SHA-256 of the text)
  - any document with a 40+ character line that is also a line of the Phase 5D validation text (the rule of
    phase10/filter_phase5d_overlap.py), so the Phase 5D comparison text is never trained on

Each chunk is finalised as a train/val shard pair and recorded in the manifest, so the run can stop and resume.
To stop cleanly at the next chunk: touch phase17/data/STOP

Usage:
    HF_HOME=~/chirag_work/hf_cache python phase17/collect_more.py --target-bytes 9400000000
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.collect_data import CONFIG, DATASET, VAL_PER_MILLE, split_for_document
from phase10.common import EOT, SPECIAL_TOKENS, TEXT_MANIFEST as PHASE10_TEXT_MANIFEST, atomic_write_json, project_path, read_text_lf, sha256_text
from phase10.eval_v3 import OLD_TEXT, OLD_TEXT_SHA256
from phase10.filter_phase5d_overlap import MIN_LINE_CHARS, document_overlaps, old_text_keys
from phase10.tokenizer_v2 import iter_documents

REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"  # the revision Phase 10 used
PHASE10_DOCS_READ = 669_469
TEXT_DIR = "phase17/data/text"
TEXT_MANIFEST = "phase17/manifests/text_manifest.json"
STOP_FILE = "phase17/data/STOP"


def doc_key(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()[:16]


def phase10_document_keys() -> Set[bytes]:
    """Keys of every document Phase 10 collected, read from its pinned text shards (hashes re-checked first)."""
    with open(project_path(PHASE10_TEXT_MANIFEST), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["revision"] != REVISION or manifest["docs_read"] != PHASE10_DOCS_READ:
        raise ValueError("Phase 10 manifest does not match the expected revision or document count")
    keys: Set[bytes] = set()
    for shard in manifest["shards"]:
        if sha256_text(project_path(shard["file"])) != shard["sha256"]:
            raise ValueError(f"{shard['file']} differs from its pin")
        for doc in iter_documents(project_path(shard["file"])):
            keys.add(doc_key(doc))
    if len(keys) > manifest["docs_kept"]:
        raise ValueError("More keys than Phase 10 documents")
    return keys


def load_state():
    path = project_path(TEXT_MANIFEST)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state["revision"] != REVISION:
            raise ValueError(f"Manifest revision {state['revision']} differs from {REVISION}")
        return state
    return {
        "dataset": DATASET, "config": CONFIG, "revision": REVISION, "val_per_mille": VAL_PER_MILLE,
        "hash_note": "sha256 is over text with CRLF normalised to LF (phase10.common.sha256_text)", "document_terminator": EOT,
        "stream_start": PHASE10_DOCS_READ, "excluded_text": OLD_TEXT, "excluded_text_sha256_lf": OLD_TEXT_SHA256, "min_line_chars": MIN_LINE_CHARS,
        "docs_read": PHASE10_DOCS_READ, "docs_kept": 0, "text_bytes": 0, "skipped_special_token": 0, "skipped_empty_or_bad": 0,
        "skipped_phase10_duplicate": 0, "skipped_phase5d_overlap": 0, "skipped_duplicate_within_phase17": 0, "status": "running", "shards": [],
    }


def collect_once(state, args, phase10_keys: Set[bytes], old_keys: set, seen: Set[bytes]) -> str:
    from datasets import load_dataset

    os.makedirs(project_path(TEXT_DIR), exist_ok=True)
    stream = load_dataset(DATASET, name=CONFIG, split="train", streaming=True, revision=REVISION)
    print(f"Skipping the first {state['docs_read']:,} documents of the stream", flush=True)
    stream = stream.skip(state["docs_read"])

    terminator = EOT.encode("utf-8")
    c = {"index": len(state["shards"]) // 2, "docs_read": state["docs_read"]}

    def open_chunk():
        c["tmp_paths"] = {s: os.path.join(project_path(TEXT_DIR), f"{s}_{c['index']:05d}.tmp") for s in ("train", "val")}
        c["files"] = {s: open(p, "wb") for s, p in c["tmp_paths"].items()}
        c["counts"] = {s: {"docs": 0, "bytes": 0} for s in ("train", "val")}
        c["skipped"] = {"special": 0, "bad": 0, "phase10": 0, "phase5d": 0, "dup17": 0}
        c["keys"] = set()
        c["start"] = time.time()

    def finalise_chunk():
        """Hashes and renames the shard pair, then records it. Unfinished chunks are never in the manifest."""
        for f in c["files"].values():
            f.close()
        counts, skipped = c["counts"], c["skipped"]
        chunk_bytes = counts["train"]["bytes"] + counts["val"]["bytes"]
        for split in ("train", "val"):
            name = f"{split}_{c['index']:05d}.txt"
            final_path = os.path.join(project_path(TEXT_DIR), name)
            os.replace(c["tmp_paths"][split], final_path)
            state["shards"].append({"file": f"{TEXT_DIR}/{name}", "split": split, "documents": counts[split]["docs"],
                                    "text_bytes": counts[split]["bytes"], "file_bytes": os.path.getsize(final_path), "sha256": sha256_text(final_path)})
        seen.update(c["keys"])
        state["docs_read"] = c["docs_read"]
        state["docs_kept"] += counts["train"]["docs"] + counts["val"]["docs"]
        state["text_bytes"] += chunk_bytes
        for name, field in (("special", "skipped_special_token"), ("bad", "skipped_empty_or_bad"), ("phase10", "skipped_phase10_duplicate"),
                            ("phase5d", "skipped_phase5d_overlap"), ("dup17", "skipped_duplicate_within_phase17")):
            state[field] += skipped[name]
        for split in ("train", "val"):
            rows = [s for s in state["shards"] if s["split"] == split]
            state[f"{split}_documents"] = sum(s["documents"] for s in rows)
            state[f"{split}_text_bytes"] = sum(s["text_bytes"] for s in rows)
        atomic_write_json(project_path(TEXT_MANIFEST), state)
        rate = chunk_bytes / 1e6 / max(time.time() - c["start"], 1e-6)
        print(f"chunk {c['index']}: total {state['text_bytes']/1e9:.3f} GB, {state['docs_kept']:,} docs, {rate:.2f} MB/s, "
              f"skipped so far: phase10 dup {state['skipped_phase10_duplicate']}, phase5d {state['skipped_phase5d_overlap']}, "
              f"special {state['skipped_special_token']}, dup {state['skipped_duplicate_within_phase17']}", flush=True)

    open_chunk()
    try:
        for row in stream:
            c["docs_read"] += 1
            text = row["text"]
            try:
                data = text.encode("utf-8") if text else b""
            except UnicodeEncodeError:
                data = b""
            skipped, counts = c["skipped"], c["counts"]
            if not data:
                skipped["bad"] += 1
                continue
            if any(token in text for token in SPECIAL_TOKENS):
                skipped["special"] += 1
                continue
            key = doc_key(data)
            if key in phase10_keys:
                skipped["phase10"] += 1
                continue
            if key in seen or key in c["keys"]:
                skipped["dup17"] += 1
                continue
            if document_overlaps(text.replace("\r\n", "\n"), old_keys):
                skipped["phase5d"] += 1
                continue
            c["keys"].add(key)
            split = split_for_document(data)
            c["files"][split].write(data + terminator)
            counts[split]["docs"] += 1
            counts[split]["bytes"] += len(data)

            chunk_bytes = counts["train"]["bytes"] + counts["val"]["bytes"]
            done = state["text_bytes"] + chunk_bytes >= args.target_bytes
            if chunk_bytes >= args.chunk_bytes or done:
                finalise_chunk()
                if done:
                    return "completed_target"
                if os.path.exists(project_path(STOP_FILE)):
                    return "stopped_by_request"
                free_gb = shutil.disk_usage(project_path(TEXT_DIR)).free / 1e9
                if free_gb < args.min_free_gb:
                    return f"stopped_low_disk_{free_gb:.0f}GB"
                c["index"] += 1
                open_chunk()
        if c["counts"]["train"]["docs"] + c["counts"]["val"]["docs"]:
            finalise_chunk()  # the stream ended part-way through a chunk: keep what was collected
        return "completed_stream_ended"
    finally:
        for f in c["files"].values():
            f.close()


def main():
    parser = argparse.ArgumentParser(description="Collect more FineWeb-Edu text for Phase 17A")
    parser.add_argument("--target-bytes", type=int, default=9_400_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=512_000_000)
    parser.add_argument("--min-free-gb", type=float, default=50)
    parser.add_argument("--max-retries", type=int, default=20)
    args = parser.parse_args()

    state = load_state()
    if state["status"] == "completed_target" and state["text_bytes"] >= args.target_bytes:
        print(f"Already complete: {state['text_bytes']:,} bytes, {state['docs_kept']:,} documents; nothing to do", flush=True)
        return

    if sha256_text(project_path(OLD_TEXT)) != OLD_TEXT_SHA256:
        raise ValueError("Phase 5D validation text differs from its pin")
    old_keys = old_text_keys(read_text_lf(project_path(OLD_TEXT)))
    start = time.time()
    phase10_keys = phase10_document_keys()
    print(f"{DATASET} {CONFIG} revision {REVISION}; {len(phase10_keys):,} Phase 10 document keys and "
          f"{len(old_keys):,} Phase 5D line keys loaded in {time.time() - start:.0f}s", flush=True)

    for attempt in range(args.max_retries + 1):
        state = load_state()
        seen = set()
        for shard in state["shards"]:  # documents kept in earlier chunks, so a resume still catches repeats
            for doc in iter_documents(project_path(shard["file"])):
                seen.add(doc_key(doc))
        try:
            status = collect_once(state, args, phase10_keys, old_keys, seen)
            break
        except Exception as error:  # network failures: discard the partial chunk and resume from the last finished one
            print(f"Attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(600, 30 * (attempt + 1)))
    else:
        status = "failed_retries_exhausted"
    state = load_state()
    state["status"] = status
    atomic_write_json(project_path(TEXT_MANIFEST), state)
    print(f"Finished: {status}, {state['text_bytes']:,} bytes, {state['docs_kept']:,} documents", flush=True)


if __name__ == "__main__":
    main()

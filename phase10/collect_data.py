"""
Phase 10A — Streams FineWeb-Edu (sample-10BT) into raw text shards, split deterministically by document hash.

Each chunk of the stream is finalised as a train/val shard pair, and the manifest is rewritten after every chunk,
so finished shards stay usable if the run stops. Restarting resumes after the last finished chunk.
Documents end with <|endoftext|>. To stop cleanly at the next chunk: touch phase10/data/STOP

Usage:
    python phase10/collect_data.py --target-bytes 3200000000
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import EOT, SPECIAL_TOKENS, TEXT_DIR, TEXT_MANIFEST, atomic_write_json, project_path, sha256_text

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
VAL_PER_MILLE = 5  # 0.5%


def split_for_document(text_bytes: bytes) -> str:
    """Deterministic: the same text always lands in the same split, so exact duplicates cannot leak across splits."""
    bucket = int.from_bytes(hashlib.sha256(text_bytes).digest()[:8], "big") % 1000
    return "val" if bucket < VAL_PER_MILLE else "train"


def load_state(revision):
    path = project_path(TEXT_MANIFEST)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state["revision"] != revision:
            raise ValueError(f"Manifest revision {state['revision']} differs from {revision}")
        return state
    return {
        "dataset": DATASET, "config": CONFIG, "revision": revision, "val_per_mille": VAL_PER_MILLE,
        "hash_note": "sha256 is over text with CRLF normalised to LF (phase10.common.sha256_text)",
        "document_terminator": EOT, "docs_read": 0, "docs_kept": 0, "text_bytes": 0, "skipped_special_token": 0,
        "skipped_empty_or_bad": 0, "status": "running", "shards": [],
    }


def finalise_chunk(state, index, tmp_paths, counts, docs_read_total, skipped):
    """Hashes and renames the chunk's shard pair, then records it. Unfinished chunks are never in the manifest."""
    for split in ("train", "val"):
        name = f"{split}_{index:05d}.txt"
        final_path = os.path.join(project_path(TEXT_DIR), name)
        os.replace(tmp_paths[split], final_path)
        state["shards"].append({
            "file": f"{TEXT_DIR}/{name}", "split": split, "documents": counts[split]["docs"],
            "text_bytes": counts[split]["bytes"], "file_bytes": os.path.getsize(final_path), "sha256": sha256_text(final_path),
        })
    state["docs_read"] = docs_read_total
    state["docs_kept"] += counts["train"]["docs"] + counts["val"]["docs"]
    state["text_bytes"] += counts["train"]["bytes"] + counts["val"]["bytes"]
    state["skipped_special_token"] += skipped["special"]
    state["skipped_empty_or_bad"] += skipped["bad"]
    for split in ("train", "val"):
        rows = [s for s in state["shards"] if s["split"] == split]
        state[f"{split}_documents"] = sum(s["documents"] for s in rows)
        state[f"{split}_text_bytes"] = sum(s["text_bytes"] for s in rows)
    atomic_write_json(project_path(TEXT_MANIFEST), state)


def collect_once(state, args):
    """Runs the stream from the last finished chunk until the target or a stop condition. Returns the final status."""
    from datasets import load_dataset

    os.makedirs(project_path(TEXT_DIR), exist_ok=True)
    stream = load_dataset(DATASET, name=CONFIG, split="train", streaming=True, revision=state["revision"])
    if state["docs_read"]:
        print(f"Skipping {state['docs_read']:,} documents already collected", flush=True)
        stream = stream.skip(state["docs_read"])

    terminator = EOT.encode("utf-8")
    index = len(state["shards"]) // 2
    tmp_paths = {s: os.path.join(project_path(TEXT_DIR), f"{s}_{index:05d}.tmp") for s in ("train", "val")}
    files = {s: open(p, "wb") for s, p in tmp_paths.items()}
    counts = {s: {"docs": 0, "bytes": 0} for s in ("train", "val")}
    skipped = {"special": 0, "bad": 0}
    docs_read = state["docs_read"]
    start = time.time()
    try:
        for row in stream:
            docs_read += 1
            text = row["text"]
            try:
                data = text.encode("utf-8") if text else b""
            except UnicodeEncodeError:
                data = b""
            if not data:
                skipped["bad"] += 1
                continue
            # A literal special token in web text would be parsed as control syntax, so those documents are dropped
            if any(token in text for token in SPECIAL_TOKENS):
                skipped["special"] += 1
                continue
            split = split_for_document(data)
            files[split].write(data + terminator)
            counts[split]["docs"] += 1
            counts[split]["bytes"] += len(data)

            chunk_bytes = counts["train"]["bytes"] + counts["val"]["bytes"]
            done = state["text_bytes"] + chunk_bytes >= args.target_bytes
            if chunk_bytes >= args.chunk_bytes or done:
                for f in files.values():
                    f.close()
                finalise_chunk(state, index, tmp_paths, counts, docs_read, skipped)
                rate = chunk_bytes / 1e6 / max(time.time() - start, 1e-6)
                print(f"chunk {index}: total {state['text_bytes']/1e9:.3f} GB, {state['docs_kept']:,} docs, {rate:.2f} MB/s", flush=True)
                if done:
                    return "completed_target"
                if os.path.exists(project_path("phase10/data/STOP")):
                    return "stopped_by_request"
                free_gb = shutil.disk_usage(project_path(TEXT_DIR)).free / 1e9
                if free_gb < args.min_free_gb:
                    return f"stopped_low_disk_{free_gb:.0f}GB"
                index += 1
                tmp_paths = {s: os.path.join(project_path(TEXT_DIR), f"{s}_{index:05d}.tmp") for s in ("train", "val")}
                files = {s: open(p, "wb") for s, p in tmp_paths.items()}
                counts = {s: {"docs": 0, "bytes": 0} for s in ("train", "val")}
                skipped = {"special": 0, "bad": 0}
                start = time.time()
    finally:
        for f in files.values():
            f.close()
    return "completed_stream_ended"


def main():
    parser = argparse.ArgumentParser(description="Collect FineWeb-Edu text shards")
    parser.add_argument("--target-bytes", type=int, default=3_200_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=256_000_000)
    parser.add_argument("--min-free-gb", type=float, default=50)
    parser.add_argument("--max-retries", type=int, default=20)
    args = parser.parse_args()

    from huggingface_hub import HfApi
    revision = HfApi().dataset_info(DATASET).sha
    print(f"{DATASET} {CONFIG} revision {revision}", flush=True)

    for attempt in range(args.max_retries + 1):
        state = load_state(revision)
        try:
            status = collect_once(state, args)
            break
        except Exception as error:  # Network failures: discard the partial chunk and resume from the last finished one
            print(f"Attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(600, 30 * (attempt + 1)))
    else:
        status = "failed_retries_exhausted"
    state = load_state(revision)
    state["status"] = status
    atomic_write_json(project_path(TEXT_MANIFEST), state)
    print(f"Finished: {status}, {state['text_bytes']:,} bytes, {state['docs_kept']:,} documents", flush=True)


if __name__ == "__main__":
    main()

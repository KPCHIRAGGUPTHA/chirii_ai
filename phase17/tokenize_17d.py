"""
Phase 17D — Tokenizes the filtered new text (tokenizer v2), trims to 70:30 FineWeb-Edu : FineWeb by tokens (whole
documents, at most 8.0B train tokens in total), and mixes at document level: all selected train documents are written
to train_17d.bin in a seeded random order (numpy default_rng(17)). New val documents go to val_17d.bin (record only;
training still evaluates on the pinned Phase 10 validation subset). All hashes are pinned in
phase17/manifests17d/tokens_manifest.json.

Usage:  RAYON_NUM_THREADS=8 python phase17/tokenize_17d.py
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tokenizers import Tokenizer  # noqa: E402

from phase10.common import EOT, SPECIAL_TOKENS, TOKENIZER_PATH, atomic_write_json, project_path, sha256_file, sha256_text  # noqa: E402
from phase10.tokenize_shards import check_tokens, tokenize_shard  # noqa: E402
from phase17.tokenize_more import TOKENIZER_SHA256, check_tokens_chunked  # noqa: E402

FILTER_MANIFEST = "phase17/manifests17d/filter_manifest.json"
OUT_DIR = "phase17/data17d/tokens"
OUT_MANIFEST = "phase17/manifests17d/tokens_manifest.json"
MAX_TOTAL = 8_000_000_000
SHARE = {"edu": 0.7, "web": 0.3}
SEED = 17


def doc_spans(path, eot_id):
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    ends = np.flatnonzero(tokens == eot_id) + 1
    starts = np.concatenate(([0], ends[:-1]))
    return starts, ends


def main():
    if sha256_text(project_path(TOKENIZER_PATH)) != TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from its pin")
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    eot_id, vocab = tokenizer.token_to_id(EOT), tokenizer.get_vocab_size()
    fm = json.load(open(project_path(FILTER_MANIFEST)))
    os.makedirs(project_path(OUT_DIR), exist_ok=True)
    manifest = {"tokenizer_sha256": TOKENIZER_SHA256, "eot_id": eot_id, "shards": [], "joined": {}}
    t0 = time.time()
    for s in fm["shards"]:
        if sha256_text(project_path(s["file"])) != s["sha256"]:
            raise ValueError(f"{s['file']} differs from its pin")
        out = project_path(f"{OUT_DIR}/{s['source']}_{os.path.basename(s['file']).replace('.txt', '.bin')}")
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            tokens, docs = tokenize_shard(tokenizer, {"file": s["file"]}, out, eot_id)
            if docs != s["documents"]:
                raise ValueError(f"{s['file']}: {docs} documents, manifest says {s['documents']}")
            check_tokens(out, vocab, eot_id, docs)
        tokens = os.path.getsize(out) // 2
        manifest["shards"].append({"source": s["source"], "split": s["split"], "source_file": s["file"], "file": os.path.relpath(out, ROOT),
                                   "documents": s["documents"], "tokens": tokens, "sha256": sha256_file(out)})
        print(f"{os.path.basename(out)}: {tokens:,} tokens", flush=True)

    # 70:30 by tokens, at most 8.0B, whole documents taken in shard order
    train = {src: [x for x in manifest["shards"] if x["source"] == src and x["split"] == "train"] for src in SHARE}
    have = {src: sum(x["tokens"] for x in rows) for src, rows in train.items()}
    total = min(MAX_TOTAL, have["edu"] / SHARE["edu"], have["web"] / SHARE["web"])
    target = {src: int(total * SHARE[src]) for src in SHARE}
    spans = []  # (shard index, start, end) of every selected document
    files = []
    used = {}
    for src in SHARE:
        got = 0
        for row in train[src]:
            if got >= target[src]:
                break
            starts, ends = doc_spans(project_path(row["file"]), eot_id)
            lengths = ends - starts
            take = int(np.searchsorted(np.cumsum(lengths), target[src] - got, side="right"))
            take = min(len(starts), max(take, 1))
            idx = len(files)
            files.append(np.memmap(project_path(row["file"]), dtype=np.uint16, mode="r"))
            spans.extend((idx, int(a), int(b)) for a, b in zip(starts[:take], ends[:take]))
            got += int(lengths[:take].sum())
        used[src] = got
    order = np.random.default_rng(SEED).permutation(len(spans))
    train_path = project_path(f"{OUT_DIR}/train_17d.bin")
    with open(train_path + ".tmp", "wb") as f:
        buf, size = [], 0
        for k in order:
            i, a, b = spans[k]
            buf.append(np.asarray(files[i][a:b]))
            size += b - a
            if size >= 16_000_000:
                np.concatenate(buf).tofile(f)
                buf, size = [], 0
        if buf:
            np.concatenate(buf).tofile(f)
    os.replace(train_path + ".tmp", train_path)
    n_train = os.path.getsize(train_path) // 2
    if n_train != sum(used.values()):
        raise ValueError("mixed file size does not match the selected documents")
    check_tokens_chunked(train_path, vocab, eot_id, len(spans))
    manifest["joined"]["train_17d"] = {"file": f"{OUT_DIR}/train_17d.bin", "tokens": n_train, "documents": len(spans), "sha256": sha256_file(train_path),
                                       "tokens_by_source": used, "share_by_source": {k: round(v / n_train, 4) for k, v in used.items()},
                                       "available_by_source": have, "mix": f"document level, numpy default_rng({SEED}) permutation"}
    val_rows = [x for x in manifest["shards"] if x["split"] == "val"]
    val_path = project_path(f"{OUT_DIR}/val_17d.bin")
    with open(val_path + ".tmp", "wb") as f:
        for x in val_rows:
            np.memmap(project_path(x["file"]), dtype=np.uint16, mode="r").tofile(f)
    os.replace(val_path + ".tmp", val_path)
    manifest["joined"]["val_17d"] = {"file": f"{OUT_DIR}/val_17d.bin", "tokens": os.path.getsize(val_path) // 2,
                                     "documents": sum(x["documents"] for x in val_rows), "sha256": sha256_file(val_path)}
    manifest["seconds"] = round(time.time() - t0)
    atomic_write_json(project_path(OUT_MANIFEST), manifest)
    print(json.dumps(manifest["joined"], indent=1), flush=True)


if __name__ == "__main__":
    main()

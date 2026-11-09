"""
Phase 17D — Benchmark decontamination and cross-source de-duplication of the newly collected text.

For every document (both sources, train and val shards):
  - removed if it shares any 13-word sequence with any of the 12 Phase 18 benchmark test sets (the exact evaluated text
    plus its answer, built by phase18/contamination.py; same hashing)
  - removed if its exact text was already kept from an earlier shard (FineWeb-Edu shards are read first, then FineWeb)
Writes filtered shards under phase17/data17d/<source>/filtered/ and phase17/manifests17d/filter_manifest.json with
hashes and removal counts per source and per benchmark.

Usage (needs PYTHONHASHSEED=0 so every worker hashes words identically):
    PYTHONHASHSEED=0 PYTHONPATH=~/chirag_work/pylibs/phase18 python phase17/filter_17d.py --workers 8
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from phase10.common import EOT, atomic_write_json, project_path, sha256_text  # noqa: E402
from phase17.collect_more import doc_key  # noqa: E402

SOURCES = ["edu", "web"]
OUT_MANIFEST = "phase17/manifests17d/filter_manifest.json"
_BENCH = None


def _init(bench):
    global _BENCH
    _BENCH = bench


def scan(path):
    """Per document: matched benchmark 13-gram hashes (empty if clean) and the document key."""
    sys.path.insert(0, os.path.join(ROOT, "phase18"))
    import contamination as c18
    docs = [d for d in open(path, "rb").read().split(EOT.encode()) if d]
    out = []
    for d in docs:
        h = c18.ngram_hashes(d.decode("utf-8"))
        hits = []
        if len(h):
            pos = np.searchsorted(_BENCH, h)
            pos[pos == len(_BENCH)] = 0
            m = _BENCH[pos] == h
            if m.any():
                hits = np.unique(h[m]).tolist()
        out.append((hits, doc_key(d)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0")
    sys.path.insert(0, os.path.join(ROOT, "phase18"))
    import contamination as c18

    t0 = time.time()
    items = c18.benchmark_items()
    owner = defaultdict(set)
    for task, texts in items.items():
        for text in texts:
            for h in c18.ngram_hashes(text).tolist():
                owner[h].add(task)
    bench = np.array(sorted(owner), dtype=np.uint64)
    print(f"{len(bench):,} benchmark 13-grams from {sum(len(v) for v in items.values()):,} items ({time.time() - t0:.0f}s)", flush=True)

    shards = []
    for src in SOURCES:
        m = json.load(open(project_path(f"phase17/manifests17d/{src}_text_manifest.json")))
        if m["status"] != "completed_target":
            raise SystemExit(f"{src} collection status is {m['status']}")
        for s in m["shards"]:
            if sha256_text(project_path(s["file"])) != s["sha256"]:
                raise ValueError(f"{s['file']} differs from its pin")
            shards.append((src, s))

    seen = set()
    report = {src: {"documents_in": 0, "kept": 0, "removed_benchmark_overlap": 0, "removed_duplicate": 0,
                    "removed_by_benchmark": Counter(), "bytes_kept": 0} for src in SOURCES}
    out_shards = []
    with Pool(args.workers, initializer=_init, initargs=(bench,)) as pool:
        results = pool.imap(scan, [project_path(s["file"]) for _, s in shards])
        for (src, s), res in zip(shards, results):
            docs = [d for d in open(project_path(s["file"]), "rb").read().split(EOT.encode()) if d]
            assert len(docs) == len(res) == s["documents"], s["file"]
            out_dir = project_path(f"phase17/data17d/{src}/filtered")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(s["file"]))
            kept = kept_bytes = 0
            r = report[src]
            with open(out_path + ".tmp", "wb") as f:
                for d, (hits, key) in zip(docs, res):
                    r["documents_in"] += 1
                    if hits:
                        r["removed_benchmark_overlap"] += 1
                        for task in sorted(set().union(*(owner[h] for h in hits))):
                            r["removed_by_benchmark"][task] += 1
                    elif key in seen:
                        r["removed_duplicate"] += 1
                    else:
                        seen.add(key)
                        f.write(d + EOT.encode())
                        kept += 1
                        kept_bytes += len(d)
            os.replace(out_path + ".tmp", out_path)
            r["kept"] += kept
            r["bytes_kept"] += kept_bytes
            out_shards.append({"source": src, "split": s["split"], "file": os.path.relpath(out_path, ROOT), "from": s["file"],
                               "documents": kept, "text_bytes": kept_bytes, "sha256": sha256_text(out_path)})
            print(f"{src} {os.path.basename(s['file'])}: kept {kept:,} of {s['documents']:,}", flush=True)
    for src in SOURCES:
        report[src]["removed_by_benchmark"] = dict(report[src]["removed_by_benchmark"])
    atomic_write_json(project_path(OUT_MANIFEST), {"method": __doc__.strip().split("\n\n")[1], "benchmark_13grams": int(len(bench)),
                                                   "report": report, "shards": out_shards, "seconds": round(time.time() - t0)})
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""
Phase 18 — 13-gram overlap between each benchmark's evaluated items and MiniGPT's training data.

An item is the exact text lm-eval evaluates (its prompt) plus its correct answer, taken from lm-eval's own task
objects. Text is lowercased and reduced to words ([a-z0-9]+). An item counts as "overlapping" if ANY of its 13-word
sequences also occurs in the training text. Items with fewer than 13 words cannot be tested and are reported apart.

Training text checked:
  - pretraining: every Phase 10 train text shard (phase10/data/text/train_*.txt, hash-checked). This includes the
    ~1% of documents that were later removed by the Phase 5D filter, so the pretraining rate can only be overstated.
  - SFT: phase11/data/sft_v2_train.jsonl (the released model's instruction data, hash-checked), all text fields.

13-grams are compared by a 64-bit hash (word hashes combined polynomially). PYTHONHASHSEED is fixed by the runner so
all worker processes hash words identically. A 64-bit collision could only add false overlaps, never hide one.

Usage (from phase18/, so the repo's evaluate.py cannot shadow Hugging Face evaluate):
    PYTHONHASHSEED=0 python contamination.py --workers 6
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

N = 13
WORD = re.compile(r"[a-z0-9]+")
PRIME = np.uint64(1_000_003)
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "sciq", "openbookqa", "boolq", "lambada_openai", "mmlu", "gsm8k", "triviaqa"]
SFT_FILE = ("phase11/data/sft_v2_train.jsonl", "b4a7dadef407b40954f068297426ed1247a3f3c4cf2faad223f6751f3dd7efa0")
OUT = os.path.join(os.path.dirname(__file__), "results", "contamination.json")


def ngram_hashes(text: str) -> np.ndarray:
    words = WORD.findall(text.lower())
    if len(words) < N:
        return np.empty(0, dtype=np.uint64)
    w = np.fromiter((hash(x) & 0xFFFFFFFFFFFFFFFF for x in words), dtype=np.uint64, count=len(words))
    h = np.zeros(len(words) - N + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h = h * PRIME + w[k:len(words) - N + 1 + k]
    return h


def benchmark_items() -> Dict[str, List[str]]:
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.api.task import Task

    def flatten(d):
        for key, value in d.items():
            if isinstance(value, Task):
                yield value
            elif isinstance(value, dict):
                yield from flatten(value)

    items: Dict[str, List[str]] = {}
    tasks = get_task_dict(TASKS, TaskManager())
    for top in TASKS:
        sub = {k: v for k, v in tasks.items() if (getattr(k, "group_name", None) or getattr(k, "group", None) or k) == top}
        texts = []
        for task in flatten(sub if sub else {top: tasks[top]}):
            for doc in task.eval_docs:
                text = task.doc_to_text(doc)
                choices = task.doc_to_choice(doc) if task.OUTPUT_TYPE == "multiple_choice" else None
                if isinstance(text, int) and choices:
                    text = choices[text]
                target = task.doc_to_target(doc)
                if isinstance(target, int) and choices:
                    target = choices[target]
                elif isinstance(target, list):
                    target = target[0] if target else ""
                texts.append(f"{text} {target}")
        items[top] = texts
    return items


def scan_file(args):
    path, kind, bench_sorted = args
    found = []
    if kind == "text":
        with open(path, "rb") as f:
            docs = f.read().decode("utf-8").split("<|endoftext|>")
    else:
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append("\n".join(v for k, v in row.items() if isinstance(v, str) and k not in ("id", "source", "subtype")))
    for doc in docs:
        h = ngram_hashes(doc)
        if len(h):
            pos = np.searchsorted(bench_sorted, h)
            pos[pos == len(bench_sorted)] = 0
            hit = bench_sorted[pos] == h
            if hit.any():
                found.append(np.unique(h[hit]))
    return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.uint64)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0 so every worker hashes words the same way")
    sys.path.insert(0, PROJECT_ROOT)
    from phase10.common import TEXT_MANIFEST, project_path, sha256_text
    sys.path.remove(PROJECT_ROOT)

    start = time.time()
    items = benchmark_items()
    item_hashes = {task: [ngram_hashes(t) for t in texts] for task, texts in items.items()}
    bench = np.unique(np.concatenate([h for hs in item_hashes.values() for h in hs if len(h)]))
    print(f"{sum(len(v) for v in items.values()):,} items, {len(bench):,} distinct 13-grams ({time.time()-start:.0f}s)", flush=True)

    with open(project_path(TEXT_MANIFEST), encoding="utf-8") as f:
        shards = [s for s in json.load(f)["shards"] if s["split"] == "train"]
    for s in shards:
        if sha256_text(project_path(s["file"])) != s["sha256"]:
            raise ValueError(f"{s['file']} differs from its pin")
    sft_path = project_path(SFT_FILE[0])
    if sha256(sft_path) != SFT_FILE[1]:
        raise ValueError("SFT v2 data differs from its pin")

    sources = {"pretraining": [(project_path(s["file"]), "text", bench) for s in shards], "sft_v2": [(sft_path, "jsonl", bench)]}
    matched = {}
    with Pool(args.workers) as pool:
        for name, jobs in sources.items():
            t = time.time()
            parts = pool.map(scan_file, jobs)
            matched[name] = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
            print(f"{name}: {len(matched[name]):,} benchmark 13-grams found ({time.time()-t:.0f}s)", flush=True)

    report = {"method": __doc__.strip().split("\n\n")[1:3], "n": N, "sources": {
        "pretraining": {"files": len(shards), "text_bytes": sum(s["text_bytes"] for s in shards), "documents": sum(s["documents"] for s in shards)},
        "sft_v2": {"file": SFT_FILE[0], "sha256": SFT_FILE[1]}}, "tasks": {}}
    for task, hashes in item_hashes.items():
        testable = [h for h in hashes if len(h)]
        row = {"items": len(hashes), "too_short_to_test": len(hashes) - len(testable)}
        for name, found in matched.items():
            hit = sum(bool(np.isin(h, found).any()) for h in testable)
            row[name] = {"overlapping_items": hit, "rate_of_testable": round(hit / len(testable), 4) if testable else None}
        both = sum(bool(np.isin(h, matched["pretraining"]).any() or np.isin(h, matched["sft_v2"]).any()) for h in testable)
        row["either"] = {"overlapping_items": both, "rate_of_testable": round(both / len(testable), 4) if testable else None}
        report["tasks"][task] = row
        print(task, json.dumps(row), flush=True)
    report["seconds"] = round(time.time() - start, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

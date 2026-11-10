"""
Phase 20 — contamination check for the sealed v4 sets, XSTest and the dev misfire probes, against ALL training data.
Prints and saves ONLY counts and flagged item ids: no prompt, answer or training text is ever printed or written.

Checks per item (same as the Phase 19 showcase checker):
  1. all SFT prompts (phase11/data/*.jsonl, phase17/sft_v5/sft_v5_{train,val}.jsonl): exact normalised prompt, any
     shared 8-word run, word-set Jaccard >= 0.7
  2. earlier non-sealed evaluation prompts (phase9 goal sets v1/v2, phase11 dev v2.1, old over-refusal): same checks
  3. (unless --sft-only) all 101 pretraining training shards (Phase 10, 17A, filtered 17D): any shared 13-word run
     with the prompt + expected answer
  4. across the given files: exact / 8-word / Jaccard >= 0.7 between items of DIFFERENT files

Usage:  PYTHONHASHSEED=0 python phase20/check_contamination_v4.py [--sft-only] --out NAME file1.jsonl [file2.jsonl ...]
Writes phase20/results/contamination_NAME.json (ids and reasons only).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase18"))

import contamination as c18  # noqa: E402
from phase10.common import TEXT_MANIFEST as P10_MANIFEST, project_path  # noqa: E402

SFT_FILES = sorted(glob.glob(os.path.join(ROOT, "phase11/data/*.jsonl"))) + [
    os.path.join(ROOT, "phase17/sft_v5/sft_v5_train.jsonl"), os.path.join(ROOT, "phase17/sft_v5/sft_v5_val.jsonl")]
PREVIOUS = [("phase10-foundation-v2", "phase9/goal_eval/goal_prompts.jsonl"), ("phase10-foundation-v2", "phase9/goal_eval/goal_prompts_v2.jsonl"),
            ("phase11-instruct", "phase11/goal_eval/goal_prompts_v2_1.jsonl"), ("phase11-instruct", "phase11/overrefusal/overrefusal_prompts.jsonl")]
PRETRAIN_MANIFESTS = [P10_MANIFEST, "phase17/manifests/text_manifest.json", "phase17/manifests17d/filter_manifest.json"]
WORD = re.compile(r"[a-z0-9]+")


def words(t):
    return WORD.findall(t.lower().replace("’", "'"))


def grams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def answer_text(it):
    e = it.get("expected")
    if isinstance(e, dict):
        return " ".join(e.get("all", []) + e.get("any", [])[:1])
    return str(e) if e is not None else ""


def jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-only", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0")
    items = {}
    for f in a.files:
        for line in open(f, encoding="utf-8"):
            it = json.loads(line)
            items[(os.path.basename(f), it["id"])] = it

    sft = set()
    for path in SFT_FILES:
        for line in open(path, encoding="utf-8"):
            p = json.loads(line).get("prompt")
            if isinstance(p, str):
                sft.add(" ".join(words(p)))
    prev = []
    for branch, path in PREVIOUS:
        out = subprocess.run(["git", "show", f"{branch}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        prev += [json.loads(l).get("prompt", "") for l in out.splitlines() if l.strip()]

    qw = {k: words(it["prompt"]) for k, it in items.items()}
    q8 = {k: grams(w, 8) for k, w in qw.items()}
    qs = {k: set(w) for k, w in qw.items()}
    all_q8 = set().union(*q8.values())
    flags = {k: set() for k in items}
    for n in sft:
        ws = n.split()
        if len(ws) >= 8 and all_q8:
            g = grams(ws, 8) & all_q8
            if g:
                for k in items:
                    if g & q8[k]:
                        flags[k].add("8gram_sft")
        s = set(ws)
        if len(s) > 60:
            continue
        for k, q in qs.items():
            if q & s and jacc(q, s) >= 0.7:
                flags[k].add("jaccard_sft")
    for k, w in qw.items():
        if " ".join(w) in sft:
            flags[k].add("exact_sft")
    prev_n = {" ".join(words(p)) for p in prev}
    prev_8 = set().union(*(grams(words(p), 8) for p in prev))
    prev_s = [set(words(p)) for p in prev if words(p)]
    for k, w in qw.items():
        if " ".join(w) in prev_n:
            flags[k].add("exact_earlier_eval")
        if q8[k] & prev_8:
            flags[k].add("8gram_earlier_eval")
        if any(jacc(qs[k], t) >= 0.7 for t in prev_s):
            flags[k].add("jaccard_earlier_eval")
    keys = list(items)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if k1[0] == k2[0]:
                continue
            if qw[k1] == qw[k2] or (q8[k1] & q8[k2]) or jacc(qs[k1], qs[k2]) >= 0.7:
                flags[k1].add(f"cross_file:{k2[0]}")
                flags[k2].add(f"cross_file:{k1[0]}")

    testable = 0
    if not a.sft_only:
        hashes = {k: c18.ngram_hashes(f"{it['prompt']} {answer_text(it)}") for k, it in items.items()}
        tk = {k: h for k, h in hashes.items() if len(h)}
        testable = len(tk)
        bench = np.unique(np.concatenate(list(tk.values())))
        shards = []
        for m in PRETRAIN_MANIFESTS:
            for s in json.load(open(project_path(m)))["shards"]:
                if s.get("split") == "train":
                    shards.append(project_path(s["file"]))
        with Pool(8) as pool:
            parts = pool.map(c18.scan_file, [(p, "text", bench) for p in shards], chunksize=1)
        found = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
        for k, h in tk.items():
            if np.isin(h, found).any():
                flags[k].add("13gram_pretraining")

    by_file = {}
    for (f, i), fl in flags.items():
        d = by_file.setdefault(f, {"items": 0, "flagged": {}})
        d["items"] += 1
        if fl:
            d["flagged"][i] = sorted(fl)
    report = {"sft_prompts": len(sft), "earlier_eval_prompts": len(prev), "pretraining_checked": not a.sft_only,
              "pretraining_testable_items": testable, "files": by_file}
    json.dump(report, open(os.path.join(ROOT, f"phase20/results/contamination_{a.out}.json"), "w"), indent=1)
    for f, d in by_file.items():
        print(f"{f}: {d['items']} items, {len(d['flagged'])} flagged", flush=True)
        for i, fl in d["flagged"].items():
            print("  ", i, fl)


if __name__ == "__main__":
    main()

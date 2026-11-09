"""
Phase 19 showcase — contamination check of the 40 new questions against ALL training data, before any model is run.

Checked against:
  1. Instruction (SFT) data of both models and every other SFT version: phase11/data/*.jsonl (v1.0.1 = sft_v2) and
     phase17/sft_v5/sft_v5_{train,val}.jsonl (v2.0-preview), prompts only:
       - exact match of the normalised prompt
       - any shared run of 8 consecutive words
       - word-set overlap (Jaccard) of 0.7 or more
     The nearest SFT prompt by Jaccard is recorded for every question (to judge closeness to templates).
  2. Earlier, non-sealed evaluation prompts (phase9 goal sets v1/v2, phase11 dev v2.1, the old over-refusal set):
     exact, 8-word run, Jaccard >= 0.7. Sealed sets (phase13/14 sealed, phase17 sealed_v3) and the Phase 11 held-out
     set are deliberately NOT opened.
  3. All pretraining text: Phase 10 (34M and v4 base), Phase 17A (v4) and the filtered Phase 17D shards (v4 extended):
     any shared run of 13 words with the question + its expected answer (the Phase 18 method; shorter items cannot
     be tested). File sizes are checked against the manifests.

Usage:  PYTHONHASHSEED=0 python phase19/showcase/check_contamination.py questions_vN.jsonl
Writes contamination_<input name>.json next to the input.
"""

import glob
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
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


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0")
    src = os.path.abspath(sys.argv[1])
    qs = [json.loads(l) for l in open(src, encoding="utf-8")]

    sft = {}
    for path in SFT_FILES:
        for line in open(path, encoding="utf-8"):
            p = json.loads(line).get("prompt")
            if isinstance(p, str):
                sft.setdefault(" ".join(words(p)), p)
    prev = []
    for branch, path in PREVIOUS:
        out = subprocess.run(["git", "show", f"{branch}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        prev += [json.loads(l).get("prompt", "") for l in out.splitlines() if l.strip()]
    print(f"SFT files: {len(SFT_FILES)}; distinct SFT prompts: {len(sft):,}; earlier eval prompts: {len(prev):,}", flush=True)

    qwords = {q["id"]: words(q["prompt"]) for q in qs}
    q8 = {k: grams(w, 8) for k, w in qwords.items()}
    all_q8 = set().union(*q8.values())
    hit8 = {k: [] for k in qwords}
    best = {k: (0.0, "") for k in qwords}
    qsets = {k: set(w) for k, w in qwords.items()}
    for n, p in sft.items():
        ws = n.split()
        if all_q8 and len(ws) >= 8:
            g = grams(ws, 8) & all_q8
            if g:
                for k in qwords:
                    if g & q8[k]:
                        hit8[k].append(sft[n][:200])
        s = set(ws)
        if len(s) > 60:
            continue
        for k, qs_ in qsets.items():
            inter = len(qs_ & s)
            if inter:
                j = inter / len(qs_ | s)
                if j > best[k][0]:
                    best[k] = (j, sft[n][:200])
    prev_n = {" ".join(words(p)) for p in prev}
    prev_8 = set().union(*(grams(words(p), 8) for p in prev))
    prev_sets = [set(words(p)) for p in prev if words(p)]

    report = {"input": os.path.basename(src), "checks": __doc__.strip().split("\n\n")[1], "sft_files": [os.path.relpath(p, ROOT) for p in SFT_FILES],
              "distinct_sft_prompts": len(sft), "earlier_eval_prompts": len(prev), "items": {}}
    for q in qs:
        k, ws = q["id"], qwords[q["id"]]
        reasons = []
        if " ".join(ws) in sft:
            reasons.append("exact_sft_prompt")
        if hit8[k]:
            reasons.append("8gram_sft_prompt")
        if best[k][0] >= 0.7:
            reasons.append("jaccard_ge_0.7_sft_prompt")
        if " ".join(ws) in prev_n:
            reasons.append("exact_earlier_eval")
        if grams(ws, 8) & prev_8:
            reasons.append("8gram_earlier_eval")
        if any(len(set(ws) & t) / len(set(ws) | t) >= 0.7 for t in prev_sets):
            reasons.append("jaccard_ge_0.7_earlier_eval")
        report["items"][k] = {"flags": reasons, "nearest_sft_jaccard": round(best[k][0], 3), "nearest_sft_prompt": best[k][1],
                              "sft_8gram_examples": hit8[k][:3]}

    texts = {q["id"]: f"{q['prompt']} {q['expected']}" for q in qs}
    hashes = {k: c18.ngram_hashes(t) for k, t in texts.items()}
    testable = {k: h for k, h in hashes.items() if len(h)}
    bench = np.unique(np.concatenate(list(testable.values())))
    shards = []
    for manifest in PRETRAIN_MANIFESTS:
        for s in json.load(open(project_path(manifest)))["shards"]:
            if s.get("split") == "train":
                path = project_path(s["file"])
                if "file_bytes" in s and os.path.getsize(path) != s["file_bytes"]:
                    raise ValueError(f"{s['file']} size differs from its manifest")
                shards.append(path)
    print(f"pretraining shards: {len(shards)} ({sum(os.path.getsize(p) for p in shards) / 1e9:.1f} GB)", flush=True)
    with Pool(8) as pool:
        parts = pool.map(c18.scan_file, [(p, "text", bench) for p in shards], chunksize=1)
    found = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
    for k in report["items"]:
        if k in testable:
            ov = bool(np.isin(testable[k], found).any())
            report["items"][k]["pretraining_13gram"] = "overlap" if ov else "none"
            if ov:
                report["items"][k]["flags"].append("13gram_pretraining")
        else:
            report["items"][k]["pretraining_13gram"] = "too_short"
    report["pretraining_shards"] = len(shards)
    report["flagged"] = [k for k, v in report["items"].items() if v["flags"]]
    out = os.path.join(os.path.dirname(src), "contamination_" + os.path.basename(src).replace(".jsonl", ".json"))
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    print("flagged:", {k: report["items"][k]["flags"] for k in report["flagged"]})
    print("too short for 13-gram:", [k for k, v in report["items"].items() if v["pretraining_13gram"] == "too_short"])


if __name__ == "__main__":
    main()

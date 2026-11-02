"""
Phase 11D — Builds the sft_v3 dataset: the v2 data, with the over-refusal guard and support-contrast sources replaced by
diversified versions, plus contrast pairs (harmless uses of risky keywords and their refused counterparts).

A global cap limits every benign reply text (harmless contrast, diversified guard, diversified support) to 3 uses.
Refusals are exempt. New rows are contamination-checked against dev v2.1, held-out, the over-refusal set and the demo prompts.
The written files are re-verified. Only counts are recorded for the sealed sets.

Usage:
    RAYON_NUM_THREADS=8 python phase11/build_sft_v3.py
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path, sha256_text
from phase11.build_sft_data import DEV_PATH, HELDOUT_PATH, find_matches, length_filter, split_for, token_stats
from phase11.build_sft_v2 import DEMO_PROMPTS, OVERREFUSAL_PATH, knowledge_match, tokens_of
from phase11.goal_eval.scorer_v3 import load_goal_set
from phase11.synthetic_v3 import generate_all_v3
from phase9.check_contamination import ContaminationChecker, check_dataset

SEED = 1133
MAX_REPLY_USES = 3
V2 = {"train": ("phase11/data/sft_v2_train.jsonl", "b4a7dadef407b40954f068297426ed1247a3f3c4cf2faad223f6751f3dd7efa0"),
      "val": ("phase11/data/sft_v2_val.jsonl", "a74d51955015903765475063f66843ab8516d9fa8156c59acf550aca8c5635c9")}
REPLACED_SOURCES = {"synthetic_over_refusal_guard", "synth_support_contrast"}  # rewritten for reply diversity
CAPPED_SOURCES = {"synth_contrast_harmless", "synth_guard_v3", "synth_support_v3"}  # benign replies capped at 3 uses
OUT = {"train": "phase11/data/sft_v3_train.jsonl", "val": "phase11/data/sft_v3_val.jsonl"}
MANIFEST = "phase11/manifests/sft_v3_manifest.json"
STATS = "phase11/manifests/sft_v3_stats.json"


def read_pinned(path: str, expected: str) -> List[Dict[str, Any]]:
    if sha256_text(project_path(path)) != expected:
        raise ValueError(f"Pinned hash mismatch for {path}")
    with open(project_path(path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    dev = load_goal_set(*DEV_PATH)
    heldout = load_goal_set(*HELDOUT_PATH)
    overrefusal = load_goal_set(*OVERREFUSAL_PATH)
    demo = [{"id": f"demo_{i:02d}", "prompt": prompt} for i, prompt in enumerate(DEMO_PROMPTS)]
    sets = {"dev_v2_1": dev, "heldout": heldout, "overrefusal": overrefusal, "demo": demo}
    checkers = {name: ContaminationChecker(items) for name, items in sets.items()}
    goal_tokens = {name: [tokens_of(item.get("question", item["prompt"])) for item in items] for name, items in sets.items()}

    # v2 rows, dropping the two sources we are replacing. Record their old most-repeated reply for the before/after report.
    v2 = {split: read_pinned(*spec) for split, spec in V2.items()}
    before = {}
    for src in REPLACED_SOURCES:
        reps = Counter(r["text"] for rows in v2.values() for r in rows if r["source"] == src)
        before[src] = {"rows": sum(reps.values()), "distinct": len(reps), "max_reply_uses": reps.most_common(1)[0][1] if reps else 0}
    kept_v2 = {split: [r for r in rows if r["source"] not in REPLACED_SOURCES] for split, rows in v2.items()}

    counts: Dict[str, Counter] = defaultdict(Counter)
    new_kept: Dict[str, List[Dict[str, Any]]] = {}
    for name, rows in generate_all_v3(SEED).items():
        counts[name]["generated"] = len(rows)
        rows = length_filter(rows, tokenizer, counts[name])
        clean = []
        for row in rows:
            hit = [label for label, toks in goal_tokens.items() if knowledge_match(row, toks)]
            wording = [label for label, checker in checkers.items() if find_matches(checker, row)]
            for label in hit:
                counts[name][f"removed_knowledge_{label}"] += 1
            for label in wording:
                counts[name][f"removed_wording_{label}"] += 1
            if hit or wording:
                counts[name]["removed_contamination"] += 1
            else:
                clean.append(row)
        new_kept[name] = clean

    # Global cap on benign reply texts; refusals (synth_contrast_refused) exempt.
    reply_uses: Counter = Counter()
    for name in CAPPED_SOURCES:
        capped = []
        for row in new_kept[name]:
            if reply_uses[row["text"]] >= MAX_REPLY_USES:
                counts[name]["dropped_reply_cap"] += 1
                continue
            reply_uses[row["text"]] += 1
            capped.append(row)
        new_kept[name] = capped
    for name in new_kept:
        counts[name]["final"] = len(new_kept[name])

    split_rows: Dict[str, List[Dict[str, Any]]] = {split: [dict(r) for r in rows] for split, rows in kept_v2.items()}
    for name, rows in new_kept.items():
        for row in rows:
            record = {"source": name, "subtype": row["subtype"], "prompt": row["prompt"], "text": row["text"], "total_tokens": row["total_tokens"], "reply_tokens": row["reply_tokens"]}
            split = split_for(row["prompt"])
            split_rows[split].append(record)
            counts[name][split] += 1

    rng = random.Random(SEED)
    for split, rows in split_rows.items():
        rng.shuffle(rows)
        with open(project_path(OUT[split]), "w", encoding="utf-8", newline="\n") as f:
            for number, row in enumerate(rows):
                record = {"id": f"sftv3_{split}_{number:06d}", **{k: row[k] for k in ("source", "subtype", "prompt", "text", "total_tokens", "reply_tokens")}}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if {r["prompt"] for r in split_rows["train"]} & {r["prompt"] for r in split_rows["val"]}:
        raise SystemExit("A prompt appears in both train and val")
    verification = {}
    for split, path in OUT.items():
        for label, items in sets.items():
            report = check_dataset(project_path(path), items)
            verification[f"{split}_{label}"] = {"records": report["records"], "matched": len(report["goal_items_matched"])}
    if any(e["matched"] for e in verification.values()):
        raise SystemExit(f"Contamination remains: {verification}")

    # After: most-repeated benign reply among the new diversified sources.
    after = {}
    for src, repl in (("synthetic_over_refusal_guard", "synth_guard_v3"), ("synth_support_contrast", "synth_support_v3")):
        reps = Counter(r["text"] for rows in split_rows.values() for r in rows if r["source"] == repl)
        after[repl] = {"rows": sum(reps.values()), "distinct": len(reps), "max_reply_uses": reps.most_common(1)[0][1] if reps else 0}

    examples = {name: [{"prompt": r["prompt"], "reply": r["text"]} for r in random.Random(f"{SEED}-{name}").sample(rows, min(5, len(rows)))] for name, rows in new_kept.items()}
    stats = {
        "seed": SEED, "max_reply_uses": MAX_REPLY_USES,
        "per_source": {name: dict(sorted(counts[name].items())) for name in new_kept},
        "subtypes": {name: dict(Counter(r["subtype"] for r in rows)) for name, rows in new_kept.items()},
        "reply_diversity_before": before, "reply_diversity_after": after,
        "replaced_sources_removed": {split: sum(r["source"] in REPLACED_SOURCES for r in rows) for split, rows in v2.items()},
        "totals": {split: len(rows) for split, rows in split_rows.items()}, "train_tokens": sum(r["total_tokens"] for r in split_rows["train"]),
        "verification_after_removal": verification, "examples": examples,
    }
    atomic_write_json(project_path(STATS), stats)
    atomic_write_json(project_path(MANIFEST), {
        "seed": SEED, "tokenizer_sha256": sha256_text(project_path(TOKENIZER_PATH)),
        "base_dataset": {split: {"file": path, "sha256": sha} for split, (path, sha) in V2.items()},
        "replaced_sources": sorted(REPLACED_SOURCES), "capped_sources": sorted(CAPPED_SOURCES),
        "sets_checked": {"dev_v2_1": DEV_PATH[0], "heldout": HELDOUT_PATH[0], "overrefusal": OVERREFUSAL_PATH[0], "demo_prompts": DEMO_PROMPTS},
        "files": {split: {"file": path, "records": len(split_rows[split]), "sha256": sha256_text(project_path(path))} for split, path in OUT.items()},
    })
    print(json.dumps({k: stats[k] for k in ("per_source", "subtypes", "reply_diversity_before", "reply_diversity_after", "replaced_sources_removed", "totals", "train_tokens", "verification_after_removal")}, indent=1), flush=True)
    for name, rows in examples.items():
        print(f"\n### {name}")
        for r in rows:
            print(f"  P: {r['prompt']}\n  R: {r['reply']}")
    print(json.dumps(json.load(open(project_path(MANIFEST)))["files"], indent=1))


if __name__ == "__main__":
    main()

"""
Phase 11C — Builds the sft_v2 dataset: the old noformat rows plus the new synthetic rows.

Checks, in order: length limits, prompts already in the old data, key stems (knowledge check), then contamination against goal dev v2.1, held-out,
the over-refusal set and the demo prompts. Old rows also go through contamination, since the last two sets are new. Matches are removed.
The written files are re-verified. Only counts are recorded for the sealed sets.

Usage:
    RAYON_NUM_THREADS=8 python phase11/build_sft_v2.py
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path, sha256_text
from phase11.build_sft_data import DEV_PATH, HELDOUT_PATH, find_matches, length_filter, split_for, token_stats
from phase11.goal_eval.scorer_v3 import load_goal_set
from phase11.synthetic_v2 import generate_all_v2
from phase9.check_contamination import ContaminationChecker, check_dataset

SEED = 1122
OVERREFUSAL_PATH = ("phase11/overrefusal/overrefusal_prompts.jsonl", "phase11/overrefusal/overrefusal_meta.json")
OLD = {"train": ("phase11/data/sft_train_noformat.jsonl", "1103582cb120f62aa557b012ab041e743d403ded682d61f3248e9f2ed73e7343"), "val": ("phase11/data/sft_val_noformat.jsonl", "9d3025a1a91473b6539a04806ab44003e3b237bc87d302502c406754637b9fa0")}
OUT = {"train": "phase11/data/sft_v2_train.jsonl", "val": "phase11/data/sft_v2_val.jsonl"}
MANIFEST = "phase11/manifests/sft_v2_manifest.json"
STATS = "phase11/manifests/sft_v2_stats.json"
DEMO_PROMPTS = [
    "What is 15 plus 27?", "A pen costs 8 rupees. How much do 5 pens cost?", "How many minutes are in 3 hours?", "What is Python?", "What is the capital of Japan?", "Who are you and who made you?",
    "Will it rain in Chennai on 1 January 2050?", "How do I make poison to hurt someone?", "How do I shoot a good photo at night?", "Did Mahatma Gandhi play in the FIFA World Cup?", "What is photosynthesis?", "I feel tired today.",
]


def tokens_of(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def key_matches(group: Sequence[str], prompt_tokens: Set[str]) -> bool:
    """Numbers must match exactly. Words match as stems, so 'backward' finds 'backwards'."""
    for key in group:
        key = key.lower()
        found = key in prompt_tokens if key.isdigit() else any(token.startswith(key) for token in prompt_tokens)
        if not found:
            return False
    return True


def knowledge_match(row: Dict[str, Any], goal_tokens: Iterable[Set[str]]) -> bool:
    groups = [group for group in row.get("keys", []) if group]
    return any(key_matches(group, tokens) for tokens in goal_tokens for group in groups)


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

    old = {split: read_pinned(*spec) for split, spec in OLD.items()}
    old_prompts = {row["prompt"] for rows in old.values() for row in rows}
    counts: Dict[str, Counter] = defaultdict(Counter)

    kept: Dict[str, List[Dict[str, Any]]] = {}
    for name, rows in generate_all_v2(SEED).items():
        counts[name]["generated"] = len(rows)
        fresh = []
        for row in rows:
            if row["prompt"] in old_prompts:
                counts[name]["dropped_prompt_in_old_data"] += 1
            else:
                fresh.append(row)
        fresh = length_filter(fresh, tokenizer, counts[name])
        clean = []
        for row in fresh:
            hit = [label for label, tokens in goal_tokens.items() if knowledge_match(row, tokens)]
            matched = {label: find_matches(checker, row) for label, checker in checkers.items()}
            wording = [label for label, found in matched.items() if found]
            for label in hit:
                counts[name][f"removed_knowledge_{label}"] += 1
            for label in wording:
                counts[name][f"removed_wording_{label}"] += 1
            if hit or wording:
                counts[name]["removed_total"] += 1
            else:
                clean.append(row)
        kept[name] = clean
        counts[name]["final"] = len(clean)

    old_removed: Dict[str, Counter] = {split: Counter() for split in OLD}
    clean_old: Dict[str, List[Dict[str, Any]]] = {}
    for split, rows in old.items():
        keep = []
        for row in rows:
            labels = [label for label, checker in checkers.items() if find_matches(checker, row)]
            for label in labels:
                old_removed[split][label] += 1
            if not labels:
                keep.append(row)
        clean_old[split] = keep

    split_rows: Dict[str, List[Dict[str, Any]]] = {split: list(rows) for split, rows in clean_old.items()}
    for name, rows in kept.items():
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
                record = {"id": f"sftv2_{split}_{number:06d}", **{key: row[key] for key in ("source", "subtype", "prompt", "text", "total_tokens", "reply_tokens")}}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if {row["prompt"] for row in split_rows["train"]} & {row["prompt"] for row in split_rows["val"]}:
        raise SystemExit("A prompt appears in both train and val")
    verification = {}
    for split, path in OUT.items():
        for label, items in sets.items():
            report = check_dataset(project_path(path), items)
            verification[f"{split}_{label}"] = {"records": report["records"], "matched": len(report["goal_items_matched"])}
    if any(entry["matched"] for entry in verification.values()):
        raise SystemExit(f"Contamination remains: {verification}")

    examples = {name: [{"prompt": r["prompt"], "reply": r["text"]} for r in random.Random(f"{SEED}-{name}").sample(rows, 5)] for name, rows in kept.items()}
    subtype_counts = {name: dict(Counter(r["subtype"] for r in rows)) for name, rows in kept.items()}
    stats = {
        "seed": SEED, "per_source": {name: dict(sorted(counts[name].items())) for name in kept}, "subtypes": subtype_counts, "token_stats": {name: {"total": token_stats([r["total_tokens"] for r in rows]), "reply": token_stats([r["reply_tokens"] for r in rows])} for name, rows in kept.items()},
        "old_rows_removed": {split: dict(counter) for split, counter in old_removed.items()}, "old_rows_kept": {split: len(rows) for split, rows in clean_old.items()},
        "totals": {split: len(rows) for split, rows in split_rows.items()}, "train_tokens": sum(r["total_tokens"] for r in split_rows["train"]), "verification_after_removal": verification, "examples": examples,
    }
    atomic_write_json(project_path(STATS), stats)
    atomic_write_json(project_path(MANIFEST), {
        "seed": SEED, "tokenizer_sha256": sha256_text(project_path(TOKENIZER_PATH)), "old_data": {split: {"file": path, "sha256": sha} for split, (path, sha) in OLD.items()},
        "sets_checked": {"dev_v2_1": DEV_PATH[0], "heldout": HELDOUT_PATH[0], "overrefusal": OVERREFUSAL_PATH[0], "demo_prompts": DEMO_PROMPTS},
        "files": {split: {"file": path, "records": len(split_rows[split]), "sha256": sha256_text(project_path(path))} for split, path in OUT.items()},
    })
    print(json.dumps({key: stats[key] for key in ("per_source", "subtypes", "old_rows_removed", "old_rows_kept", "totals", "train_tokens", "verification_after_removal")}, indent=1), flush=True)
    for name, rows in examples.items():
        print(f"\n### {name}")
        for row in rows:
            print(f"  P: {row['prompt']}\n  R: {row['reply']}")
    print(json.dumps(json.load(open(project_path(MANIFEST)))["files"], indent=1))


if __name__ == "__main__":
    main()

"""
Phase 9 — Reports overlap between any dataset and the goal set. Future training data must pass it.

Checks:
    exact      same text after lowercasing and removing punctuation
    near       word-set similarity (Jaccard) at or above the threshold, for short records
    contained  a goal prompt or RAG passage of at least MIN_CONTAINED_WORDS words appears inside a longer record

Accepts .jsonl (text fields below) or plain text (each line is a record). Reads line by line, so large files stay
out of RAM. Exits 1 if anything is found, unless --allow-flagged is set and only goal items already marked
"overlap_sft" match.

Usage:
    python phase9/check_contamination.py phase6/data/sft/train_sft.jsonl
    python phase9/check_contamination.py data/phase5d/train_corpus.txt --out phase9/contamination/fineweb_train.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOAL_SET_PATH = os.path.join(PROJECT_ROOT, "phase9", "goal_eval", "goal_prompts_v2.jsonl")
TEXT_FIELDS = ("instruction", "prompt", "question", "text", "input", "context", "passage")
NEAR_THRESHOLD = 0.8
MIN_NEAR_WORDS = 3
MAX_NEAR_WORDS = 60
MIN_CONTAINED_WORDS = 6
MAX_EXAMPLES = 50


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def goal_texts(item: Dict[str, Any]) -> List[str]:
    """The question is what a model would see trained on. RAG passages are checked as well."""
    texts = [item.get("question", item["prompt"])]
    if item.get("passage"):
        texts.append(item["passage"])
    return texts


class ContaminationChecker:

    def __init__(self, items: List[Dict[str, Any]], threshold: float = NEAR_THRESHOLD):
        self.threshold = threshold
        self.flagged = {item["id"] for item in items if item.get("overlap_sft")}
        self.exact: Dict[str, List[str]] = defaultdict(list)
        self.word_sets: Dict[str, Tuple[Set[str], str]] = {}
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.contained: List[Tuple[str, str]] = []
        for item in items:
            for position, text in enumerate(goal_texts(item)):
                normalized = normalize(text)
                self.exact[normalized].append(item["id"])
                words = set(normalized.split())
                key = f"{item['id']}#{position}"
                self.word_sets[key] = (words, item["id"])
                for word in words:
                    self.index[word].add(key)
                if len(normalized.split()) >= MIN_CONTAINED_WORDS:
                    self.contained.append((f" {normalized} ", item["id"]))

    def check_text(self, text: str) -> List[Tuple[str, str, float]]:
        """Returns (kind, goal id, similarity) for every match in one record's text."""
        normalized = normalize(text)
        if not normalized:
            return []
        matches: Dict[str, Tuple[str, float]] = {}
        for goal_id in self.exact.get(normalized, []):
            matches[goal_id] = ("exact", 1.0)

        words = set(normalized.split())
        if MIN_NEAR_WORDS <= len(words) <= MAX_NEAR_WORDS:
            candidates = {key for word in words for key in self.index.get(word, ())}
            for key in candidates:
                goal_words, goal_id = self.word_sets[key]
                if len(goal_words) < MIN_NEAR_WORDS or goal_id in matches:
                    continue
                similarity = len(words & goal_words) / len(words | goal_words)
                if similarity >= self.threshold:
                    matches[goal_id] = ("near", round(similarity, 3))

        padded = f" {normalized} "
        for phrase, goal_id in self.contained:
            if goal_id not in matches and phrase in padded:
                matches[goal_id] = ("contained", 1.0)
        return [(kind, goal_id, similarity) for goal_id, (kind, similarity) in matches.items()]


def iter_records(path: str) -> Iterable[Tuple[int, List[str]]]:
    """Yields (line number, text fields) for a JSONL or plain text file."""
    is_jsonl = path.endswith(".jsonl")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, start=1):
            if is_jsonl:
                record = json.loads(line)
                yield number, [str(record[field]) for field in TEXT_FIELDS if record.get(field)]
            else:
                yield number, [line]


def check_dataset(path: str, items: List[Dict[str, Any]], threshold: float = NEAR_THRESHOLD) -> Dict[str, Any]:
    checker = ContaminationChecker(items, threshold)
    by_kind = {"exact": set(), "near": set(), "contained": set()}
    examples = []
    records = 0
    for number, texts in iter_records(path):
        records += 1
        for text in texts:
            for kind, goal_id, similarity in checker.check_text(text):
                by_kind[kind].add(goal_id)
                if len(examples) < MAX_EXAMPLES:
                    examples.append({"line": number, "kind": kind, "goal_id": goal_id, "similarity": similarity, "text": text.strip()[:120]})
    matched = set().union(*by_kind.values())
    return {
        "dataset": path,
        "records": records,
        "threshold": threshold,
        "goal_items_matched": sorted(matched),
        "exact": sorted(by_kind["exact"]),
        "near": sorted(by_kind["near"] - by_kind["exact"]),
        "contained": sorted(by_kind["contained"] - by_kind["exact"] - by_kind["near"]),
        "unflagged_matches": sorted(matched - checker.flagged),
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Check a dataset against the goal set")
    parser.add_argument("dataset", help="Path to a .jsonl or plain text file")
    parser.add_argument("--threshold", type=float, default=NEAR_THRESHOLD, help="Near-duplicate Jaccard threshold")
    parser.add_argument("--allow-flagged", action="store_true", help="Pass if only items marked overlap_sft match")
    parser.add_argument("--out", default=None, help="Write the report to this JSON file")
    args = parser.parse_args()

    with open(GOAL_SET_PATH, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    report = check_dataset(args.dataset, items, args.threshold)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"{args.dataset}: {report['records']:,} records")
    print(f"  exact {len(report['exact'])}, near {len(report['near'])}, contained {len(report['contained'])}, unflagged {len(report['unflagged_matches'])}")
    for kind in ("exact", "near", "contained"):
        if report[kind]:
            print(f"  {kind}: {', '.join(report[kind])}")
    blocking = report["unflagged_matches"] if args.allow_flagged else report["goal_items_matched"]
    print("  RESULT: " + ("FAIL" if blocking else "PASS"))
    if blocking:
        sys.exit(1)


if __name__ == "__main__":
    main()

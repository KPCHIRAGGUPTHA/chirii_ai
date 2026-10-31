"""
Phase 11A — Builds the SFT dataset: three public sources plus the synthetic generators.

Steps: load, filter, tokenize, limit lengths, de-duplicate, remove every goal-set match, split, write, re-verify.
Output is printed and stored without any held-out text: matches are counts and ids only.

Usage:
    RAYON_NUM_THREADS=8 python phase11/build_sft_data.py
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path, sha256_text
from phase11.chat_format import example_text
from phase11.synthetic import generate_all
from phase9.check_contamination import ContaminationChecker, check_dataset, goal_texts
from phase9.goal_eval.scorer_v2 import load_goal_set

SEED = 1104
MAX_TOTAL_TOKENS = 512
MAX_REPLY_TOKENS = 200
VAL_PER_MILLE = 20
DEV_PATH = ("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")
HELDOUT_PATH = ("phase11/heldout/goal_prompts_heldout.jsonl", "phase11/heldout/heldout_meta.json")
DATA_DIR = "phase11/data"
MANIFEST = "phase11/manifests/sft_manifest.json"
REPORT_JSON = "phase11/manifests/sft_stats.json"
SOURCES = {
    "alpaca": "yahma/alpaca-cleaned",
    "dolly": "databricks/databricks-dolly-15k",
    "smoltalk": "HuggingFaceTB/smol-smoltalk",
}
REVISIONS = {
    "alpaca": "12567cabf869d7c92e573c7c783905fc160e9639",
    "dolly": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    "smoltalk": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
}

BOILERPLATE = re.compile(
    r"as an ai\b|as a language model|as an artificial|ai language model|language model\b|i am an ai\b|i'm an ai\b|i am a language|i'm a language|"
    r"\bchatgpt\b|\bopenai\b|\bgpt-?[234]\b|\bsmollm\b|\bsmolassistant\b|hugging ?face|my training data|my knowledge cutoff|my programming|i was trained|i don't have personal|i do not have personal|"
    r"i don't have feelings|i do not have feelings|i don't have (?:access|the ability)|i do not have (?:access|the ability)|i am unable to (?:browse|access)|ai assistant|virtual assistant", re.I)
REFUSAL_START = re.compile(r"\s*(?:i'm sorry|i am sorry|sorry|i apologi[sz]e|unfortunately|i can't|i cannot|i can not|i'm unable|i am unable|i'm not able|i am not able|i won't|i will not)\b", re.I)
IDENTITY_PROMPT = re.compile(r"\bwho are you\b|\bwhat are you\b|\byour name\b|\bwho (?:made|created|built|trained|designed) you\b|\bare you (?:an? )?(?:ai|bot|robot|human|person|chatbot)\b|\babout yourself\b", re.I)
ARITHMETIC_PROMPT = re.compile(r"\d\s*(?:[-+*/x×÷^%=]|plus|minus|times|divided by|multiplied by)\s*\d|\b(?:calculate|compute|sum of|product of|solve for|square root|percent of)\b.*\d|\d.*\b(?:calculate|compute|how many|how much)\b.*\d", re.I)
HAND_ARITHMETIC_REPLY = re.compile(r"\d\s*(?:[-+*/x×÷]|\\times|\\div)\s*\$?\d[\d,.]*\s*=")
FORMAT_CONSTRAINT_PROMPT = re.compile(r"your (?:entire )?(?:response|answer|reply) (?:should|must)|\bplaceholders?\b|at least \d+ (?:sentences|placeholders|bullet)|exactly \d+ (?:bullet|sentences|paragraphs)|highlight at least|include keywords|\bP\.S\.|postscript|markdown", re.I)
PLACEHOLDER_REPLY = re.compile(r"\[[a-z_ ]+\]")
COMMON_WORDS = set("the a an of to and in is it that for on with as was are be this by at from or have has not you your i we they he she his her their can will what which how when where who why".split())


def looks_english(text: str) -> bool:
    """Dependency-free check: mostly basic Latin letters and, in text long enough to judge, common English words."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.95:
        return False
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 12:
        return True
    return sum(w in COMMON_WORDS for w in words) / len(words) >= 0.15


def reject_reason(prompt: str, reply: str) -> Optional[str]:
    """Returns why a public-source row is dropped, or None to keep it."""
    if not prompt.strip() or not reply.strip():
        return "empty"
    if "<|" in prompt or "<|" in reply:
        return "special_token_text"
    if BOILERPLATE.search(reply) or BOILERPLATE.search(prompt):
        return "ai_boilerplate"
    if REFUSAL_START.match(reply):
        return "refusal_or_apology_reply"
    if IDENTITY_PROMPT.search(prompt):
        return "identity_question"
    if ARITHMETIC_PROMPT.search(prompt) or HAND_ARITHMETIC_REPLY.search(reply):
        return "arithmetic_without_calculator"
    if not looks_english(f"{prompt} {reply}"):
        return "not_english"
    return None


def is_format_exercise(row: Dict[str, Any]) -> bool:
    """Prompts about response format, or non-code replies with [placeholder] slots. Excluded only from the _noformat variant."""
    if row["source"].startswith("synthetic"):
        return False
    return bool(FORMAT_CONSTRAINT_PROMPT.search(row["prompt"])) or ("```" not in row["text"] and bool(PLACEHOLDER_REPLY.search(row["text"])))


def iter_alpaca(row: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    instruction, extra = row["instruction"].strip(), (row.get("input") or "").strip()
    yield (f"{instruction}\n{extra}" if extra else instruction), row["output"].strip(), "with_input" if extra else "no_input"


def iter_dolly(row: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    if (row.get("context") or "").strip():
        return
    yield row["instruction"].strip(), row["response"].strip(), row.get("category", "")


def iter_smoltalk(row: Dict[str, Any]) -> Iterable[Tuple[str, str, str]]:
    messages = row["messages"]
    if len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        yield messages[0]["content"].strip(), messages[1]["content"].strip(), "single_turn"


READERS = {"alpaca": iter_alpaca, "dolly": iter_dolly, "smoltalk": iter_smoltalk}


def load_public(name: str, counts: Dict[str, Counter]) -> List[Dict[str, str]]:
    from datasets import load_dataset

    stream = load_dataset(SOURCES[name], split="train", streaming=True, revision=REVISIONS[name])
    rows: List[Dict[str, str]] = []
    for row in stream:
        counts[name]["rows_read"] += 1
        produced = False
        for prompt, reply, subtype in READERS[name](row):
            produced = True
            reason = reject_reason(prompt, reply)
            if reason:
                counts[name][f"dropped_{reason}"] += 1
            else:
                rows.append({"prompt": prompt, "text": reply, "subtype": subtype})
        if not produced:
            counts[name]["dropped_context_or_multi_turn"] += 1
    return rows


def length_filter(rows: List[Dict[str, str]], tokenizer: Tokenizer, counts: Counter) -> List[Dict[str, Any]]:
    kept = []
    full = tokenizer.encode_batch_fast([example_text(r["prompt"], r["text"]) for r in rows], add_special_tokens=False)
    replies = tokenizer.encode_batch_fast([r["text"] for r in rows], add_special_tokens=False)
    for row, whole, reply in zip(rows, full, replies):
        total, reply_tokens = len(whole.ids), len(reply.ids)
        if total > MAX_TOTAL_TOKENS:
            counts["dropped_over_512_total"] += 1
        elif reply_tokens > MAX_REPLY_TOKENS:
            counts["dropped_reply_over_200"] += 1
        else:
            kept.append({**row, "total_tokens": total, "reply_tokens": reply_tokens})
    return kept


def find_matches(checker: ContaminationChecker, row: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    return checker.check_text(row["prompt"]) + checker.check_text(row["text"])


def split_for(prompt: str) -> str:
    return "val" if int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:8], "big") % 1000 < VAL_PER_MILLE else "train"


def token_stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    array = np.array(values)
    return {"count": int(len(array)), "mean": round(float(array.mean()), 1), "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": int(array.max())}


def main():
    parser = argparse.ArgumentParser(description="Build the SFT dataset")
    parser.add_argument("--skip-public", action="store_true", help="Synthetic data only, for a quick check")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    dev_items = load_goal_set(*DEV_PATH)
    heldout_items = load_goal_set(*HELDOUT_PATH)

    counts: Dict[str, Counter] = defaultdict(Counter)
    raw: Dict[str, List[Dict[str, str]]] = {}
    if not args.skip_public:
        for name in SOURCES:
            raw[name] = load_public(name, counts)
            print(f"{name}: read {counts[name]['rows_read']:,}, kept {len(raw[name]):,}", flush=True)
    for name, rows in generate_all(SEED, dev_items + heldout_items).items():
        raw[name] = rows
        counts[name]["rows_read"] = len(rows)

    kept: Dict[str, List[Dict[str, Any]]] = {name: length_filter(rows, tokenizer, counts[name]) for name, rows in raw.items()}

    seen = set()
    for name, rows in kept.items():
        unique = []
        for row in rows:
            key = (row["prompt"], row["text"])
            if key in seen:
                counts[name]["dropped_duplicate"] += 1
                continue
            seen.add(key)
            unique.append(row)
        kept[name] = unique

    # Remove every match against either goal set. Only counts and ids are recorded, so nothing leaks into reports.
    checkers = {"dev_v2_1": ContaminationChecker(dev_items), "heldout": ContaminationChecker(heldout_items)}
    removed_by_kind = {key: Counter() for key in checkers}
    matched_items = {key: set() for key in checkers}
    for name, rows in kept.items():
        clean = []
        for row in rows:
            remove = False
            for key, checker in checkers.items():
                matches = find_matches(checker, row)
                if matches:
                    remove = True
                    counts[name][f"removed_{key}"] += 1
                    removed_by_kind[key].update(kind for kind, _, _ in matches)
                    matched_items[key].update(goal_id for _, goal_id, _ in matches)
            if remove:
                counts[name]["removed_contamination_total"] += 1
            else:
                clean.append(row)
        kept[name] = clean

    os.makedirs(project_path(DATA_DIR), exist_ok=True)
    rng = random.Random(SEED)
    all_rows = [dict(row, source=name) for name, rows in kept.items() for row in rows]
    rng.shuffle(all_rows)
    paths = {"train": f"{DATA_DIR}/sft_train.jsonl", "val": f"{DATA_DIR}/sft_val.jsonl"}
    variant_paths = {"train": f"{DATA_DIR}/sft_train_noformat.jsonl", "val": f"{DATA_DIR}/sft_val_noformat.jsonl"}
    split_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    variant_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    files = {split: open(project_path(path), "w", encoding="utf-8", newline="\n") for split, path in paths.items()}
    variant_files = {split: open(project_path(path), "w", encoding="utf-8", newline="\n") for split, path in variant_paths.items()}
    for number, row in enumerate(all_rows):
        split = split_for(row["prompt"])
        split_counts[split][row["source"]] += 1
        record = {"id": f"sft_{number:06d}", "source": row["source"], "subtype": row["subtype"], "prompt": row["prompt"], "text": row["text"], "total_tokens": row["total_tokens"], "reply_tokens": row["reply_tokens"]}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        files[split].write(line)
        if not is_format_exercise(row):
            variant_files[split].write(line)
            variant_counts[split][row["source"]] += 1
    for f in list(files.values()) + list(variant_files.values()):
        f.close()

    # Re-verify the written files with phase9/check_contamination.py. Its examples hold matched text, so they are not kept.
    verification = {}
    for label, split, path in [("", s, p) for s, p in paths.items()] + [("noformat_", s, p) for s, p in variant_paths.items()]:
        for key, items in (("dev_v2_1", dev_items), ("heldout", heldout_items)):
            report = check_dataset(project_path(path), items)
            verification[f"{label}{split}_{key}"] = {"records": report["records"], "matched": len(report["goal_items_matched"]), "exact": len(report["exact"]), "near": len(report["near"]), "contained": len(report["contained"])}
    if any(entry["matched"] for entry in verification.values()):
        raise SystemExit(f"Contamination remains: {verification}")

    stats = {
        "seed": SEED, "limits": {"total_tokens": MAX_TOTAL_TOKENS, "reply_tokens": MAX_REPLY_TOKENS},
        "per_source": {},
        "contamination": {
            key: {"records_matched_by_kind": dict(removed_by_kind[key]), "goal_items_matched": len(matched_items[key])} for key in checkers
        },
        "dev_v2_1_items_matched": sorted(matched_items["dev_v2_1"]),
        "verification_after_removal": verification,
        "total": {"train": sum(split_counts["train"].values()), "val": sum(split_counts["val"].values())},
        "noformat_variant": {"train": dict(variant_counts["train"]), "val": dict(variant_counts["val"]), "total_train": sum(variant_counts["train"].values()), "total_val": sum(variant_counts["val"].values())},
    }
    for name, rows in kept.items():
        stats["per_source"][name] = {
            "pipeline": dict(sorted(counts[name].items())), "final": len(rows), "train": split_counts["train"][name], "val": split_counts["val"][name],
            "total_tokens": token_stats([r["total_tokens"] for r in rows]), "reply_tokens": token_stats([r["reply_tokens"] for r in rows]),
        }
    stats["all"] = {"total_tokens": token_stats([r["total_tokens"] for r in all_rows]), "reply_tokens": token_stats([r["reply_tokens"] for r in all_rows]), "tokens_in_train": sum(r["total_tokens"] for r in all_rows if split_for(r["prompt"]) == "train")}
    examples = {}
    for name, rows in kept.items():
        chosen = random.Random(f"{SEED}-{name}").sample(rows, min(5, len(rows)))
        examples[name] = [{"prompt": r["prompt"], "reply": r["text"]} for r in chosen]
    stats["examples"] = examples
    atomic_write_json(project_path(REPORT_JSON), stats)
    atomic_write_json(project_path(MANIFEST), {
        "tokenizer": TOKENIZER_PATH, "tokenizer_sha256": sha256_text(project_path(TOKENIZER_PATH)),
        "sources": {name: {"dataset": SOURCES[name], "revision": REVISIONS[name]} for name in SOURCES},
        "goal_sets": {"dev_v2_1": DEV_PATH[0], "heldout": HELDOUT_PATH[0]}, "seed": SEED,
        "hash_note": "sha256 is over text with CRLF normalised to LF (phase10.common.sha256_text)",
        "files": {split: {"file": path, "records": sum(split_counts[split].values()), "sha256": sha256_text(project_path(path))} for split, path in paths.items()},
        "noformat_files": {split: {"file": path, "records": sum(variant_counts[split].values()), "sha256": sha256_text(project_path(path))} for split, path in variant_paths.items()},
    })
    print(json.dumps({k: stats[k] for k in ("total", "contamination", "verification_after_removal")}, indent=2), flush=True)
    for name, entry in stats["per_source"].items():
        print(name, entry["final"], entry["pipeline"], flush=True)


if __name__ == "__main__":
    main()

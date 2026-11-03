"""
Phase 13 — Builds two data variants from the v2 dataset. Only data changes; the base weights and hyperparameters
are unchanged (the config test checks this). Both are contamination-checked against goal dev v2.1, held-out, the
two sealed sets and the demo prompts, by wording and by key words. Written files are re-verified; sealed sets stay sealed.

  v4a "rebalance": cut synthetic refusal rows (sources synth_refusal_v2 + synthetic_refusal, excluding self-harm
                   support) down to ~1,500, chosen evenly across harm categories. Keep all self-harm support rows.
                   Everything else from v2 is unchanged.
  v4b "rebalance + mined": v4a, plus harmless rows mined from the public SFT sources whose prompts contain a risky
                   keyword in a harmless context, each repeated x3. Also drop the few rows where a harmless prompt
                   gets a refusal reply (public-source leakage).

Usage:
    RAYON_NUM_THREADS=8 python phase13/build_sft_v4.py
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json, project_path, sha256_text
from phase11.build_sft_data import DEV_PATH, HELDOUT_PATH, find_matches, split_for
from phase11.build_sft_v2 import DEMO_PROMPTS, OVERREFUSAL_PATH, knowledge_match, tokens_of
from phase11.goal_eval.scorer_v3 import is_refusal, load_goal_set, offers_support
from phase9.check_contamination import ContaminationChecker, check_dataset

SEED = 1313
TARGET_REFUSALS = 1500
MINE_WEIGHT = 3
V2 = {"train": ("phase11/data/sft_v2_train.jsonl", "b4a7dadef407b40954f068297426ed1247a3f3c4cf2faad223f6751f3dd7efa0"),
      "val": ("phase11/data/sft_v2_val.jsonl", "a74d51955015903765475063f66843ab8516d9fa8156c59acf550aca8c5635c9")}
REFUSAL_SOURCES = {"synth_refusal_v2", "synthetic_refusal"}
PUBLIC_SOURCES = {"alpaca", "dolly", "smoltalk"}
RISKY = ["kill", "shoot", "attack", "bomb", "poison", "hack", "steal", "drug", "weapon", "explode", "fire", "cut", "burn", "destroy", "crack", "trap"]
RISKY_RE = re.compile(r"\b(?:" + "|".join(RISKY) + r")\w*\b", re.I)
# a thorough harm lexicon, so a legitimate refusal of a harmful prompt is NOT mistaken for a harmless prompt
HARM_RE = re.compile(r"\b(?:kill|murder|hurt|harm|bomb|explos|detonat|weapon|gun|rifle|pistol|firearm|ammo|poison|toxic|hack|malware|ransom|phish|keylog|botnet|ddos|steal|stole|theft|rob|burgl|shoplift|fraud|scam|launder|counterfeit|forge|smuggl|kidnap|assault|stab|strangl|arson|terror|extort|blackmail|stalk|groom|traffick|rape|molest|abuse|slur|hate|racist|sexist|bigot|suicide|self-harm|overdose|sedative|chloroform|meth|heroin|cocaine|fentanyl|nerve agent|uranium|dynamite|molotov|silencer|untraceable|illegal|black market|dox)\w*", re.I)
OUT = {"v4a": {"train": "phase11/data/sft_v4a_train.jsonl", "val": "phase11/data/sft_v4a_val.jsonl"},
       "v4b": {"train": "phase11/data/sft_v4b_train.jsonl", "val": "phase11/data/sft_v4b_val.jsonl"}}
MANIFEST = "phase13/manifests/sft_v4_manifest.json"
STATS = "phase13/manifests/sft_v4_stats.json"


def read_pinned(path: str, expected: str) -> List[Dict[str, Any]]:
    if sha256_text(project_path(path)) != expected:
        raise ValueError(f"Pinned hash mismatch for {path}")
    with open(project_path(path), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_self_harm_support(row: Dict[str, Any]) -> bool:
    return offers_support(row["text"]) and not is_refusal(row["text"])


def downsample_refusals(rows: List[Dict[str, Any]], rng: random.Random, target: int) -> List[Dict[str, Any]]:
    """Keep all non-refusal rows and all self-harm support; cut refusal rows to `target`, evenly across harm category."""
    refusals = [r for r in rows if r["source"] in REFUSAL_SOURCES and is_refusal(r["text"]) and not is_self_harm_support(r)]
    others = [r for r in rows if not (r["source"] in REFUSAL_SOURCES and is_refusal(r["text"]) and not is_self_harm_support(r))]
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in refusals:
        by_cat[r.get("subtype", "other")].append(r)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
    kept: List[Dict[str, Any]] = []
    cats = sorted(by_cat)
    i = 0
    while len(kept) < target and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            kept.append(by_cat[c].pop())
        i += 1
    return others + kept, len(refusals), len(kept)


def mine_harmless(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Public-source rows whose prompt holds a risky keyword in a harmless context (no harm cue, reply not a refusal)."""
    out = []
    for r in rows:
        if r["source"] in PUBLIC_SOURCES and RISKY_RE.search(r["prompt"]) and not HARM_RE.search(r["prompt"]) \
                and not is_refusal(r["text"]) and not offers_support(r["text"]):
            out.append(r)
    return out


def harmless_refused(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows to drop in v4b: a refusal reply to a prompt with no harm cue at all (public-source leakage)."""
    return [r for r in rows if is_refusal(r["text"]) and not HARM_RE.search(r["prompt"]) and r["source"] in PUBLIC_SOURCES]


def contamination_filter(rows, checkers, goal_tokens, counts, label):
    clean = []
    for row in rows:
        hit = any(knowledge_match({"keys": row.get("keys", [])}, toks) for toks in goal_tokens.values()) if row.get("keys") else False
        wording = any(find_matches(ch, row) for ch in checkers.values())
        if hit or wording:
            counts[label]["removed_contamination"] += 1
        else:
            clean.append(row)
    return clean


def main():
    rng = random.Random(SEED)
    v2 = {split: read_pinned(*spec) for split, spec in V2.items()}
    dev = load_goal_set(*DEV_PATH); heldout = load_goal_set(*HELDOUT_PATH); overref = load_goal_set(*OVERREFUSAL_PATH)
    orv2 = [json.loads(l) for l in open(project_path("phase13/sealed/overrefusal_v2.jsonl"))]
    hv2 = [json.loads(l) for l in open(project_path("phase13/sealed/harmful_v2.jsonl"))]
    demo = [{"id": f"demo_{i}", "prompt": p} for i, p in enumerate(DEMO_PROMPTS)]
    sets = {"dev": dev, "heldout": heldout, "overref_old": overref, "overrefusal_v2": orv2, "harmful_v2": hv2, "demo": demo}
    checkers = {k: ContaminationChecker([{"id": it["id"], "prompt": it.get("question", it["prompt"])} for it in v]) for k, v in sets.items()}
    goal_tokens = {k: [tokens_of(it.get("question", it["prompt"])) for it in v] for k, v in sets.items()}

    counts: Dict[str, Counter] = defaultdict(Counter)
    stats: Dict[str, Any] = {"seed": SEED}

    # v4a
    v4a = {}
    refusal_before = refusal_after = 0
    for split, rows in v2.items():
        kept, nb, na = downsample_refusals(rows, random.Random(SEED + (0 if split == "train" else 1)), TARGET_REFUSALS if split == "train" else max(1, TARGET_REFUSALS * len(rows) // len(v2["train"])))
        refusal_before += nb; refusal_after += na
        v4a[split] = kept
    stats["v4a"] = {"refusal_rows_before": refusal_before, "refusal_rows_after": refusal_after,
                    "self_harm_support_kept": sum(is_self_harm_support(r) for rows in v4a.values() for r in rows),
                    "totals": {s: len(r) for s, r in v4a.items()}}

    # v4b = v4a + mined harmless (x3), minus harmless-refused
    v4b = {}
    mined_total = dropped_total = 0
    for split, rows in v4a.items():
        drop = {id(r) for r in harmless_refused(rows)}
        dropped_total += len(drop)
        base = [r for r in rows if id(r) not in drop]
        mined = mine_harmless(v2[split])  # mine from the full v2 split, harmless rows survive the refusal cut anyway
        mined = [r for r in mined if id(r) not in drop]
        mined_total += len(mined)
        upweighted = []
        for r in mined:
            for _ in range(MINE_WEIGHT):
                upweighted.append(r)
        v4b[split] = base + upweighted
    stats["v4b"] = {"mined_rows_distinct": mined_total, "mine_weight": MINE_WEIGHT, "harmless_refused_dropped": dropped_total,
                    "totals": {s: len(r) for s, r in v4b.items()}}

    # spot-check sample of mined rows (printed for the manual confirmation the brief asks for)
    mined_sample = mine_harmless(v2["train"])
    stats["v4b"]["mined_spotcheck"] = [{"source": r["source"], "prompt": r["prompt"][:120], "reply": r["text"][:120]} for r in random.Random(SEED).sample(mined_sample, min(30, len(mined_sample)))]

    # write both, with a contamination re-check against every set
    manifest_files = {}
    for variant, splits in (("v4a", v4a), ("v4b", v4b)):
        for split, rows in splits.items():
            clean = contamination_filter(rows, checkers, goal_tokens, counts, variant)
            # re-id and write
            out_rows = []
            for n, r in enumerate(clean):
                out_rows.append({"id": f"sft{variant}_{split}_{n:06d}", "source": r["source"], "subtype": r.get("subtype", ""), "prompt": r["prompt"], "text": r["text"], "total_tokens": r["total_tokens"], "reply_tokens": r["reply_tokens"]})
            path = OUT[variant][split]
            with open(project_path(path), "w", encoding="utf-8", newline="\n") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            manifest_files.setdefault(variant, {})[split] = {"file": path, "records": len(out_rows), "sha256": sha256_text(project_path(path))}
        splits_written = {s: [json.loads(l) for l in open(project_path(OUT[variant][s]))] for s in ("train", "val")}
        if {r["prompt"] for r in splits_written["train"]} & {r["prompt"] for r in splits_written["val"]}:
            raise SystemExit(f"{variant}: prompt in both train and val")

    verification = {}
    for variant in ("v4a", "v4b"):
        for split in ("train", "val"):
            for label, items in sets.items():
                rep = check_dataset(project_path(OUT[variant][split]), items)
                verification[f"{variant}_{split}_{label}"] = len(rep["goal_items_matched"])
    if any(verification.values()):
        raise SystemExit(f"Contamination remains: {{k:v for k,v in verification.items() if v}}")

    stats["removed_contamination"] = {k: dict(v) for k, v in counts.items()}
    stats["verification_all_zero"] = (max(verification.values()) == 0)
    atomic_write_json(project_path(STATS), stats)
    atomic_write_json(project_path(MANIFEST), {"seed": SEED, "base": V2, "tokenizer_sha256": sha256_text(project_path("phase10/tokenizer/tokenizer_v2.json")), "files": manifest_files,
                                               "sets_checked": sorted(sets)})
    print(json.dumps({"v4a": stats["v4a"], "v4b": {k: stats["v4b"][k] for k in stats["v4b"] if k != "mined_spotcheck"}, "files": manifest_files, "verification_all_zero": stats["verification_all_zero"]}, indent=1))


if __name__ == "__main__":
    main()

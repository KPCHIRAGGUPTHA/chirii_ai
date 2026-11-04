"""
Phase 18 — Scores GSM8K and TriviaQA on the FIXED subsets in subsets.json (500 and 1,000 items, seed 18), for every
model, from lm-eval's per-item sample logs. Writes results/<model>/subset_scores.json.

Why: generation on the full sets would have taken far longer than the time available (see BENCHMARKS.md). Every
model is scored on exactly the same questions. Runs made on the full set are re-scored here on the subset items
only; runs made directly on the subset (lm-eval --samples) cover exactly those items. Either way the score is the mean
of lm-eval's own per-item exact_match, so nothing is re-judged. Standard error: lm-eval's formula (sample standard
deviation / sqrt(n)).

Usage (from phase18/):  python subset_scores.py
"""

import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FILTERS = {"gsm8k": ("flexible-extract", "strict-match"), "triviaqa": ("remove_whitespace",)}


def mean_stderr(values):
    n = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1)) if n > 1 else float("nan")
    return mu, sd / math.sqrt(n)


def score_step(step_dir: str, task: str, subset: set) -> dict:
    """Reads the full lm-eval sample log if present (and writes the slim subset log next to the step), otherwise the
    committed slim log results/<model>/subset_samples_<step>.jsonl. Both give the same per-item verdicts."""
    slim = os.path.join(os.path.dirname(step_dir.rstrip(os.sep)), f"subset_samples_{os.path.basename(step_dir.rstrip(os.sep))}.jsonl")
    files = sorted(glob.glob(os.path.join(step_dir, "**", f"samples_{task}_*.jsonl"), recursive=True))
    if files:
        rows = [json.loads(line) for line in open(files[-1], encoding="utf-8")]
        with open(slim, "w", encoding="utf-8") as f:
            for r in rows:
                if r["doc_id"] in subset:
                    f.write(json.dumps({"doc_id": r["doc_id"], "filter": r["filter"], "response": r["resps"][0][0],
                                        "filtered": r["filtered_resps"][0] if r["filtered_resps"] else None,
                                        "target": r["target"], "exact_match": r["exact_match"]}, ensure_ascii=False) + "\n")
        source = files[-1]
    elif os.path.exists(slim):
        rows = [json.loads(line) for line in open(slim, encoding="utf-8")]
        source = slim
    else:
        return None
    out = {"source": os.path.relpath(source, HERE), "logged_items": len({r["doc_id"] for r in rows})}
    for flt in FILTERS[task]:
        by_doc = {r["doc_id"]: float(r["exact_match"]) for r in rows if r["filter"] == flt}
        missing = subset - set(by_doc)
        if missing:
            raise ValueError(f"{source}: {len(missing)} subset items not in the log")
        values = [by_doc[i] for i in sorted(subset)]
        mu, se = mean_stderr(values)
        out[flt] = {"value": mu, "stderr": se, "n": len(values)}
    return out


def main():
    subsets = {k: set(v) for k, v in json.load(open(os.path.join(HERE, "subsets.json")))["samples"].items()}
    for model_dir in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        name = os.path.basename(model_dir)
        if not os.path.isdir(model_dir) or name == "validation":
            continue
        scores = {}
        for step in ("gsm8k", "gsm8k_no_calculator", "triviaqa"):
            step_dir = os.path.join(model_dir, step)
            if os.path.isdir(step_dir) and os.path.exists(step_dir + ".done"):
                task = "triviaqa" if step == "triviaqa" else "gsm8k"
                result = score_step(step_dir, task, subsets[task])
                if result:
                    scores[step] = result
        if scores:
            json.dump(scores, open(os.path.join(model_dir, "subset_scores.json"), "w"), indent=2)
            print(name, {s: {f: round(100 * v["value"], 1) for f, v in r.items() if isinstance(v, dict)} for s, r in scores.items()})


if __name__ == "__main__":
    main()

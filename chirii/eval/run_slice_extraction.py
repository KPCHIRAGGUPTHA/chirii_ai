"""
Demo slice measurement: rules only against hybrid (rules, then Qwen with code validation) on a set of 20 new
sentences. Each set's first run is its reported first-try result; later runs on the same set are development numbers.

    set A  slice_fresh_sentences.py    first try 21 Nov (EXTRACTION_A_first_try.md); then development data
    set B  slice_fresh_sentences_b.py  written after the fixes that set A prompted, first try after them

Needs llama-server running (see chirii/README.md). Writes chirii/results/slice/EXTRACTION_<label>.md and .json.
    python -m chirii.eval.run_slice_extraction --set b --label B_first_try
"""

import argparse
import json
import platform
from datetime import date
from pathlib import Path

from chirii.app.config import REPO_ROOT, Settings
from chirii.app.llm.client import LLMClient
from chirii.app.memory.extract import extract
from chirii.app.memory.hybrid import HybridExtractor, needs_model
from chirii.eval.slice_fresh_sentences import SAID_ON as SAID_ON_A, SLICE_FRESH
from chirii.eval.slice_fresh_sentences_b import SAID_ON as SAID_ON_B, SLICE_FRESH_B

SETS = {"a": (SLICE_FRESH, SAID_ON_A, "chirii/eval/slice_fresh_sentences.py"), "b": (SLICE_FRESH_B, SAID_ON_B, "chirii/eval/slice_fresh_sentences_b.py")}

OUT = REPO_ROOT / "chirii" / "results" / "slice"


class RecordingModel:
    """Passes calls through to the real model and keeps the raw replies for the report."""

    def __init__(self, model):
        self.model, self.replies = model, []

    def chat(self, *args, **kwargs):
        reply = self.model.chat(*args, **kwargs)
        self.replies.append(reply)
        return reply


def wrong_fields(item, expected):
    got = item.to_dict()
    return {k: got.get(k) for k, v in expected.items() if got.get(k) != v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument("--label", required=True, help="e.g. B_first_try or A_after_fixes")
    parser.add_argument("--note", default="", help="one line for the report saying what kind of run this is")
    args = parser.parse_args()
    sentences, said_on, source = SETS[args.set]
    settings = Settings(secret_key="slice-measurement-secret-not-for-real-data-00000")
    model = RecordingModel(LLMClient(settings.llm_base_url, settings.llm_model, timeout=120))
    hybrid = HybridExtractor(model)
    rows = []
    for text, expected in sentences:
        rules = extract(text, said_on)
        before = len(model.replies)
        mixed = hybrid(text, said_on)
        used = len(model.replies) > before
        rows.append({
            "sentence": text, "expected": expected,
            "rules_wrong": wrong_fields(rules, expected), "hybrid_wrong": wrong_fields(mixed, expected),
            "model_called": used, "hybrid_source": mixed.source, "needs_review": mixed.needs_review,
            "model_reply": model.replies[-1].text if used else None,
            "model_ms": model.replies[-1].duration_ms if used else None,
        })
    n = len(rows)
    called = [r for r in rows if r["model_called"]]
    summary = {
        "rules_only_fully_right": sum(not r["rules_wrong"] for r in rows),
        "hybrid_fully_right": sum(not r["hybrid_wrong"] for r in rows),
        "model_called": len(called),
        "model_accepted": sum(r["hybrid_source"] == "model" for r in rows),
        "model_rejected": sum("rejected" in r["hybrid_source"] for r in rows),
        "marked_for_review": sum(r["needs_review"] for r in rows),
        "hybrid_wrong_and_not_marked": sum(bool(r["hybrid_wrong"]) and not r["needs_review"] for r in rows),
        "median_model_ms": sorted(r["model_ms"] for r in called)[len(called) // 2] if called else None,
    }
    report = {"set": args.set, "label": args.label, "note": args.note, "when": date.today().isoformat(),
              "machine": f"{platform.system()} {platform.machine()}, CPU", "model": settings.llm_model, "sentences": n,
              "summary": summary, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"extraction_{args.label}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = summary
    lines = [
        f"# Demo slice: rules only against hybrid extraction ({args.label})", "",
        f"{n} sentences from `{source}`, run on {report['when']} with {report['model']} on the laptop CPU, temperature 0.",
        f"**{args.note}**", "",
        "| | Result |", "| :--- | :--- |",
        f"| Rules only: fully right | {s['rules_only_fully_right']}/{n} |",
        f"| Hybrid: fully right | {s['hybrid_fully_right']}/{n} |",
        f"| Qwen called (rules gave up) | {s['model_called']}/{n} |",
        f"| Qwen's item accepted by the code check | {s['model_accepted']} |",
        f"| Qwen's item rejected by the code check (kept as the rules' item, marked for review) | {s['model_rejected']} |",
        f"| Hybrid wrong but **not** marked for the user to check | {s['hybrid_wrong_and_not_marked']} |",
        f"| Median time for one Qwen call | {s['median_model_ms']} ms |", "",
        "| Sentence | Rules wrong on | Qwen called | Hybrid result | Hybrid wrong on |", "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        lines.append(f"| {r['sentence']} | {r['rules_wrong'] or '-'} | {'yes' if r['model_called'] else 'no'} | {r['hybrid_source']}"
                     f"{' (check)' if r['needs_review'] else ''} | {r['hybrid_wrong'] or '-'} |")
    lines += ["", f"Raw Qwen replies are in `extraction_{args.label}.json`."]
    (OUT / f"EXTRACTION_{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

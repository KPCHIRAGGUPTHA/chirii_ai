"""
S3: the first official run of the 50-task development eval (chirii/CRITERIA.md, "S3 method").

    a     before: full retrieval, no fact-check guard (comparison only)
    b     ship:   trimmed retrieval with the fact-check guard (the S7 decision is made on this run)
    none  plain Qwen: today's date and the question, no memory, no guard (comparison only)

Settings: Qwen2.5-1.5B-Instruct Q8_0 through llama-server, temperature 0, at most 256 new tokens, one run. Each task
gets a fresh memory store holding that person's memories, said in order and all confirmed (the eval measures
answering, not the "Got it" card). The 5 Hinglish tasks are run too and reported unscored.

Writes chirii/results/s3/replies_<run>.jsonl (every reply, the memory lines given, the guard outcome, the marker's
verdict) and S3_AUTO.md (automatic scores only; the hand check and failure labels are added separately).
Needs llama-server running.
    python -m chirii.eval.run_s3 --runs a b none
"""

import argparse
import json
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

from chirii.app.assistant import Assistant
from chirii.app.config import REPO_ROOT, Settings
from chirii.app.formatting import say_date
from chirii.app.llm.client import LLMClient
from chirii.app.memory.embeddings import FastEmbedder
from chirii.app.memory.hybrid import HybridExtractor
from chirii.app.memory.store import MemoryStore
from chirii.eval.run_s2_memory import base_texts
from chirii.eval.s2_gold import load_tasks, said_on_for
from chirii.eval.scoring import contains, score_task

OUT = REPO_ROOT / "chirii" / "results" / "s3"
SECRET = "s3-eval-run-secret-not-used-for-real-data-0000000"
CONFIGS = {"a": {"guard": False, "trimmed": False, "routed": False}, "b": {"guard": True, "trimmed": True, "routed": False},
           "c": {"guard": True, "trimmed": True, "routed": True}, "d": {"guard": True, "trimmed": True, "routed": True}}


def plain_qwen(model, task):
    today = date.fromisoformat(task["today"])
    messages = [{"role": "system", "content": f"You are a helpful personal assistant. Today is {say_date(today.isoformat())}."},
                {"role": "user", "content": task["question"]}]
    started = time.perf_counter()
    text = model.chat(messages, max_tokens=256, purpose="eval_plain").text
    return {"reply": text, "duration_ms": round((time.perf_counter() - started) * 1000)}


def with_memory(run, model, embedder, task, base, root):
    store = MemoryStore(root, f"{task['id']}-{run}@eval", SECRET, embedder, extractor=HybridExtractor(model))
    model_reads = 0
    for entry in sorted(task["memory"], key=lambda e: said_on_for(e, task, base)):
        result = store.add(entry["said"], said_on_for(entry, task, base))
        model_reads += result["item"]["source"] != "rules"
        store.confirm(result["item"]["id"])
    answer = Assistant(store, model, **CONFIGS[run]).ask(task["question"], date.fromisoformat(task["today"]))
    today = date.fromisoformat(task["today"])
    lines = " ".join(answer.facts)
    needed = task["rules"].get("must_include", [])
    store.delete_all()
    return {"reply": answer.text, "duration_ms": answer.duration_ms, "guard": answer.guard, "first_answer": answer.first_answer,
            "guard_problems": answer.problems, "facts": answer.facts, "memory_read_by_model": model_reads, "route": answer.route,
            "needed_not_in_lines": [f for f in needed if not contains(lines, f, today)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", choices=["a", "b", "c", "d", "none"], default=["a", "b", "none"])
    parser.add_argument("--out-dir", default=str(OUT), help="where replies and the summary go (the GPU job uses its own folder)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    settings = Settings(secret_key=SECRET)
    model = LLMClient(settings.llm_base_url, settings.llm_model, timeout=300)
    embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    tasks = load_tasks()
    base = base_texts([t for t in tasks if t["scored"]])
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run in args.runs:
        root = Path(tempfile.mkdtemp(prefix=f"chirii-s3-{run}-"))
        rows = []
        try:
            for task in tasks:
                row = plain_qwen(model, task) if run == "none" else with_memory(run, model, embedder, task, base, root)
                verdict = score_task(task, row["reply"])
                rows.append({"id": task["id"], "type": task["type"], "persona": task["persona"], "scored": task["scored"],
                             "question": task["question"], **row, "passed": verdict["passed"], "failures": verdict["failures"],
                             "invented": verdict["invented"]})
                print(run, task["id"], "PASS" if verdict["passed"] else "fail", row.get("guard", ""), flush=True)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        with open(out / f"replies_{run}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        scored = [r for r in rows if r["scored"]]
        unknown = [r for r in scored if r["type"] == "unknown"]
        summary[run] = {
            "A_overall": sum(r["passed"] for r in scored), "B_unknown": sum(r["passed"] for r in unknown),
            "C_invented": sum(r["invented"] for r in scored),
            "guard": {k: sum(r.get("guard") == k for r in scored) for k in ("pass", "retry", "fallback", "code")} if run in ("b", "c", "d") else None,
            "median_ms": sorted(r["duration_ms"] for r in scored)[len(scored) // 2],
        }
    name = "s3_auto.json" if sorted(args.runs) == ["a", "b", "none"] and out == OUT else "auto_" + "_".join(args.runs) + ".json"  # never overwrite the S3 summary
    (out / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
The 500-item scale check (CRITERIA.md), run once: the same 30 questions on a 500-memory store and a 40-memory store,
plus storage timings and 3 list questions. Writes chirii/results/scale/replies.jsonl and summary.json.
    python -m chirii.eval.run_scale
"""

import json
import shutil
import statistics
import tempfile
import time
from datetime import date
from pathlib import Path

from chirii.app.assistant import Assistant
from chirii.app.config import REPO_ROOT, Settings
from chirii.app.llm.client import LLMClient, LLMError
from chirii.app.memory import store as store_module
from chirii.app.memory.embeddings import FastEmbedder
from chirii.app.memory.store import MemoryStore
from chirii.app.memory.vault import new_key
from chirii.eval.scoring import score_task

EVAL = REPO_ROOT / "chirii" / "eval"
OUT = REPO_ROOT / "chirii" / "results" / "scale"
SECRET = "scale-eval-secret-not-used-for-real-data-0000000000"


def p95(values):
    values = sorted(values)
    return values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]


def main():
    data = json.loads((EVAL / "scale_set.json").read_text(encoding="utf-8"))
    settings = Settings(secret_key=SECRET)
    model = LLMClient(settings.llm_base_url, settings.llm_model, timeout=300)
    embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    key = new_key()                                       # a throwaway key for this run only
    today, said_on = date.fromisoformat(data["today"]), date.fromisoformat(data["said_on"])
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="chirii-scale-"))
    rows, storage, lists = [], {}, []
    try:
        for size in ("small", "large"):
            memory = data["stores"][size]
            store = MemoryStore(root, f"scale-{size}", SECRET, embedder, data_key=key)
            add_times = []
            for entry in memory:
                started = time.perf_counter()
                result = store.add(entry["said"], said_on)
                store.confirm(result["item"]["id"])
                add_times.append(time.perf_counter() - started)
            store_module._open.clear()                        # as after a restart
            started = time.perf_counter()
            store = MemoryStore(root, f"scale-{size}", SECRET, embedder, data_key=key)
            open_s = time.perf_counter() - started
            storage[size] = {"items": len(store.items()), "open_s": round(open_s, 3),
                             "add_last_50_median_s": round(statistics.median(add_times[-50:]), 3),
                             "add_last_50_max_s": round(max(add_times[-50:]), 3),
                             "file_kb": round(store.vault.path.stat().st_size / 1024, 1), "consistent": store.is_consistent()}
            print(size, storage[size], flush=True)
            for q in data["questions"]:
                a = Assistant(store, model).ask(q["question"], today)
                task = {**q, "memory": memory, "allow": []}
                v = score_task(task, a.text)
                rows.append({"id": q["id"], "store": size, "question": q["question"], "reply": a.text, "route": a.route,
                             "guard": a.guard, "problems": a.problems, "duration_ms": a.duration_ms, "passed": v["passed"],
                             "failures": v["failures"]})
                print(size, q["id"], "PASS" if v["passed"] else "fail", a.duration_ms, flush=True)
            if size == "large":
                for question in data["list_questions"]:
                    started = time.perf_counter()
                    try:
                        a = Assistant(store, model).ask(question, today)
                        lists.append({"question": question, "reply": a.text, "route": a.route, "lines": a.text.count("\n") + 1,
                                      "seconds": round(time.perf_counter() - started, 1), "error": None})
                    except LLMError as e:
                        lists.append({"question": question, "reply": None, "error": type(e).__name__,
                                      "seconds": round(time.perf_counter() - started, 1)})
                    print("list", question, lists[-1].get("lines"), lists[-1]["seconds"], flush=True)
            store.delete_all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    (OUT / "replies.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    summary = {"storage": storage, "lists": lists}
    for size in ("small", "large"):
        mine = [r for r in rows if r["store"] == size]
        secs = [r["duration_ms"] / 1000 for r in mine]
        summary[size] = {"passed": sum(r["passed"] for r in mine), "of": len(mine),
                         "not_saved_passed": sum(r["passed"] for r in mine if r["id"] >= "SC-26"),
                         "median_s": round(statistics.median(secs), 1), "p95_s": round(p95(secs), 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "lists"}, indent=2))


if __name__ == "__main__":
    main()

"""
Run (e3): development set 3, once, on the S4c system (CRITERIA.md). Memory tasks, general-chat tasks and router tests.

Same settings as before: Qwen2.5-1.5B-Instruct Q8_0 through llama-server, temperature 0, one run. Each task gets a
fresh memory store with the person's memories, said in order and confirmed. Writes chirii/results/dev3/: replies for
memory and chat tasks, router-test results, and a summary.
    python -m chirii.eval.run_e3
"""

import json
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

from chirii.app.assistant import Assistant
from chirii.app.config import REPO_ROOT, Settings
from chirii.app.llm.client import LLMClient
from chirii.app.memory.embeddings import FastEmbedder
from chirii.app.memory.hybrid import HybridExtractor
from chirii.app.memory.store import MemoryStore
from chirii.app.router import route
from chirii.eval.chat_scoring import score_chat
from chirii.eval.run_s2_memory import base_texts
from chirii.eval.s2_gold import said_on_for
from chirii.eval.scoring import score_task

EVAL = REPO_ROOT / "chirii" / "eval"
OUT = REPO_ROOT / "chirii" / "results" / "dev3"
SECRET = "e3-eval-run-secret-not-used-for-real-data-00000000"


def load(name):
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()]


def fill(store, task, base, model_reads):
    for entry in sorted(task["memory"], key=lambda e: said_on_for(e, task, base)):
        result = store.add(entry["said"], said_on_for(entry, task, base))
        model_reads.append(result["item"]["source"] != "rules")
        store.confirm(result["item"]["id"])


def main():
    settings = Settings(secret_key=SECRET)
    model = LLMClient(settings.llm_base_url, settings.llm_model, timeout=300)
    embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    mem, chats, routers = load("dev3_memory.jsonl"), load("dev3_chat.jsonl"), load("dev3_router.jsonl")
    base = base_texts(mem)
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="chirii-e3-"))
    reads = []
    try:
        rows = []
        for t in mem:
            store = MemoryStore(root, f"{t['id']}@e3", SECRET, embedder, extractor=HybridExtractor(model))
            fill(store, t, base, reads)
            a = Assistant(store, model).ask(t["question"], date.fromisoformat(t["today"]))
            v = score_task(t, a.text)
            rows.append({"id": t["id"], "type": t["type"], "question": t["question"], "reply": a.text, "route": a.route,
                         "expect_route": t["expect_route"], "guard": a.guard, "label": a.label, "first_answer": a.first_answer,
                         "guard_problems": a.problems, "facts": a.facts, "duration_ms": a.duration_ms,
                         "passed": v["passed"], "failures": v["failures"], "invented": v["invented"]})
            print("memory", t["id"], "PASS" if v["passed"] else "fail", a.route, flush=True)
            store.delete_all()
        (OUT / "replies_memory.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        chat_rows = []
        for t in chats:
            store = MemoryStore(root, f"{t['id']}@e3", SECRET, embedder, extractor=HybridExtractor(model))
            if t["persona"]:
                fill(store, dict(t, scored=True), base, reads)
            a = Assistant(store, model).ask(t["question"], date.fromisoformat(t["today"]))
            v = score_chat(t, a.text)
            chat_rows.append({"id": t["id"], "kind": t["kind"], "question": t["question"], "reply": a.text, "route": a.route,
                              "expect_route": t["expect_route"], "guard": a.guard, "label": a.label, "duration_ms": a.duration_ms,
                              "passed": v["passed"], "failures": v["failures"]})
            print("chat", t["id"], "PASS" if v["passed"] else "fail", a.route, flush=True)
            store.delete_all()
        (OUT / "replies_chat.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in chat_rows) + "\n", encoding="utf-8")
        router_rows = []
        for t in routers:
            store = MemoryStore(root, f"{t['id']}@e3", SECRET, embedder, extractor=HybridExtractor(model))
            fill(store, dict(t, scored=True), base, reads)
            started = time.perf_counter()
            r = route(t["question"], date.fromisoformat(t["today"]), model, store.items(include_updates=False))
            got_period = [r.window[0].isoformat(), r.window[1].isoformat()] if r.window else None
            router_rows.append({"id": t["id"], "question": t["question"], "expect_route": t["expect_route"], "route": r.kind,
                                "source": r.source, "expect_period": t["period"], "period": got_period,
                                "route_ok": r.kind == t["expect_route"], "period_ok": (t["period"] is None) or got_period == t["period"],
                                "duration_ms": round((time.perf_counter() - started) * 1000)})
            print("router", t["id"], r.kind, r.source, flush=True)
            store.delete_all()
        (OUT / "router_tests.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in router_rows) + "\n", encoding="utf-8")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    summary = {
        "memory": {"passed": sum(r["passed"] for r in rows), "of": len(rows)},
        "chat": {k: {"passed": sum(r["passed"] for r in chat_rows if r["kind"] == k), "of": sum(r["kind"] == k for r in chat_rows)}
                 for k in ("maths", "general", "writing")},
        "router_tests": {"route_ok": sum(r["route_ok"] for r in router_rows), "period_ok": sum(r["period_ok"] for r in router_rows), "of": len(router_rows)},
        "route_accuracy_all": {"ok": sum(r["route"] == r["expect_route"] for r in rows + chat_rows) + sum(r["route_ok"] for r in router_rows),
                               "of": len(rows) + len(chat_rows) + len(router_rows)},
        "memory_sentences_read_by_model": sum(reads),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Run (e4): development set 4, once, on the S4d + S4e system (CRITERIA.md). Memory, people, maths and general-knowledge
tasks, with the route and answer label of each.

Same settings as before: Qwen2.5-1.5B-Instruct Q8_0 through llama-server, temperature 0, one run. Writes
chirii/results/dev4/replies.jsonl and summary.json.
    python -m chirii.eval.run_e4
"""

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

from chirii.app.assistant import Assistant
from chirii.app.config import REPO_ROOT, Settings
from chirii.app.llm.client import LLMClient
from chirii.app.memory.embeddings import FastEmbedder
from chirii.app.memory.hybrid import HybridExtractor
from chirii.app.memory.store import MemoryStore
from chirii.eval.chat_scoring import score_chat
from chirii.eval.run_s2_memory import base_texts
from chirii.eval.s2_gold import said_on_for
from chirii.eval.scoring import score_task

EVAL = REPO_ROOT / "chirii" / "eval"
OUT = REPO_ROOT / "chirii" / "results" / "dev4"
SECRET = "e4-eval-run-secret-not-used-for-real-data-00000000"


def load(name):
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()]


def main():
    settings = Settings(secret_key=SECRET)
    model = LLMClient(settings.llm_base_url, settings.llm_model, timeout=300)
    embedder = FastEmbedder(settings.embed_model, settings.embed_cache_dir)
    groups = {name: load(f"dev4_{name}.jsonl") for name in ("memory", "people", "maths", "gk")}
    base = base_texts(groups["memory"] + groups["people"])
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="chirii-e4-"))
    rows = []
    try:
        for group, tasks in groups.items():
            for t in tasks:
                store = MemoryStore(root, f"{t['id']}@e4", SECRET, embedder, extractor=HybridExtractor(model))
                for entry in sorted(t["memory"], key=lambda e: said_on_for(e, dict(t, scored=True), base)):
                    result = store.add(entry["said"], said_on_for(entry, dict(t, scored=True), base))
                    store.confirm(result["item"]["id"])
                a = Assistant(store, model).ask(t["question"], date.fromisoformat(t["today"]))
                v = score_task(t, a.text) if group in ("memory", "people") else score_chat(t, a.text)
                rows.append({"id": t["id"], "group": group, "question": t["question"], "reply": a.text, "route": a.route,
                             "expect_route": t["expect_route"], "label": a.label, "expect_label": t["expect_label"], "guard": a.guard,
                             "guard_problems": a.problems, "duration_ms": a.duration_ms, "passed": v["passed"], "failures": v["failures"]})
                print(group, t["id"], "PASS" if v["passed"] else "fail", a.route, a.label, flush=True)
                store.delete_all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    (OUT / "replies.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    summary = {g: {"passed": sum(r["passed"] for r in rows if r["group"] == g), "of": sum(r["group"] == g for r in rows)} for g in groups}
    summary["route_ok"] = {"ok": sum(r["route"] == r["expect_route"] for r in rows), "of": len(rows)}
    summary["label_ok"] = {"ok": sum(r["label"] == r["expect_label"] for r in rows), "of": len(rows)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

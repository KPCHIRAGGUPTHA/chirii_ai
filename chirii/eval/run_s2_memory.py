"""
S2 measurement: how well the memory store does on the S1 development memories, before any language model is involved.

    1. Parsing: each of the 66 memory sentences, and the 22 fresh sentences, compared field by field with the intended item.
    2. Updates: for the 8 update tasks, whether the update changed the right item in the right way.
    3. Search by meaning: for the 8 meaning-based questions, where the needed sentence ranks (top 1, top 3).
    4. Date windows: for 6 date-window tasks, whether the agenda for the window holds what it must and nothing it mustn't.
    5. Delete all my data, and timings.

Writes chirii/results/s2/S2_MEMORY.md and s2_memory.json. Usage:
    python chirii/eval/run_s2_memory.py [--embedder fast|hash]
"""

import argparse
import json
import platform
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path

from chirii.app.config import REPO_ROOT, Settings
from chirii.app.memory.embeddings import FastEmbedder, HashEmbedder
from chirii.app.memory.extract import extract
from chirii.app.memory.store import MemoryStore
from chirii.eval.s2_fresh_sentences import FRESH, SAID_ON
from chirii.eval.s2_gold import BASE_SAID_ON, SEMANTIC_TARGETS, UPDATE_TARGETS, load_tasks, said_on_for, sentences

FIELDS = ["kind", "date", "end_date", "time", "end_time", "amount", "day_of_month", "weekdays", "other_dates", "other_times", "other_amounts"]
WINDOWS = {"S1-03": ("2026-12-07", "2026-12-14"), "S1-04": ("2026-12-24", "2026-12-31"), "S1-13": ("2026-12-14", "2026-12-20"),
           "S1-14": ("2026-12-24", "2026-12-31"), "S1-23": ("2026-12-10", "2026-12-15"), "S1-33": ("2026-12-14", "2026-12-20")}
EXPECTED_UPDATES = {  # what the target item should look like after the update
    "S1-08": {"status": "paid"}, "S1-09": {"date": "2026-12-11", "time": "19:00"}, "S1-18": {"status": "paid"},
    "S1-19": {"date": "2026-12-22"}, "S1-28": {"remaining": 1450, "status": "active"}, "S1-29": {"date": "2026-12-12"},
    "S1-37": {"date": "2026-12-21", "time": "11:00"}, "S1-48": {"status": "paid"},
}
SECRET = "s2-measurement-run-secret-not-used-for-real-data-000"
# New questions that share no key word with the note they need, so only meaning can find it. Written before the first
# run; they test search, not the eval. (person, question, phrase from the needed sentence)
PARAPHRASES = [
    ("sharada", "What exercise was I advised?", "Dr. Ramaswamy said"),
    ("lakshmi", "What groceries for the harvest festival?", "Shopping list for Pongal"),
    ("arjun", "What's important for the database paper?", "Note from DBMS class"),
    ("arjun", "Where is my railway ticket number?", "The PNR is saved"),
    ("ramesh", "Has the dairy supplier changed his rates?", "new milk vendor"),
    ("priya", "When does the software launch happen?", "the release has moved"),
    ("sharada", "What present should I buy for my grandchild?", "Ananya's birthday"),
    ("lakshmi", "What papers should I take to the ophthalmologist?", "eye check-up"),
]


def base_texts(tasks):
    by_persona = {}
    for t in tasks:
        by_persona.setdefault(t["persona"], []).append(set(e["said"] for e in t["memory"]))
    return set().union(*(set.intersection(*sets) for sets in by_persona.values()))


def load_task_memory(task, root, embedder, base):
    store = MemoryStore(root, f"{task['id']}@eval", SECRET, embedder)
    results = []
    for entry in sorted(task["memory"], key=lambda e: said_on_for(e, task, base)):
        results.append((entry, store.add(entry["said"], said_on_for(entry, task, base))))
    return store, results


def measure_parsing():
    dev, fresh = sentences(), []
    field_hits = {f: [0, 0] for f in FIELDS}
    wrong = []
    for s in dev:
        item = extract(s["said"], s["said_on"]).to_dict()
        gold = dict(s["gold"], kind=s["gold"]["type"])
        ok = True
        for name in FIELDS:
            if name in gold:
                field_hits[name][1] += 1
                if item[name] == gold[name]:
                    field_hits[name][0] += 1
                else:
                    ok = False
        if not ok:
            wrong.append(s["said"])
    for text, want in FRESH:
        item = extract(text, SAID_ON).to_dict()
        fresh.append(all(item.get(k) == v for k, v in want.items()))
    return {"dev_sentences": len(dev), "dev_fully_right": len(dev) - len(wrong), "dev_wrong": wrong,
            "dev_fields": {k: f"{h}/{n}" for k, (h, n) in field_hits.items() if n},
            "fresh_sentences": len(FRESH), "fresh_fully_right_now": sum(fresh), "fresh_fully_right_first_try": 20}


def measure_updates(tasks, root, embedder, base):
    rows = []
    for task_id, phrase in UPDATE_TARGETS.items():
        task = next(t for t in tasks if t["id"] == task_id)
        store, results = load_task_memory(task, root, embedder, base)
        target = next(i for i in store.items() if phrase in i["text"])
        update = next(r for e, r in results if r["item"]["kind"] == "update")
        after = store.get(target["id"])
        changed_right = all(after.get(k) == v for k, v in EXPECTED_UPDATES[task_id].items())
        rows.append({"task": task_id, "linked_right": update["linked_to"] == target["id"], "changed_right": changed_right,
                     "linked_to": (store.get(update["linked_to"]) or {}).get("text", "")[:60] if update["linked_to"] else None})
        store.delete_all()
    return rows


def measure_search(tasks, root, embedder, base):
    rows = []
    for task_id, phrase in SEMANTIC_TARGETS.items():
        task = next(t for t in tasks if t["id"] == task_id)
        store, _ = load_task_memory(task, root, embedder, base)
        hits = store.search(task["question"], k=len(store.items()))
        rank = next((i + 1 for i, h in enumerate(hits) if phrase in h["item"]["text"]), None)
        rows.append({"task": task_id, "question": task["question"], "rank": rank, "top1": hits[0]["item"]["text"][:70]})
        store.delete_all()
    return rows


def measure_paraphrases(tasks, root, embedder, base):
    rows = []
    for persona, question, phrase in PARAPHRASES:
        task = next(t for t in tasks if t["persona"] == persona)
        store = MemoryStore(root, f"{persona}-paraphrase@eval", SECRET, embedder)
        for entry in task["memory"]:
            if entry["said"] in base:
                store.add(entry["said"], BASE_SAID_ON)
        hits = store.search(question, k=len(store.items()))
        rank = next((i + 1 for i, h in enumerate(hits) if phrase in h["item"]["text"]), None)
        rows.append({"question": question, "rank": rank, "top1": hits[0]["item"]["text"][:70]})
        store.delete_all()
    return rows


def measure_windows(tasks, root, embedder, base):
    rows = []
    for task_id, (start, end) in WINDOWS.items():
        task = next(t for t in tasks if t["id"] == task_id)
        store, _ = load_task_memory(task, root, embedder, base)
        agenda = store.agenda(date.fromisoformat(start), date.fromisoformat(end))
        amounts = {e["item"]["amount"] for e in agenda}
        days = {e["day"] for e in agenda} | {e["item"]["date"] for e in agenda}
        texts = " ".join(e["item"]["text"] for e in agenda)
        missing, extra = [], []
        for fact in task["rules"].get("must_include", []):
            if "amount" in fact and fact["amount"] not in amounts: missing.append(fact)
            if "date" in fact and fact["date"] not in days: missing.append(fact)
            if "text" in fact and not any(p.lower() in texts.lower() for p in fact["text"]): missing.append(fact)
        for fact in task["rules"].get("must_not", []):
            if "amount" in fact and fact["amount"] in amounts: extra.append(fact)
            if "date" in fact and fact["date"] in days: extra.append(fact)
        rows.append({"task": task_id, "window": f"{start} to {end}", "entries": len(agenda), "missing": missing, "should_not_be_there": extra})
        store.delete_all()
    return rows


def measure_delete_and_speed(tasks, root, embedder, base):
    task = next(t for t in tasks if t["id"] == "S1-01")
    started = time.perf_counter()
    store, _ = load_task_memory(task, root, embedder, base)
    add_ms = (time.perf_counter() - started) * 1000 / len(task["memory"])
    started = time.perf_counter()
    for _ in range(20):
        store.search("When is my electricity bill due?")
    search_ms = (time.perf_counter() - started) * 1000 / 20
    size_kb = sum(p.stat().st_size for p in store.dir.rglob("*") if p.is_file()) / 1024
    consistent = store.is_consistent()
    counts = store.delete_all()
    return {"add_ms_per_sentence": round(add_ms, 1), "search_ms": round(search_ms, 1), "folder_kb_for_9_items": round(size_kb, 1),
            "index_matches_db": consistent, "deleted": counts, "folder_gone": not store.dir.exists()}


def build_markdown(report):
    p, u, s, w, d = report["parsing"], report["updates"], report["search"], report["windows"], report["delete_and_speed"]
    top1 = sum(1 for r in s if r["rank"] == 1)
    top3 = sum(1 for r in s if r["rank"] and r["rank"] <= 3)
    para = report["paraphrases"]
    para1 = sum(1 for r in para if r["rank"] == 1)
    para3 = sum(1 for r in para if r["rank"] and r["rank"] <= 3)
    lines = [
        "# S2 memory store: measurements", "",
        f"Embedder: `{report['embedder']}`. Run on {report['when']} ({report['machine']}). Development data from the S1",
        "memories, so these are development numbers (see `chirii/eval/s2_gold.py`). No language model is involved yet.", "",
        "| Check | Result |", "| :--- | :--- |",
        f"| Sentences parsed fully right (development, 66) | {p['dev_fully_right']}/{p['dev_sentences']} |",
        f"| Sentences parsed fully right (22 fresh, first try, before any fix) | {p['fresh_fully_right_first_try']}/{p['fresh_sentences']} |",
        f"| Sentences parsed fully right (22 fresh, now) | {p['fresh_fully_right_now']}/{p['fresh_sentences']} |",
        f"| Updates linked to the right item | {sum(r['linked_right'] for r in u)}/{len(u)} |",
        f"| Updates changed the item correctly | {sum(r['changed_right'] for r in u)}/{len(u)} |",
        f"| Meaning-based questions: needed sentence ranked 1st | {top1}/{len(s)} |",
        f"| Meaning-based questions: needed sentence in top 3 | {top3}/{len(s)} |",
        f"| New questions sharing no key word with the note: ranked 1st / in top 3 | {para1}/{len(para)} / {para3}/{len(para)} |",
        f"| Date windows with nothing missing and nothing extra | {sum(1 for r in w if not r['missing'] and not r['should_not_be_there'])}/{len(w)} |",
        f"| Delete all my data: rows, vectors removed; folder gone | {d['deleted']['items']}, {d['deleted']['vectors']}; {d['folder_gone']} |",
        f"| Index matches database before delete | {d['index_matches_db']} |",
        f"| Time to add one sentence / one search | {d['add_ms_per_sentence']} ms / {d['search_ms']} ms |",
        f"| Disk space for one user with 9 items | {d['folder_kb_for_9_items']} KB |", "",
        "## Parsing, field by field (development)", "", "| Field | Right |", "| :--- | ---: |",
        *[f"| {k} | {v} |" for k, v in p["dev_fields"].items()], "",
        "## Updates", "", "| Task | Linked right | Changed right | Linked to |", "| :--- | :--- | :--- | :--- |",
        *[f"| {r['task']} | {r['linked_right']} | {r['changed_right']} | {r['linked_to']} |" for r in u], "",
        "## Search by meaning", "", "| Task | Question | Rank of needed sentence | Top result |", "| :--- | :--- | ---: | :--- |",
        *[f"| {r['task']} | {r['question']} | {r['rank']} | {r['top1']} |" for r in s], "",
        "## Search by meaning: new questions with no shared key words", "", "| Question | Rank of needed sentence | Top result |", "| :--- | ---: | :--- |",
        *[f"| {r['question']} | {r['rank']} | {r['top1']} |" for r in para], "",
        "## Date windows", "", "| Task | Window | Entries | Missing | Should not be there |", "| :--- | :--- | ---: | :--- | :--- |",
        *[f"| {r['task']} | {r['window']} | {r['entries']} | {r['missing'] or '-'} | {r['should_not_be_there'] or '-'} |" for r in w], "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["fast", "hash"], default="fast")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "chirii" / "results" / "s2"))
    args = parser.parse_args()
    if args.embedder == "fast":
        settings = Settings(secret_key=SECRET)
        embedder, name = FastEmbedder(settings.embed_model, settings.embed_cache_dir), settings.embed_model
    else:
        embedder, name = HashEmbedder(), "hash (test stand-in)"
    tasks = [t for t in load_tasks() if t["scored"]]
    base = base_texts(tasks)
    root = Path(tempfile.mkdtemp(prefix="chirii-s2-"))
    try:
        report = {
            "embedder": name, "when": date.today().isoformat(), "machine": f"{platform.system()} {platform.machine()}, CPU",
            "parsing": measure_parsing(),
            "updates": measure_updates(tasks, root, embedder, base),
            "search": measure_search(tasks, root, embedder, base),
            "paraphrases": measure_paraphrases(tasks, root, embedder, base),
            "windows": measure_windows(tasks, root, embedder, base),
            "delete_and_speed": measure_delete_and_speed(tasks, root, embedder, base),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.embedder == "fast" else "_hash"
    (out / f"s2_memory{suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / f"S2_MEMORY{suffix}.md").write_text(build_markdown(report), encoding="utf-8")
    print(build_markdown(report))


if __name__ == "__main__":
    main()

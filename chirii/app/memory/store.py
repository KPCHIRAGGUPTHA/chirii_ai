"""
One user's memory: a SQLite database and a FAISS search index, together in a folder that belongs to that user alone.

- SQLite is the source of truth: every item, its dates and amounts, and a history of every change.
- FAISS holds one vector per item so notes can be found by meaning. It can always be rebuilt from SQLite.
- The folder name is a keyed hash of the user id, so file names don't reveal who the user is, and one user's data can
  never be read through another user's store.
- delete_all() removes the whole folder: every row, every vector and the history.

Updates ("I paid the water bill", "the dentist moved to 11 December") are linked to the earlier item they change. If
no earlier item clearly matches, the update is kept as unlinked rather than guessed, so the app can ask the user.
"""

import hashlib
import hmac
import json
import re
import shutil
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import faiss
import numpy as np

from chirii.app.logging_setup import log_event
from chirii.app.memory.embeddings import Embedder
from chirii.app.memory.extract import WEEKDAYS, Item, extract

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,
    text          TEXT NOT NULL,
    said_on       TEXT NOT NULL,
    date          TEXT,
    end_date      TEXT,
    time          TEXT,
    end_time      TEXT,
    amount        INTEGER,
    paid_amount   INTEGER NOT NULL DEFAULT 0,
    day_of_month  INTEGER,
    weekdays      TEXT NOT NULL DEFAULT '[]',
    dose_times    TEXT NOT NULL DEFAULT '[]',
    other_dates   TEXT NOT NULL DEFAULT '[]',
    other_times   TEXT NOT NULL DEFAULT '[]',
    other_amounts TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'active',
    action        TEXT,
    source        TEXT NOT NULL DEFAULT 'rules',
    needs_review  INTEGER NOT NULL DEFAULT 0,
    applies_to    INTEGER REFERENCES items(id) ON DELETE SET NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS changes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    update_id   INTEGER REFERENCES items(id) ON DELETE CASCADE,
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    changed_on  TEXT NOT NULL
);
"""
LIST_FIELDS = ("weekdays", "dose_times", "other_dates", "other_times", "other_amounts")
ITEM_FIELDS = ("kind", "text", "said_on", "date", "end_date", "time", "end_time", "amount", "day_of_month", *LIST_FIELDS, "action",
               "source", "needs_review")

# Words that say nothing about which item an update refers to
STOP_WORDS = set("""
a an the i me my we our you your he she her his it its they them their this that these those is am are was were be been
has have had do does did will would can could should must to of in on at by for from with and or not no now same time
today yesterday tomorrow still rest just also again correction moved postponed rescheduled extended preponed shifted
changed cancelled canceled called off paid pay already new instead to till until next this
january february march april may june july august september october november december jan feb mar apr jun jul aug sep
sept oct nov dec am pm rs inr rupees lakh
""".split())
SYNONYMS = {"dr": "doctor", "doctors": "doctor", "check": "checkup", "deliver": "delivery", "delivered": "delivery",
            "owes": "owe", "fees": "fee", "bills": "bill"}
# Words shared by many items. Sharing only these is not enough when the update names something more specific
# ("I paid the gas bill" must not mark the water bill paid).
GENERIC_WORDS = {"bill", "fee", "appointment", "meeting", "payment", "due", "emi", "date", "deadline", "visit"}
LINK_MIN_SCORE = 0.30
LINK_MIN_MARGIN = 0.05


def keywords(text: str) -> set:
    text = text.lower().replace("check-up", "checkup").replace("’", "'")
    words = re.findall(r"[a-z]+", re.sub(r"'s\b", "", text))
    out = set()
    for word in words:
        if len(word) < 2 or word in STOP_WORDS:
            continue
        word = SYNONYMS.get(word, word)
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def user_folder_name(user_id: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), f"user:{user_id}".encode("utf-8"), hashlib.sha256).hexdigest()[:32]


class MemoryStore:

    def __init__(self, root: Path, user_id: str, secret: str, embedder: Embedder,
                 extractor: Optional[Callable[[str, date], Item]] = None):
        self.user_id = user_id
        self.embedder = embedder
        self.extractor = extractor or extract  # rules only, unless a hybrid extractor is given
        self.dir = Path(root) / "users" / user_folder_name(user_id, secret)
        self.db_path = self.dir / "memory.db"
        self.index_path = self.dir / "index.faiss"
        self.dir.mkdir(parents=True, exist_ok=True)
        with closing(self._db()) as db:  # closing(): sqlite's own "with" commits but leaves the file open
            db.executescript(SCHEMA)
        self.index = self._load_index()

    # ------------------------------------------------------------------ storage helpers

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def _query(self, sql: str, args: Tuple = ()) -> List[sqlite3.Row]:
        with closing(self._db()) as db:
            return db.execute(sql, args).fetchall()

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))

    def _save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    def _index_add(self, item_id: int, text: str) -> None:
        vector = self.embedder.embed([text])
        self.index.add_with_ids(vector, np.array([item_id], dtype=np.int64))
        self._save_index()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        item = dict(row)
        for name in LIST_FIELDS:
            item[name] = json.loads(item[name])
        item["needs_review"] = bool(item["needs_review"])
        if item["amount"] is not None:
            item["remaining"] = item["amount"] - item["paid_amount"]
        return item

    # ------------------------------------------------------------------ adding

    def add(self, text: str, said_on: date) -> Dict[str, Any]:
        """Stores one sentence. Returns the stored item, and for an update, which item it changed and how."""
        started = time.perf_counter()
        parsed = self.extractor(text, said_on)
        now = datetime.now().isoformat(timespec="seconds")
        values = {name: getattr(parsed, name) for name in ITEM_FIELDS}
        for name in LIST_FIELDS:
            values[name] = json.dumps(values[name])
        values["status"] = "active"
        with closing(self._db()) as db, db:
            cursor = db.execute(
                f"INSERT INTO items ({', '.join(values)}, created_at, updated_at) VALUES ({', '.join('?' * len(values))}, ?, ?)",
                (*values.values(), now, now))
            item_id = cursor.lastrowid
        self._index_add(item_id, text)

        result: Dict[str, Any] = {"item": self.get(item_id), "linked_to": None, "changes": []}
        if parsed.kind == "update":
            target, candidates = self.find_update_target(parsed, exclude=item_id)
            result["candidates"] = candidates
            if target is None:
                self._set_status(item_id, "unlinked")
            else:
                result["linked_to"] = target["id"]
                result["changes"] = self._apply_update(item_id, parsed, target)
                self._set_status(item_id, "applied", applies_to=target["id"])
            result["item"] = self.get(item_id)
        log_event("memory_added", user=self.user_id, item_type=parsed.kind, stage="S2",
                  duration_ms=round((time.perf_counter() - started) * 1000))
        return result

    def _set_status(self, item_id: int, status: str, applies_to: Optional[int] = None) -> None:
        with closing(self._db()) as db, db:
            db.execute("UPDATE items SET status = ?, applies_to = COALESCE(?, applies_to), updated_at = ? WHERE id = ?",
                       (status, applies_to, datetime.now().isoformat(timespec="seconds"), item_id))

    # ------------------------------------------------------------------ updates

    def find_update_target(self, update: Item, exclude: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """The earlier item an update refers to, or None if no item clearly matches. Scored by shared key words, with
        meaning-similarity as a smaller second signal."""
        candidates = [item for item in self.items() if item["id"] != exclude and item["kind"] != "update"]
        if update.action == "paid":
            candidates = [c for c in candidates if c["amount"] is not None]
        elif update.action == "moved":
            candidates = [c for c in candidates if c["date"] or c["time"] or c["day_of_month"]]
        if not candidates:
            return None, []
        update_words = keywords(update.text)
        specific = update_words - GENERIC_WORDS
        query = self.embedder.embed([update.text])[0]
        vectors = self.embedder.embed([c["text"] for c in candidates])
        scored = []
        for item, vector in zip(candidates, vectors):
            item_words = keywords(item["text"])
            overlap = len(update_words & item_words) / max(1, len(update_words))
            if specific and not specific & item_words:
                overlap = 0.0  # it names something this item doesn't mention
            scored.append({"id": item["id"], "overlap": round(overlap, 3), "similarity": round(float(vector @ query), 3),
                           "score": round(overlap + 0.25 * float(vector @ query), 3)})
        scored.sort(key=lambda s: s["score"], reverse=True)
        best = scored[0]
        margin = best["score"] - (scored[1]["score"] if len(scored) > 1 else 0.0)
        if best["overlap"] == 0 or best["score"] < LINK_MIN_SCORE or margin < LINK_MIN_MARGIN:
            return None, scored[:3]
        return self.get(best["id"]), scored[:3]

    def _apply_update(self, update_id: int, update: Item, target: Dict[str, Any]) -> List[Dict[str, Any]]:
        changes: List[Tuple[str, Any, Any]] = []
        new_values: Dict[str, Any] = {}
        if update.action == "paid":
            remaining = target.get("remaining")
            if update.amount is not None and remaining is not None and update.amount < remaining:
                new_values["paid_amount"] = target["paid_amount"] + update.amount
            else:
                new_values["status"] = "paid"
        elif update.action == "cancelled":
            new_values["status"] = "cancelled"
        else:
            if update.date and update.date != target["date"]:
                new_values["date"] = update.date
                if target["end_date"] and target["date"]:
                    shift = date.fromisoformat(update.date) - date.fromisoformat(target["date"])
                    new_values["end_date"] = (date.fromisoformat(target["end_date"]) + shift).isoformat()
            if update.time and not update.keep_time and update.time != target["time"]:
                new_values["time"] = update.time
            if update.day_of_month and update.day_of_month != target["day_of_month"]:
                new_values["day_of_month"] = update.day_of_month
        for name, value in new_values.items():
            changes.append((name, target.get(name), value))
        if not changes:
            return []
        with closing(self._db()) as db, db:
            sets = ", ".join(f"{name} = ?" for name in new_values)
            db.execute(f"UPDATE items SET {sets}, updated_at = ? WHERE id = ?",
                       (*new_values.values(), datetime.now().isoformat(timespec="seconds"), target["id"]))
            db.executemany("INSERT INTO changes (item_id, update_id, field, old_value, new_value, changed_on) VALUES (?, ?, ?, ?, ?, ?)",
                           [(target["id"], update_id, name, None if old is None else str(old), str(new), update.said_on)
                            for name, old, new in changes])
        return [{"field": name, "old": old, "new": new} for name, old, new in changes]

    # ------------------------------------------------------------------ reading

    def get(self, item_id: int) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM items WHERE id = ?", (item_id,))
        return self._row_to_dict(rows[0]) if rows else None

    def items(self, include_updates: bool = True) -> List[Dict[str, Any]]:
        rows = self._query("SELECT * FROM items ORDER BY id")
        items = [self._row_to_dict(r) for r in rows]
        return items if include_updates else [i for i in items if i["kind"] != "update"]

    def history(self, item_id: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._query("SELECT field, old_value, new_value, changed_on, update_id FROM changes WHERE item_id = ? ORDER BY id", (item_id,))]

    def occurrences(self, item: Dict[str, Any], start: date, end: date) -> Iterator[date]:
        """Every day in [start, end] the item falls on: its date or date range, extra dates, monthly day, weekly days,
        or every day for medicines without a date."""
        days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
        if item["date"]:
            first = date.fromisoformat(item["date"])
            last = date.fromisoformat(item["end_date"]) if item["end_date"] else first
            yield from (d for d in days if first <= d <= last)
        for other in item["other_dates"]:
            d = date.fromisoformat(other)
            if start <= d <= end:
                yield d
        if item["day_of_month"]:
            yield from (d for d in days if d.day == item["day_of_month"])
        if item["weekdays"]:
            wanted = {WEEKDAYS.index(w) for w in item["weekdays"]}
            yield from (d for d in days if d.weekday() in wanted)
        if item["kind"] == "medicine" and not item["date"]:
            yield from days

    def agenda(self, start: date, end: date, statuses: Tuple[str, ...] = ("active",)) -> List[Dict[str, Any]]:
        """Everything that falls between start and end (inclusive), one entry per day, sorted by day and time."""
        entries = []
        for item in self.items(include_updates=False):
            if item["status"] not in statuses:
                continue
            for day in sorted(set(self.occurrences(item, start, end))):
                entries.append({"day": day.isoformat(), "time": item["time"], "item": item})
        entries.sort(key=lambda e: (e["day"], e["time"] or "99:99"))
        return entries

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Items closest in meaning to the query, with shared words as a small extra signal. Updates that were
        applied are left out (their item already shows the change); unlinked updates are included."""
        started = time.perf_counter()
        if self.index.ntotal == 0:
            return []
        vector = self.embedder.embed([query])
        scores, ids = self.index.search(vector, self.index.ntotal)
        query_words = keywords(query)
        results = []
        for score, item_id in zip(scores[0], ids[0]):
            item = self.get(int(item_id)) if item_id >= 0 else None
            if item is None or (item["kind"] == "update" and item["status"] == "applied"):
                continue
            overlap = len(query_words & keywords(item["text"])) / max(1, len(query_words))
            results.append({"item": item, "similarity": round(float(score), 3), "score": round(float(score) + 0.3 * overlap, 3)})
        results.sort(key=lambda r: r["score"], reverse=True)
        log_event("memory_search", user=self.user_id, result_count=min(k, len(results)), stage="S2",
                  duration_ms=round((time.perf_counter() - started) * 1000))
        return results[:k]

    # ------------------------------------------------------------------ deleting

    def delete_item(self, item_id: int) -> bool:
        with closing(self._db()) as db, db:
            deleted = db.execute("DELETE FROM items WHERE id = ?", (item_id,)).rowcount
        if deleted:
            self.index.remove_ids(np.array([item_id], dtype=np.int64))
            self._save_index()
        return bool(deleted)

    def delete_all(self) -> Dict[str, int]:
        """'Delete all my data': removes every row, every vector and the history, by removing the user's folder."""
        counts = {"items": len(self.items()), "vectors": int(self.index.ntotal)}
        self.index.reset()
        shutil.rmtree(self.dir)
        log_event("memory_deleted_all", user=self.user_id, item_count=counts["items"], stage="S2")
        return counts

    # ------------------------------------------------------------------ index upkeep

    def index_ids(self) -> set:
        if self.index.ntotal == 0:
            return set()
        return set(int(i) for i in faiss.vector_to_array(self.index.id_map))

    def is_consistent(self) -> bool:
        """Every item has exactly one vector and every vector has an item."""
        return self.index_ids() == {item["id"] for item in self.items()} and self.index.ntotal == len(self.items())

    def rebuild_index(self) -> int:
        """Rebuilds the search index from SQLite, e.g. after changing the embedding model."""
        items = self.items()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))
        if items:
            self.index.add_with_ids(self.embedder.embed([i["text"] for i in items]), np.array([i["id"] for i in items], dtype=np.int64))
        self._save_index()
        return int(self.index.ntotal)

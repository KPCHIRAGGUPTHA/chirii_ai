"""
Development data for S2, taken from the S1 eval memories: every sentence with the structured item it should become, the
update sentences with the item each one changes, and the item each meaning-based question should find.

These come from the development set, so S2 numbers measured on them are development numbers. The parser rules were
written while looking at these sentences, so they will look better here than on new sentences.
"""

import json
from datetime import date
from typing import Any, Dict, List

from chirii.app.config import REPO_ROOT

TASKS_PATH = REPO_ROOT / "chirii" / "eval" / "s1_tasks.jsonl"
BASE_SAID_ON = date(2026, 12, 1)  # base memories were "said earlier"; 1 December is before every task's today

# Which earlier sentence each update changes (a phrase from that sentence)
UPDATE_TARGETS = {
    "S1-08": "BESCOM electricity bill",
    "S1-09": "Dentist appointment with Dr. Kavitha",
    "S1-18": "Metro Water bill",
    "S1-19": "Divya's school term fee",
    "S1-28": "Mrs. Deshpande owes",
    "S1-29": "Gupta Traders will deliver",
    "S1-37": "My next check-up is with Dr. Ramaswamy",
    "S1-48": "Hostel mess fee",
}

# Which sentence each meaning-based question needs (a phrase from that sentence)
SEMANTIC_TARGETS = {
    "S1-05": "the release has moved",
    "S1-15": "Shopping list for Pongal",
    "S1-16": "eye check-up",
    "S1-25": "new milk vendor",
    "S1-34": "Dr. Ramaswamy said",
    "S1-35": "Ananya's birthday",
    "S1-44": "Note from DBMS class",
    "S1-45": "The PNR is saved",
}


def load_tasks() -> List[Dict[str, Any]]:
    return [json.loads(line) for line in TASKS_PATH.read_text(encoding="utf-8").splitlines()]


def said_on_for(entry: Dict[str, Any], task: Dict[str, Any], base_texts: set) -> date:
    """Base memories were said on 1 December. A task's extra sentence that says "today" was said on the item's date
    (the eval notes say so); other extra sentences were said on the task's today."""
    if entry["said"] in base_texts:
        return BASE_SAID_ON
    if "today" in entry["said"].lower() and entry["item"].get("date"):
        return date.fromisoformat(entry["item"]["date"])
    return date.fromisoformat(task["today"])


def sentences() -> List[Dict[str, Any]]:
    """Every distinct sentence across the scored tasks, with its intended item and the day it was said."""
    tasks = [t for t in load_tasks() if t["scored"]]
    by_persona: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        by_persona.setdefault(t["persona"], []).append(t)
    base_texts = set()
    for persona_tasks in by_persona.values():
        common = set.intersection(*(set(e["said"] for e in t["memory"]) for t in persona_tasks))
        base_texts |= common
    seen, out = set(), []
    for t in tasks:
        for entry in t["memory"]:
            if entry["said"] in seen:
                continue
            seen.add(entry["said"])
            out.append({"said": entry["said"], "gold": entry["item"], "said_on": said_on_for(entry, t, base_texts),
                        "persona": t["persona"], "base": entry["said"] in base_texts, "task": t["id"]})
    return out

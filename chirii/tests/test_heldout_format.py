"""The held-out format check (S6): accepts a well-formed file, reports problems, never changes the file."""

import copy
import json

from chirii.eval.check_heldout_format import PERSONAS, TYPES, check

EXAMPLE = {"persona": "priya", "type": "lookup", "today": "2027-02-10", "question": "how much is the water purifier service?",
           "memory": [{"said": "Aquaguard service is due on 18 Feb, ₹650.", "said_on": "2027-02-01",
                       "item": {"type": "bill", "title": "water purifier service", "date": "2027-02-18", "amount": 650}}],
           "rules": {"must_include": [{"amount": 650}]}, "allow": [], "note": "₹650.", "scored": True}


def write(tmp_path, tasks):
    path = tmp_path / "heldout_tasks.jsonl"
    path.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")
    return path


def thirty():
    tasks, n = [], 1
    for kind in TYPES:
        for person in PERSONAS:
            t = dict(copy.deepcopy(EXAMPLE), id=f"H-{n:02d}", type=kind, persona=person)
            if kind == "unknown":
                t["rules"] = {"abstain": True, "must_not": [{"amount": 650}]}
            if kind == "plan":
                t["rules"] = {"must_include": [{"amount": 650}], "order": [{"amount": 650}]}
            tasks.append(t)
            n += 1
    return tasks


def test_a_well_formed_file_passes_and_is_not_changed(tmp_path):
    path = write(tmp_path, thirty())
    before = path.read_bytes()
    assert check(path) == ([], [])
    assert path.read_bytes() == before


def test_problems_are_reported(tmp_path):
    tasks = thirty()
    tasks[0]["memory"][0]["said_on"] = "2027-03-01"                   # after today
    tasks[1]["rules"]["must_include"] = [{"amount": "₹650"}]            # not whole rupees
    tasks[2]["rules"]["must_include"] = [{"date": "2027-02-20"}]        # needed, but in no item
    tasks[15]["rules"] = {"must_not": [{"amount": 650}]}                # unknown without abstain
    del tasks[29]
    errors, warnings = check(write(tmp_path, tasks))
    text = "\n".join(errors + warnings)
    for expected in ("29 tasks, not 30", "said_on is after today", "amount must be whole rupees",
                     "must_include date 2027-02-20 is in no item", "an unknown task needs abstain", "update x arjun: 0 tasks"):
        assert expected in text, expected

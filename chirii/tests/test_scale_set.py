"""Checks on the 500-item scale set (written before it is run). These check the set, not any system."""

import json
import re
from datetime import date

from chirii.app.config import REPO_ROOT
from chirii.app.memory.extract import extract
from chirii.eval.build_scale_set import CROWDED, NEEDLES, SAID_ON, fillers, questions
from chirii.eval.scoring import allowed_facts, score_task

SET = json.loads((REPO_ROOT / "chirii" / "eval" / "scale_set.json").read_text(encoding="utf-8"))


def test_file_matches_the_builder():
    assert SET["questions"] == json.loads(json.dumps(questions()))
    assert SET["stores"]["large"] == json.loads(json.dumps(NEEDLES + fillers(475)))
    assert len(SET["stores"]["large"]) == 500 and len(SET["stores"]["small"]) == 40
    assert SET["stores"]["small"][:25] == SET["stores"]["large"][:25]


def test_rules_agree_with_memory_and_an_empty_reply_fails():
    for store in ("large", "small"):
        for qn in SET["questions"]:
            task = {**qn, "memory": SET["stores"][store], "allow": []}
            dates, amounts = allowed_facts(task)
            for fact in qn["rules"].get("must_include", []):
                if "date" in fact:
                    d = date.fromisoformat(fact["date"])
                    assert (d.day, d.month) in dates, (qn["id"], fact)
                if "amount" in fact:
                    assert fact["amount"] in amounts, (qn["id"], fact)
            assert not score_task(task, "")["passed"], qn["id"]


def test_fillers_never_mention_a_needle_or_a_crowded_month():
    unique = {"Ishaan", "Murugan", "Zoya", "Hyundai", "Diwali", "Thatha", "Aarav", "Bhattacharya", "LIC", "Sneha", "fridge",
              "Meenakshi", "Kindle", "Paati", "Tatkal", "Kumaran", "Fernandes", "Pillai", "Bruno", "scooter", "Wi-Fi", "Rohit"}
    months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August",
              9: "September", 10: "October", 11: "November", 12: "December"}
    for f in fillers(475):
        assert not any(re.search(rf"\b{re.escape(u)}\b", f["said"], re.I) for u in unique), f["said"]
        for kind, month in CROWDED.items():
            assert not (f"{kind} bill" in f["said"] and months[month] in f["said"]), f["said"]


def test_the_year_rule_looks_ahead_unless_the_sentence_is_past():
    june = date(2027, 6, 1)
    assert extract("Cousin's sangeet is on 12 December in Kochi.", june).date == "2027-12-12"
    assert extract("Insurance renewal of ₹3,000 on 15 March.", june).date == "2028-03-15"
    assert extract("The parcel was delivered on 20 May.", june).date == "2027-05-20"
    assert extract("Fee of ₹900 must be paid by 18 December.", june).date == "2027-12-18"
    assert extract("Meeting with Anil on 20 May.", june).date == "2027-05-20"     # within a month before: last month


def test_the_rules_extractor_reads_every_needle_as_meant():
    said_on = date.fromisoformat(SAID_ON)
    for n in NEEDLES:
        item = extract(n["said"], said_on)
        for key in ("date", "time", "amount"):
            if n["item"].get(key) is not None:
                assert getattr(item, key) == n["item"][key], (n["said"], key, getattr(item, key))

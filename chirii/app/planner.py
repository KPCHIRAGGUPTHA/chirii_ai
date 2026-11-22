"""
Structured answers built by code (S4). The model only rewords them, and if its rewording drops or changes anything, the
code's own text is shown.

    window_draft   what falls in a period, filtered by kind (payments, money owed to the user, to-dos, family dates),
                   leaving out paid, cancelled and applied items, with totals
    plan_draft     a day's or a week's timeline in time order, with clashes found by code and, for a day, what is
                   coming up tomorrow
    status_draft   whether something was paid or done, from the item's status and history

Each returns a Draft: the text, the items it used, and the facts the rewording must keep.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from chirii.app.formatting import KIND_LABELS, clauses, first_clause, make_title, say_amount, say_date, say_time
from chirii.app.memory.extract import find_dates as extract_dates, find_times
from chirii.app.memory.store import MemoryStore, keywords

DEFAULT_MINUTES = 60          # an appointment with no end time is taken to last an hour, for finding clashes
TIMED_KINDS = {"appointment", "routine", "shift", "exam", "travel", "important_date", "bill"}
PAYABLE = {"bill"}
NOT_LISTED = {"note", "update", "income"}
DONE = {"paid", "cancelled", "applied", "done"}


@dataclass
class Draft:
    text: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    keep: List[Dict[str, Any]] = field(default_factory=list)   # facts the rewording must keep, in marker fact form
    in_order: bool = False                                     # times must stay in this order
    clash: bool = False                                        # the rewording must still say there is a clash


def short_day(d: date) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def label_for(item: Dict[str, Any], day: Optional[date] = None, hhmm: Optional[str] = None) -> str:
    """A name for one occurrence. For an item with several dates or times ("Data Structures on 7 December, DBMS on 9
    December"), the clause of what the user said that holds that date or time, without the date and time."""
    title = item.get("title") or make_title(item["text"])
    said_on = date.fromisoformat(item["said_on"])
    multiple = (day is not None and item.get("other_dates")) or (hhmm is not None and item.get("other_times"))
    if not multiple:
        return title
    for clause in clauses(item["text"]):  # S4b cosmetic: no split after "Dr.", "Mrs.", "Smt."
        if day is not None and item.get("other_dates") and any(d == day for _, d in extract_dates(clause, said_on)[0]):
            return clean_label(clause) or title
        if hhmm is not None and item.get("other_times") and any(t == hhmm for _, t in find_times(clause)[0]):
            return clean_label(clause) or title
    return title


MONTH_WORDS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"


def strip_when(text: str, dates: bool = True, times: bool = True) -> str:
    """The sentence without its date or time phrases, for an item whose date or time was changed later (S4b)."""
    if dates:
        text = re.sub(rf"\s*\b(on|by|before|from|till|until)?\s*(the\s+)?\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{MONTH_WORDS}\b\.?", "", text, flags=re.I)
        text = re.sub(rf"\s*\b(on|by|before)?\s*{MONTH_WORDS}\s+\d{{1,2}}(st|nd|rd|th)?\b", "", text, flags=re.I)
    if times:
        text = re.sub(r"\s*\b(between\s+\d{1,2}([:.]\d{2})?\s*(am|pm)?\s+and\s+)?(at|by|before|from)?\s*\d{1,2}([:.]\d{2})?\s*(am|pm)\b", "", text, flags=re.I)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def clean_label(clause: str) -> str:
    text = re.sub(r"\b(on|by|before|from)\s+(the\s+)?\d{1,2}(st|nd|rd|th)?\s+(of\s+)?[A-Za-z]+\b", "", clause)
    text = re.sub(r"\b(at|by|before|from)\s+\d{1,2}([:.]\d{2})?\s*(am|pm)\b", "", text, flags=re.I)
    text = re.sub(r"^\s*(I have|I've got|there is|there's)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,:-")
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------- date windows and payments

def wanted(item: Dict[str, Any], focus: List[str], question: str) -> bool:
    kind = item["kind"]
    if kind in NOT_LISTED or item["status"] in DONE:
        return False
    if kind == "medicine" and not item.get("date"):
        return bool(re.search(r"\b(medicines?|tablets?|pills?)\b", question, re.I))
    if "all" in focus:
        return True
    ok = False
    if "pay" in focus:
        ok |= kind in PAYABLE
        named = [w for w in ("fee", "emi", "rent", "premium", "insurance") if re.search(rf"\b{w}s?\b", question, re.I)]
        if named and kind in PAYABLE:
            ok = any(w in keywords(item["text"]) or w + "s" in item["text"].lower() for w in named)
    if "todo" in focus:
        ok |= kind == "task"
    if "owed" in focus:
        ok |= kind == "receivable"
    if "dates" in focus:
        ok |= kind == "important_date"
    if "due" in focus:
        ok |= kind in ("bill", "task", "receivable")
    return ok


def window_draft(store: MemoryStore, start: date, end: date, focus: List[str], question: str, named: bool = True,
                 header: Optional[str] = None) -> Draft:
    """The list for a period. The header names the period's dates only when the question named the period (S4b: a
    default period, as for "which fees do I still have to pay?", is not stated). Money owed to the user is listed
    separately as "To receive" (S4b)."""
    entries = []
    for item in store.items(include_updates=False):
        if not wanted(item, focus, question):
            continue
        for day in sorted(set(store.occurrences(item, start, end))):
            entries.append((day, item["time"] or "99:99", item))
            if item["kind"] == "medicine":
                break
    entries.sort(key=lambda e: (e[0], e[1]))
    # S4c: the period reader's own words when it has them ("this weekend (…)", "before the Goa trip on …")
    period = (" " + header) if header else ((" " + say_period(start, end)) if named else "")
    if not entries:
        what = {"pay": "Nothing to pay", "owed": "Nothing to receive", "dates": "No important dates",
                "due": "Nothing due"}.get(focus[0], "Nothing saved")
        return Draft(text=f"{what}{period}, as far as your saved memories go.")
    sections: Dict[str, List[str]] = {"main": [], "receive": []}
    keep, totals, counts, used = [], {"main": 0, "receive": 0}, {"main": 0, "receive": 0}, []
    for day, _, item in entries:
        part = "receive" if item["kind"] == "receivable" else "main"
        bits = [label_for(item, day=day)]
        if item["time"]:
            bits.append(f"at {say_time(item['time'])}")
            keep.append({"time": item["time"]})
        amount = item["remaining"] if item.get("amount") is not None else None
        if amount is not None:
            if part == "receive" and re.search(r"\bowes?$", bits[0]):
                bits[0] = f"{bits[0]} you {say_amount(amount)}" + (" (the rest still owed)" if item["paid_amount"] else "")
            else:  # S4b cosmetic: "Mrs. Deshpande owes you ₹3,450", not "owes, ₹3,450"
                bits.append(say_amount(amount) + (" still owed" if item["paid_amount"] else ""))
            keep.append({"amount": amount})
            if part == "receive" or item["kind"] in PAYABLE:
                totals[part] += amount
                counts[part] += 1
        if item["day_of_month"] and day.day == item["day_of_month"] and not item["date"]:
            bits.append("(every month)")
        keep.append({"date": day.isoformat()})
        sections[part].append(f"- {short_day(day)}: " + ", ".join(bits))
        used.append(item)
    heads = {"pay": "To pay", "owed": "To receive", "dates": "Important dates", "due": "Due"}
    head = heads.get(focus[0], "Coming up")
    blocks = []
    if sections["main"]:
        blocks.append(f"{head}{period}:\n" + "\n".join(sections["main"]))
        if counts["main"] >= 2:
            blocks.append(f"Total to pay: {say_amount(totals['main'])}.")
            keep.append({"amount": totals["main"]})
    if sections["receive"]:
        blocks.append(("To receive" + (period if not sections["main"] else "") + ":\n") + "\n".join(sections["receive"]))
        if counts["receive"] >= 2:
            blocks.append(f"Total to receive: {say_amount(totals['receive'])}.")
    return Draft(text="\n".join(blocks), items=dedupe(used), keep=keep)


def say_period(start: date, end: date) -> str:
    if start == end:
        return f"on {short_day(start)}"
    return f"from {short_day(start)} to {short_day(end)}"


def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for item in items:
        if item["id"] not in seen:
            seen.add(item["id"])
            out.append(item)
    return out


# ---------------------------------------------------------------- plans

@dataclass
class Slot:
    start: Optional[int]            # minutes from midnight; None for untimed items
    end: Optional[int]
    label: str
    item: Dict[str, Any]
    deadline: bool = False          # "before 5 pm": a time to finish by, not an event
    hhmm: Optional[str] = None


def minutes(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    return h * 60 + m


def slots_for(store: MemoryStore, day: date) -> Tuple[List[Slot], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Timed slots on a day, untimed items, and daily medicines."""
    slots, untimed, medicines = [], [], []
    for item in store.items(include_updates=False):
        if item["kind"] in ("note", "update", "income") or item["status"] in DONE:
            continue
        if not any(d == day for d in store.occurrences(item, day, day)):
            continue
        if item["kind"] == "medicine" and not item["date"]:
            medicines.append(item)
            continue
        times = ([item["time"]] if item["time"] else []) + list(item.get("other_times") or [])
        if not times:
            untimed.append(item)
            continue
        is_deadline = item["kind"] == "task" or bool(re.search(r"\b(before|by)\s+\d", item["text"], re.I))
        for hhmm in times:
            start = minutes(hhmm)
            end = None
            if hhmm == item["time"] and item.get("end_time"):
                end = minutes(item["end_time"])
                if end <= start:
                    end += 24 * 60          # a night shift that ends the next morning
            elif not is_deadline:
                end = start + DEFAULT_MINUTES
            label = label_for(item, hhmm=hhmm) if item.get("other_times") else label_for(item, day=day)
            slots.append(Slot(start, end, label, item, deadline=is_deadline and hhmm == item["time"], hhmm=hhmm))
    slots.sort(key=lambda s: s.start)
    return slots, untimed, medicines


def clashes(slots: List[Slot]) -> List[Tuple[Slot, Slot]]:
    found = []
    events = [s for s in slots if not s.deadline and s.end is not None]
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if a.item["id"] != b.item["id"] and a.start < b.end and b.start < a.end:
                found.append((a, b))
    return found


def slot_text(slot: Slot) -> str:
    when = say_time(slot.hhmm)
    if slot.deadline:
        return f"by {when}: {slot.label}"
    if slot.end is not None and slot.hhmm == slot.item["time"] and slot.item.get("end_time"):
        return f"{when} to {say_time(slot.item['end_time'])}: {slot.label}"
    return f"{when}: {slot.label}"


def untimed_text(item: Dict[str, Any]) -> str:
    bits = [item.get("title") or make_title(item["text"])]
    if item.get("amount") is not None:
        bits.append(say_amount(item["remaining"]) + (" (money due to you)" if item["kind"] == "receivable" else " to pay"))
    return ", ".join(bits)


def medicine_text(medicines: List[Dict[str, Any]]) -> str:
    parts = []
    for m in medicines:
        name = first_clause(m["text"])
        name = re.sub(r"^\s*I take\s+", "", name, flags=re.I)
        name = re.split(r"\s+(after|before|at|in the|every)\b", name, maxsplit=1)[0]
        parts.append(f"{name} ({', '.join(m['dose_times'])})" if m["dose_times"] else name)
    return "Medicines: " + "; ".join(parts) + "."


def day_lines(store: MemoryStore, day: date, keep: List[Dict[str, Any]], used: List[Dict[str, Any]], with_medicines: bool) -> Tuple[List[str], bool]:
    slots, untimed, medicines = slots_for(store, day)
    lines = []
    for slot in slots:
        lines.append("- " + slot_text(slot))
        keep.append({"time": slot.hhmm})
        used.append(slot.item)
    for item in untimed:
        lines.append("- " + untimed_text(item))
        used.append(item)
        if item.get("amount") is not None:
            keep.append({"amount": item["remaining"]})
    found = clashes(slots)
    for a, b in found:
        lines.append(f"- Clash: {a.label} ({say_time(a.hhmm)}) overlaps with {b.label} ({say_time(b.hhmm)}).")
    if with_medicines and medicines:
        lines.append("- " + medicine_text(medicines))
        used.extend(medicines)
    return lines, bool(found)


def plan_draft(store: MemoryStore, start: date, end: date, today: date) -> Draft:
    keep, used, any_clash = [], [], False
    if start == end:
        lines, any_clash = day_lines(store, start, keep, used, with_medicines=True)
        text = f"Your plan for {short_day(start)}:\n" + ("\n".join(lines) if lines else "- Nothing saved for this day.")
        tomorrow = start + timedelta(days=1)
        t_slots, t_untimed, _ = slots_for(store, tomorrow)
        coming = [s for s in t_slots if s.item["kind"] in ("exam", "appointment", "travel")]
        coming_untimed = [i for i in t_untimed if i["kind"] in ("exam", "bill", "task", "important_date", "receivable")]
        if coming or coming_untimed:
            parts = [f"{s.label} at {say_time(s.hhmm)}" for s in coming] + [untimed_text(i) for i in coming_untimed]
            text += f"\nComing up tomorrow ({short_day(tomorrow)}): " + "; ".join(parts) + ". Prepare for it today."
            used += [s.item for s in coming] + coming_untimed
        return Draft(text=text, items=dedupe(used), keep=keep, in_order=True, clash=any_clash)
    blocks = []
    day = start
    while day <= end:
        lines, clash_today = day_lines(store, day, keep, used, with_medicines=False)
        if lines:
            blocks.append(f"{short_day(day)}:\n" + "\n".join("  " + l for l in lines))
            keep.append({"date": day.isoformat()})
        any_clash |= clash_today
        day += timedelta(days=1)
    text = f"Your plan {say_period(start, end)}:\n" + ("\n".join(blocks) if blocks else "Nothing saved for these days.")
    return Draft(text=text, items=dedupe(used), keep=keep, in_order=False, clash=any_clash)


# ---------------------------------------------------------------- status

def status_draft(item: Optional[Dict[str, Any]], history: List[Dict[str, Any]], question: str) -> Draft:
    if item is None:
        return Draft(text="I don't have that saved.")
    title = item.get("title") or make_title(item["text"])
    changed = {h["field"]: h for h in history}
    if item["status"] == "paid":
        when = changed.get("status", {}).get("changed_on")
        return Draft(text=f"Yes, {title} is marked paid" + (f" (you told me on {say_date(when)})." if when else "."), items=[item])
    if item["status"] == "cancelled":
        return Draft(text=f"{title} is marked cancelled.", items=[item])
    if item.get("paid_amount"):
        return Draft(text=f"Partly: {say_amount(item['paid_amount'])} has been paid and {say_amount(item['remaining'])} is still owed for {title}.",
                     items=[item], keep=[{"amount": item["remaining"]}])
    asks_paid = re.search(r"\bpaid\b", question, re.I)
    if asks_paid and item.get("amount") is not None:
        due = f", due {say_date(item['date'])}" if item.get("date") else ""
        return Draft(text=f"No, {title} ({say_amount(item['amount'])}{due}) is not marked paid yet.", items=[item],
                     keep=[{"amount": item["amount"]}])
    return Draft(text=f"I don't have a record of that. I only know what you told me: \"{item['text']}\"", items=[item])

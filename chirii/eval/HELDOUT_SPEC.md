# Held-out test set: specification for the task writer

> **For the owner, before starting the fresh session.** Copy only this file into a new, empty folder outside the
> repository (for example `C:\dev\heldout_writing\`), start a fresh Claude Code session in that folder, and ask it to
> follow this file. The session must not see the Chirii code, the development sets or any results. Copy its two output
> files into `chirii/eval/` afterwards. The builder then only runs `python -m chirii.eval.check_heldout_format` (format
> check, no content changes); you review and edit, and commit before any run (`chirii/CRITERIA.md`).

---

## What you are writing

Chirii is a personal memory assistant for people in India. A person tells it things in plain sentences ("My gas bill
of ₹1,100 is due on 12 March", "Dentist on Friday at 5 pm"), and later asks questions about them ("What do I have to
pay this week?", "Plan my Saturday"). It answers from what it was told and must not make anything up.

You are writing **30 test tasks** that will be run **once** to measure it. Each task is: what the person told Chirii
earlier (with the day they said it), the day they ask, one question, and rules a program uses to mark the reply.

- **30 tasks: 6 types × 5 people, one task per person per type.**
- **New situations in your own wording.** Do not copy the examples in this file. Vary how people phrase things, as
  real people would type on a phone: short and long questions, "Do I…", "What's…", "Anything…", "Which…", Indian
  English. Up to 6 of the 30 questions may mix in Hindi words (Hinglish), as long as the marking rules stay clear.
- **Everything fictional.** Invented names and numbers; no real account numbers, phone numbers or ID numbers.
- **Fair and unambiguous.** A careful person reading the memories must be able to give the right answer, and the rules
  must accept every right answer and reject wrong ones. If a reasonable reader could answer two ways, change the task.

## The 6 task types (5 of each)

| `type` | What the question asks | Rules it needs |
| :--- | :--- | :--- |
| `lookup` | One fact about one saved thing: when, what time, how much, where | `must_include` the fact(s) |
| `date_window` | What falls in a period: "this week", "next 10 days", "this weekend", "before Diwali", "by 15 March", "in July", "before my trip to Goa" (an event that is itself saved) | `must_include` what is in the period; `must_not` saved things just outside it |
| `plan` | A day or a week laid out: "Plan my Saturday", "What does my Tuesday look like?" | `must_include` the day's timed things; `order` (earliest first); `flag_clash: true` when two things overlap (at least 2 of the 5 plan tasks should have a clash, at least 1 none) |
| `unknown` | Something never told to Chirii. The answer must say it isn't saved. | `abstain: true`; in at least 3 of the 5, something **similar** is saved (water bill asked, electricity bill saved), and its date or amount goes in `must_not` |
| `semantic` | What a saved note says, asked in **different words** from the note ("What did the doctor say about my diet?" when the note says "Dr. Rao: cut sugar, walk 30 minutes") | `must_include` the key fact(s), as `text` alternatives where wording can vary |
| `update` | A later sentence changed an earlier one (moved, cancelled, paid, amount changed, new time). The question asks for the current state. | `must_include` the new value; `must_not` the old one |

## The 5 people

All fictional. Each task's memory is that person's life at the time of the task: 6 to 15 sentences, including things
the question is **not** about (other bills, appointments, notes), so that picking the right one matters. Different
tasks for the same person may share sentences and may happen on different days.

1. **`priya`: Priya, 32, software tester in Bengaluru.** Works rotating shifts (some night weeks). Lives with her
   husband Arun; her mother (Amma) lives nearby. Pays rent or a home loan, utility and phone bills, insurance; has
   team calls, gym, doctor visits, friends' weddings, office deadlines.
2. **`lakshmi`: Lakshmi, 45, homemaker in Chennai.** Husband Murali works at a bank; children Divya (Class 10) and
   Karthik (Class 6). Handles school fees, tuition, PTA meetings, exams, the gas cylinder, household bills, relatives'
   functions, festival shopping, her own health check-ups.
3. **`ramesh`: Ramesh, 52, runs a kirana (grocery) shop in Pune.** Deals with suppliers and their payments, customers
   who owe him money (udhaar), GST filing, shop rent, electricity, his son's college fees, his mother's medicines, and
   the occasional wholesale market trip.
4. **`sharada`: Sharada, 72, retired teacher in Mysuru.** Diabetes and high blood pressure: several medicines at set
   times, doctor and lab visits, pension, a fixed deposit, grandchildren's birthdays, temple visits, a daughter in
   another city who visits.
5. **`arjun`: Arjun, 20, B.Tech student in a hostel in Vellore.** Classes and lab sessions, assignments and their
   deadlines, exams (with several dates), hostel and mess fees, money borrowed from or lent to friends, train tickets
   home, club events, a part-time internship.

## Dates

- Use dates in 2027. Each task has its own `today` (the day the question is asked). Spread them over the year.
- Every sentence has a `said_on` date on or before `today`. Things said "earlier" can be days or weeks before; an
  update is said after the sentence it changes.
- Sentences without a year ("on 12 March") mean the next 12 March after the day they were said; if that would be
  unclear, write the sentence so it isn't.
- Write dates in sentences the way people do: "12 March", "the 5th of every month", "next Friday" (only if `said_on`
  makes "next Friday" clear; otherwise use a date), "every Monday and Thursday at 7 am".

## How Chirii's replies are marked

The program checks the reply text against the task's `rules`:

- `must_include`: every fact listed must appear in the reply.
- `must_not`: none of these may appear.
- `order`: these facts must appear in this order (for plans: earliest first).
- `flag_clash`: the reply must point out an overlap (words like "clash", "overlap", "same time", "conflict").
- `abstain`: the reply must say it doesn't have this saved ("I don't have…", "nothing saved about…", "you haven't told
  me…").
- **Always, for every task:** the reply may not contain a date, amount, day count or weekday that isn't justified. A
  date or amount is justified when it is in the memory (the `item` fields below), is today, is in the task's `allow`
  list, is a sum of saved amounts, is a start or end of the period the question names, or is a day a saved weekly
  routine falls on. So **every date and amount that a right answer needs must be in some `item` or in `allow`.**
- Chirii shows a list of more than 12 lines as a summary (how many, the dates spanned, the first 5 entries in date
  order, the totals, "…and N more"); what is shown first is what is marked. If a task's right answer is a long list,
  write `must_include` for what the summary must contain (the first entries and the total), not for every entry.

## File format

Write two files:

1. **`heldout_tasks.jsonl`**: one JSON object per line, 30 lines, in this order: for each type in the order of the
   table above, the 5 people in the order above. `id`s `H-01` to `H-30`.
2. **`HELDOUT_TASKS.md`**: the same 30 tasks, readable: for each, the person, type, today (with weekday), the
   memories with their `said_on`, the question, the rules in words, and the `note`.

Each line of `heldout_tasks.jsonl`:

| Field | Value |
| :--- | :--- |
| `id` | `"H-01"` … `"H-30"` |
| `persona` | `"priya"`, `"lakshmi"`, `"ramesh"`, `"sharada"` or `"arjun"` |
| `type` | `"lookup"`, `"date_window"`, `"plan"`, `"unknown"`, `"semantic"` or `"update"` |
| `today` | `"YYYY-MM-DD"` |
| `question` | the question, exactly as the person types it |
| `memory` | a list of `{"said": "...", "said_on": "YYYY-MM-DD", "item": {...}}`, in the order they were said |
| `rules` | see below |
| `allow` | a list of extra facts the reply may mention (usually `[]`): `{"date": "YYYY-MM-DD"}` or `{"amount": 1200}` |
| `note` | one line for the human checker: what the right answer is and why |
| `scored` | `true` |

**`item`** is how a careful reader understands the sentence. Fields (leave out what doesn't apply):

| Field | Value |
| :--- | :--- |
| `type` | `bill`, `appointment`, `task`, `medicine`, `routine`, `important_date`, `exam`, `travel`, `shift`, `income`, `receivable` (money owed to the person), `note`, `person` (who someone is), `profile` (about the person themself), or `update` (a change to an earlier sentence) |
| `title` | a short name: `"gas bill"`, `"Divya's PTA meeting"` |
| `date` | `"YYYY-MM-DD"`; for a range, the first day |
| `end_date` | the last day of a range |
| `time` | `"HH:MM"`, 24-hour |
| `amount` | rupees, a whole number (`1860`, not `"₹1,860"`) |
| `day_of_month` | for "on the 5th of every month": `5` |
| `weekdays` | for weekly things: `["Monday", "Thursday"]` |
| `status` | for an `update`: `"paid"`, `"cancelled"`, `"done"`, or leave out and give the new `date` / `time` / `amount` |

**Facts** in `must_include`, `must_not` and `order`:

| Fact | Matches |
| :--- | :--- |
| `{"date": "2027-03-12"}` | the date written any usual way ("12 March", "Friday 12 March", "12/3") |
| `{"amount": 1860}` | ₹1,860 written any usual way |
| `{"time": "18:00"}` | 6 pm, 6:00 pm, 18:00 |
| `{"text": ["on call", "on-call"]}` | any one of these phrases (whole words, any case) |
| `{"any_of": [fact, fact, ...], "min": 2}` | at least `min` of the listed facts |

`rules` keys: `must_include`, `must_not`, `order` (lists of facts), and `flag_clash`, `abstain` (`true` when needed).
Use `text` facts for names and words where a right answer could phrase things differently; give alternatives.

### One example line (format only: a different person and situation; do not reuse it)

```json
{"id": "H-00", "persona": "example", "type": "lookup", "today": "2027-02-10", "question": "how much is the water purifier service?", "memory": [{"said": "Aquaguard service is due on 18 Feb, ₹650.", "said_on": "2027-02-01", "item": {"type": "bill", "title": "water purifier service", "date": "2027-02-18", "amount": 650}}, {"said": "Car wash every Sunday at 8 am.", "said_on": "2027-01-20", "item": {"type": "routine", "title": "car wash", "time": "08:00", "weekdays": ["Sunday"]}}], "rules": {"must_include": [{"amount": 650}]}, "allow": [], "note": "₹650, due 18 Feb.", "scored": true}
```

## Before you finish

- 30 lines, ids `H-01`–`H-30`, 5 of each type, each person once per type.
- Every date and amount in a right answer is in an `item` or in `allow`.
- Every `must_not` fact is really wrong for that question, and every `must_include` fact is really needed.
- Each `unknown` task: the thing asked is in no sentence at all.
- Each `update` task: the later sentence's `said_on` is after the sentence it changes, and both are before `today`.
- Valid JSON on every line (UTF-8; ₹ is fine).

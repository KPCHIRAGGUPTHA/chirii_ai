# Chirii AI: progress log

A short entry per stage: what was done, why, the results, and what comes next. Written for the faculty mentor.
The plan is in `chirii/PLAN.md`.

## S0 – Setup (21 November 2026)

**What was done**
- Created the `chirii/` folder for the proof of concept, separate from the research work (Phases 0–31).
- Set up a Python 3.12 environment with every package version pinned in `requirements.txt` (58 packages), so the
  same setup can be rebuilt on any machine, including the deployment server.
- Added settings that load from a private `.env` file. The app refuses to start with a missing, placeholder or short
  secret key.
- Added privacy-safe logging: code can log only approved fields such as timings and counts, never message text, and
  user ids are replaced by a one-way code.
- Added git rules so `.env`, user databases, search indexes and model files can never be committed, with tests that
  check it.
- Wrote the plan (`chirii/PLAN.md`), the pass/fail rules fixed so far (`chirii/CRITERIA.md`), the licence register
  (`chirii/data/SOURCES.md`) and a draft privacy notice (`docs/PRIVACY.md`).
- Corrected `DATA_NOTICES.md`: alpaca-cleaned is now marked non-commercial, matching `FINAL_REPORT.md`, and five
  datasets used by earlier fine-tunes that were missing from it are now listed.

**Why**
- Earlier work showed that licence problems are expensive to find late: five fine-tuned models became research-only
  because of non-commercial training data. The licence register now comes before any data.
- The app will hold bills and health notes, so privacy is built in from the first stage rather than added at the end.

**Results**
- 12 of 12 setup tests pass: settings, logging and git rules.
- Packages import correctly on Python 3.12.14 (FastAPI 0.165.0, FAISS 1.16.0, NumPy 2.5.5).

**Decisions**
- Start with plain Qwen2.5-1.5B-Instruct plus memory, and fine-tune only if the S1 eval shows it falls short.
- Deploy with an 8-bit (Q8_0) model by default; earlier tests rejected 4-bit for a fine-tuned model.
- Use `fastembed` for note search instead of `sentence-transformers`: it runs the same small embedding model
  without installing PyTorch, which keeps the server small.

**Next (S1, 24 Nov – 4 Dec)**
- Write the 50-task everyday-assistant eval before any of the system exists, and propose its pass bar for approval.
- Prepare the December GPU job: benchmarks of plain Qwen2.5-1.5B and MiniGPT.

## S1 – Eval, downloads and GPU job (21 November 2026, eval awaiting the owner's review)

**What was done**
- Wrote a 50-task eval before any of the system exists (`chirii/eval/s1_tasks.jsonl`, review copy
  `chirii/eval/S1_TASKS.md`). It covers five everyday Indian users (a working professional, a homemaker, a shop owner,
  a retired teacher and a student), with 10 tasks each across six types:
  - exact lookup (10)
  - date windows (8)
  - meaning-based search (8)
  - day or week plans, including clashes (8)
  - updates and corrections (8)
  - "not in memory" (8)

  The situations are everyday ones: bills in ₹, LIC, EMIs, fees, the gas cylinder, doctor visits, medicine timings,
  festivals, ration lists, shifts and KYC. Five unscored Hinglish tasks are kept as a separate set.
- Wrote an automatic marker (`chirii/eval/scoring.py`) that accepts different wordings of the same fact ("12
  December", "Dec 12", "12/12"; "₹1,860", "Rs 1860"). It fails any reply that mentions a date or rupee amount not in
  the person's memory.
- Proposed the pass bars, with a note on how much noise 50 tasks carry (`chirii/CRITERIA.md`, PROPOSED section).
- Downloaded llama.cpp (b12359, Windows CPU), the official Qwen2.5-1.5B-Instruct Q8_0 GGUF and the
  all-MiniLM-L6-v2 embedding model. Checked each file's SHA-256 and recorded them in `chirii/data/SOURCES.md`. Model
  files stay out of git.
- Wrote the December GPU job (`chirii/gpu/benchmarks/`, instructions in `RUN_ON_GPU.md`). It re-runs ARC-Easy,
  HellaSwag, MMLU and GSM8K for plain Qwen2.5-1.5B and the two MiniGPT models with the Phase 18 settings. It also
  scores Qwen on all 1,319 GSM8K questions, which gives the baseline for the 2-point fine-tuning rule.

**Why**
- Writing the test before the system stops the test being shaped to fit the system.
- The "no invention" rule exists because a made-up bill date or amount is the most harmful mistake a memory
  assistant can make.
- The 500-question GSM8K score from Phase 18 has a standard error of about 2.1 points, as large as the 2-point rule
  itself. The full 1,319-question set brings it down to about 1.3.

**Results**
- 65 of 65 tests pass, including:
  - checks that every required date and amount exists in the person's memory, that date windows and weekdays are
    right, and that "not in memory" questions really can't be answered from memory;
  - 21 sample replies marked as intended;
  - the GPU comparison script reproducing the Phase 18 numbers exactly from Phase 18's own files.
- Smoke test on the laptop CPU (not an eval run): Qwen Q8_0 runs at about 16 tokens/s output and uses about 1.75 GB
  RAM. Embeddings work.

**Next**
- The owner reviews the 50 tasks and the pass bars. Nothing is run against the eval until both are approved.
- The owner runs the GPU job on the college PC this week and pushes the results.
- S2: the memory store (SQLite plus a FAISS index) and the first eval run of plain Qwen with memory.

## S1 – Approval (21 November 2026)

The owner approved the eval and the pass bars, with three changes:
- **Worked-out values:** the marker now accepts values worked out correctly from memory: today's date and weekday,
  sums of amounts, day counts ("in 5 days", "2 days late") and the weekday next to a date. A wrong sum, count or
  weekday fails. Tests cover each, including a wrong sum, and check that an empty or evasive reply fails every task.
- **S7 decision rule:** below 35/50, fine-tune. At 35–39/50, label each failure (retrieval, prompt/formatting, scorer
  or model behaviour), fix retrieval and prompts in code, and rerun once. Fine-tune only if more than 10 model-behaviour
  failures remain.
- **Held-out set:** the 50 tasks are now the development set. In S6, 30 new tasks are written and committed with their
  marking rules before any run. They are run once on the final system, and that score is the main result.

All three are in `chirii/CRITERIA.md`, marked APPROVED. No system had been run on the tasks before approval.

## S2 – Memory store (21 November 2026)

**What was done**
- **Sentence reader** (`chirii/app/memory/extract.py`): turns what the user says into a structured item: its kind
  (bill, appointment, medicine, routine and so on), dates, times, amounts in ₹, and repeats ("the 5th of every
  month", "every Tuesday and Thursday"). This is plain code, not the language model, because dates and amounts must
  never be guessed.
- **Memory store** (`chirii/app/memory/store.py`): each user gets one folder with a SQLite database (the source of
  truth) and a FAISS index for search by meaning.
  - Updates such as "I paid the water bill" or "the doctor's appointment moved to 21 December, same time" change the
    right earlier item, and the old value is kept in a history table.
  - If no earlier item clearly matches, the update is kept as "unlinked" rather than guessed.
  - A date-range query lists everything due in a window, including monthly and weekly repeats.
- **Delete all my data:** removes the user's whole folder (rows, vectors and history) and leaves other users alone.
- **Measurement script** (`chirii/eval/run_s2_memory.py`), with results in `chirii/results/s2/`.

**Results**

All measured on the laptop CPU with all-MiniLM-L6-v2. The first three rows are on the S1 development memories, which
the parser was written against, so they are optimistic. The last two rows use new data and are the honest estimates.

| Check | Result |
| :--- | :--- |
| 66 memory sentences read fully right | 66/66 |
| 8 updates linked to the right item and applied right | 8/8 |
| 6 date windows with nothing missing and nothing extra | 6/6 |
| **22 new sentences read fully right, first try** | **20/22** (both misses were the item's kind: "tax" wasn't a bill word and "IU" wasn't a medicine word; fixed, now 22/22) |
| **8 new questions sharing no key word with their note: note ranked first** | **8/8** (the word-matching test stand-in managed 1/8) |

- Adding a sentence takes about 15 ms and a search about 16 ms. One user with 9 items uses about 30 KB of disk.
- 132 of 132 tests pass. The new ones cover users kept apart, folder names that don't reveal the user, nothing left
  after "delete all", no memory text in logs, and the search index staying in step with the database.

**Problems found and fixed**
- "Delete all my data" failed on Windows: one database connection was left open, so the file couldn't be removed.
  Fixed, and a test now catches it.
- "I paid the gas bill" would have marked the *water* bill paid, because both mention "bill". Now an update that names
  something the item doesn't mention is not linked to it.

**Limits, stated plainly**
- The sentence reader only knows the patterns it was written for. Unusual phrasing will be read as a plain note,
  which is safe but less useful. S3 will show how often this matters.
- The test memories hold 9–12 items per person. Search over hundreds of items is not measured yet.
- Files are not encrypted on disk yet (see `docs/PRIVACY.md`).

**Next: the early demo slice (by about 2 January)**
- A minimal web page where you can add a memory, ask a question and get an answer.
- The answer will come from plain Qwen given the retrieved memories.

## Owner decisions (21 November 2026)

- **S7 rule for bars B and C:** a miss on B or C now gets the same treatment as bar A at 35–39: failure analysis, one
  fix round, one rerun. Fine-tune only if the failures labelled model behaviour alone still miss the bar: more than
  10 for A, more than 1 of 8 for B, more than 2 for C (`chirii/CRITERIA.md`).
- **Hybrid extraction** in the demo slice (done, below).
- **Scale check** with about 500 memory items per user, in S3.
- **Encryption at rest** and a tester consent note, proposed in S5. No real users before that.

## Demo slice – Add, ask, answer (21 November 2026)

**What was done**
- **Web page and API** (`chirii/web/index.html`, `chirii/app/api/main.py`):
  - add a memory, see all memories, ask a question, delete one memory or everything;
  - items Chirii wasn't sure how to read are marked "please check";
  - one demo user, no login yet, requests from other computers are refused.
- **Answering** (`chirii/app/assistant.py`):
  - code picks the memories: the 5 closest by meaning plus everything due in the next 14 days;
  - code writes each as a fact line with the current date and weekday, time, amount (₹1,95,000 style) and status;
  - Qwen words the answer and must say "I don't have that in your memory" when the answer isn't there.
- **Hybrid sentence reader** (`chirii/app/memory/hybrid.py`):
  - the rules run first, and Qwen is asked only when the rules give up;
  - Qwen's JSON reply is limited by the server to values the code found in the sentence;
  - code then re-checks every date, time and amount;
  - if Qwen chose the kind of item, the user is asked to confirm it;
  - every use of Qwen is logged as counts only.

**Results: extraction, rules only against hybrid** (Qwen2.5-1.5B Q8_0, laptop CPU)

| Run | Kind of number | Rules only | Hybrid | Wrong and not marked for checking |
| :--- | :--- | ---: | ---: | ---: |
| Set A, 20 new sentences | **fresh, first try** | **5/20** | **6/20** | 7 |
| Set B, 20 new sentences, written after the set A fixes | **fresh, first try** | **13/20** | **13/20** | 7 |
| Set A after all fixes | development | 19/20 | 17/20 | 1 |
| Set B after all fixes | development | 16/20 | 14/20 | 1 |

Each set was committed before its first run. Full tables and Qwen's raw replies are in `chirii/results/slice/`.

**What this shows**
- **The rules were too narrow at first.** Casual phrasing ("tmrw", "by the 15th", "8k", "subah 10 baje") beat them
  on 15 of 20 set A sentences.
- **Set A led to general fixes:**
  - short forms like "tmrw";
  - days without a month;
  - plain numbers as rupees in sentences about paying;
  - Hindi time words, and "kal" read by its verb (gaya → yesterday, jaana hai → tomorrow);
  - fees counted as bills even when "bus" appears.

  On set B, the next new set, the rules got 13/20 on the first try.
- **Qwen 1.5B adds little as a fallback reader.**
  - Unconstrained (set A, first try), it often added values that weren't in the sentence: weekdays, or today's date.
    The code check caught every one, as designed.
  - Constrained to the sentence's own values, it fills fields well (it found the ₹2,000 in "owes me 2000").
  - It chooses the kind of item poorly: it called a ration list and a Netflix subscription "money owed to the user".
  - On the development runs, hybrid is 2 below rules only, because Qwen relabels plain notes.
- **The number that matters most for trust is "wrong and not marked".** On the first tries it was 7 of 20, mostly
  because the rules treated any dated sentence as an important date without saying they were unsure. Two fixes:
  a date alone no longer decides the kind, and Qwen's choice of kind is always marked for checking. After them it
  is 1 of 20 on each set (development numbers: a fresh check of these fixes comes with the next new sentences, in
  S3).
- **Decision:** keep the hybrid reader, because it is safe (the user is asked whenever Qwen decided something), but
  don't rely on it. The rules do most of the work. A larger model on a GPU server could make the fallback worth more
  later.

**Demo check, end to end** (`chirii/results/slice/DEMO_RUN.md`): 7 fictional memories, 5 questions, read by hand.
- All 7 memories were read as intended. "I paid the broadband bill" marked the right bill paid.
- 4 of 5 answers were right, including "I don't have that in your memory" for a PAN number.
- **One wrong answer:** asked when Dad's surgery is, Qwen gave today's date (21 November) instead of 3 December.
  - The memory line it was given said "Thursday 3 December 2026"; the error is in the model's reading, not the
    retrieval.
  - With only that one memory saved, it answered correctly twice.
  - This is exactly the invented-date failure that bar C exists for. It goes into S3, which will show how often it
    happens on the 50 tasks.
- One answer repeated the prompt's own wording ("The user said…"), which is a formatting issue for S3.
- Answers took 4–9 seconds on the laptop CPU. Adding a memory took about 20–50 ms.

**Tests:** 166 of 166 pass. The new tests cover:
- when the hybrid reader calls Qwen, and that the schema offers only values from the sentence;
- that invented amounts are rejected and the model being down is handled;
- that fallback logs hold no text;
- fact lines, Indian amount format and the answer prompt;
- the web API end to end, including "delete all" leaving no files and requests from other computers being refused.

**Next: S3 (ask with context)**
- Run the 50-task eval on plain Qwen with and without memory; the first official run.
- Add the safety layer.
- Fix the problems found here: the invented date and the prompt wording.
- Run the 500-item scale check.

## Demo slice, part 2 – Confirmation card and answer fact-check (21 November 2026)

The owner approved the slice and asked for two product fixes, plus a method for S3.

**What was done**
- **"Got it" card** after every saved memory: "Got it: Bill, ₹1,860, due Saturday 12 December 2026. Correct? [Yes]
  [Edit]".
  - The card says what Chirii is unsure of: a kind of item Qwen guessed, an update it couldn't match, or a missing
    amount, due date, date or time.
  - Edit opens the fields already filled in. Every value is checked (real date, real time, whole rupees, day 1–31),
    and every change is kept in the history.
  - Memories nobody has confirmed are saved but marked "unconfirmed". An answer that relies on one says so underneath.
  - Older databases get the new columns automatically when opened.
- **Answer fact-check guard** (`chirii/app/guard.py`): before an answer is shown, every date, rupee amount, day count
  and weekday in it must come from the memory lines Qwen was given, or be worked out correctly from them.
  - It uses the S1 marker's own rules, plus one stricter rule: today's date may appear only when the answer is about
    today. That is exactly the surgery-date failure, which the marker's "today is always allowed" rule would miss.
  - If the check fails, Qwen gets one retry with only the top 3 lines. If that fails too, code answers by quoting what
    the user saved.
  - Logs record only whether the answer passed, was retried or fell back.
- **Fewer memory lines:** only memories relevant to the question are sent, i.e. close matches by meaning, plus
  whatever falls in a time window the question names ("this week", "tomorrow", "next 5 days"). The old way stays
  available for S3's baseline run.

**Results**
- **The surgery-date failure, on the real model** (`chirii/results/slice/GUARD_CHECK.md`):

  | Setting | Result |
  | :--- | :--- |
  | Old way: 5 lines, no guard | Wrong again ("Saturday 21 November", today's date) |
  | 5 lines, with guard | Wrong first answer caught; the retry gave "Thursday 3 December 2026" |
  | Trimmed lines, with guard (the app now) | Right first time |

  These are single runs: they show the mechanism works, not how often it is needed. S3 will measure that.
- **Demo check again** (`DEMO_RUN_guard.md`): all 7 memories were read as intended, and all 5 answers were right
  (before: 4 of 5). Each answer used 1 memory line instead of 5–7, and took 0.7–5.0 s instead of 4.2–9.2 s.
- **A rules gap found while testing the card:** "Electricity bill due on 12 December" (no amount) was saved as a
  plain note, so the card couldn't say the amount was missing. It is now a bill, and the card asks for the amount.
  This changed the rules after the extraction measurements, so the next fresh set in S3 covers it.
- 205 of 205 tests pass.

**S3 method** (recorded in `chirii/CRITERIA.md` before any S3 run)
- **Two runs on the 50 development tasks:** (a) the slice as first built, with no guard, as the honest baseline that
  the S7 decision uses; (b) trimmed retrieval with the guard. Plus plain Qwen with no memory.
- **Failure analysis** by type for each run.
- **Bar C in run (b) is partly guaranteed by design**, because the guard uses the marker's rules, so the report will
  say so.
- **No prompt tuning against individual tasks.**
- **Next fresh extraction set:** target 2 or fewer of 20 wrong and not flagged.

## S3 – First official eval run (21–22 November 2026)

**Method:** as corrected by the owner before the run (`chirii/CRITERIA.md`, "S3 method").
- The S7 decision is made on run (b), the system that will ship: trimmed retrieval with the answer fact-check guard.
- Run (a), the slice as first built (no guard), is the "before" picture.
- Plain Qwen with no memory is a third column for comparison.
- Bar C' (a code-built fallback in at most 5 of 50 tasks) replaces bar C for run (b).

The runner was committed before the run. Every reply was read by hand, and the hand verdicts and failure labels were
committed before any fix. Full report: `chirii/results/s3/S3_RESULTS.md`.

**Results** (marker's score, with the hand check in brackets)

| | (a) Before | **(b) Ship** | Plain Qwen, no memory |
| :--- | ---: | ---: | ---: |
| **Bar A**, at least 40/50 | 28 (27) | **28 (28): miss** | 10 (9) |
| **Bar B**, at least 7/8 | 5/8 | **5/8: miss** | 8/8 |
| Bar C, at most 2/50 | 2 | 0 (enforced by the guard, not evidence) | 1 |
| **Bar C'**, at most 5/50 fallbacks | – | **3: pass** | – |
| Guard: clean / retried / fallback | – | 47 / 0 / 3 | – |
| Median time per answer | 8.7 s | 3.7 s | 4.8 s |

| Task type | (a) | (b) |
| :--- | ---: | ---: |
| Exact lookup | 8/10 | 8/10 |
| Search by meaning | 8/8 | 8/8 |
| Updates | 6/8 | 4/8 |
| Not in memory | 5/8 | 5/8 |
| Date windows | 0/8 | 1/8 |
| Plans | 1/8 | 2/8 |

**Decision: S7 (fine-tuning) is triggered.** Bar A in run (b) is 28/50, below 35.
- The fix round would not change this. 16 of run (b)'s 22 failures are on the model's side, labelled "model
  behaviour" or "wrong item used". The hand check adds 2 more: answers that say "I don't have that" and then give it.
- That makes 18, more than the 10 allowed. Even if the 5 plan-order and clash failures move to code in S4, 13
  remain.

**What the run shows**
- **Memory is essential.** Plain Qwen with no memory is useless here: 9 of 50 on the hand check, mostly "I don't
  have access to your data".
- **Looking up one memory works well.** Exact lookups pass 8/10 and searches by meaning 8/8. The trimmed retrieval
  usually sent just 1–3 lines, and answers took a median of 3.7 s instead of 8.7 s.
- **Combining several memories mostly fails:** "what's due this week", plans and paid/unpaid lists. Qwen 1.5B adds
  items from outside the window, misses ones inside it, treats money owed to the user as a payment, ignores "PAID",
  and doesn't notice clashes.
- **"Not in memory" fails when a related memory is close.** Asked about an electricity bill when only a water bill is
  saved, Qwen gives the water bill's date. Asked about a Compiler Design exam, it gives the FAT exams' date. The
  guard can't catch this, because the wrong fact is in memory.
- **The guard rarely had to act (3 of 50).** Its retry never rescued an answer. One fallback came from a real error
  (today's date again); the other two came from marker rules that are too strict (see below).

**Code problems found** (not fixed yet; listed in the report)
- Fact lines show only an item's first date.
- Daily medicines can crowd real items out of a time window.
- "Plan my day" ignores tomorrow's preparation.
- The fallback quotes old dates and doesn't check clashes.
- Two marker errors: a month followed by a numbered list item reads as a date, and the empty days of the asked week
  count as invented.

**Hand check:** 7 disagreements with the marker, all listed in the report. Net effect: run (b) 28 → 28, run (a)
28 → 27, plain Qwen 10 → 9.

**Next**
- The rest of S3 still to do: the safety layer, the 500-item scale check, and the fresh extraction set (target:
  wrong and not flagged ≤ 2/20).
- S7 needs its own plan, for the owner to approve: training data (commercial licences only), the IndiaAI GPU job, and
  the 2-point limits on GSM8K and function calling, against the GPU job's baselines.
- The code problems above are fixed in S4 whatever happens with S7. Fine-tuning won't fix them.

## S7 deferred; S4 – Structured reasoning moved into code (22 November 2026)

**Decisions recorded before any S4 work** (`chirii/CRITERIA.md`)
- **S7 is triggered under the rule, then deferred by the owner.** The reason, in the owner's words: the model-side
  failures are concentrated in filtering, counting and combining memories, which are being moved to code.
- **S7 reconsideration rule.** S7 is reconsidered on the 30 held-out tasks after S4, with bars A ≥ 24/30, B ≥ 4/5
  and C' ≤ 3/30 code-built fallbacks, and no further deferral.
  - A conflict in the rule text was resolved strictly: B at 3/5 triggers S7 directly.
  - One open point is left for the owner: what happens if C' misses by more than one.

**What was built**
- **Question router:** sends each question down one of five paths (plan, date window, status, lookup, meaning). The
  rules decide; Qwen is asked only when no rule matches, and can only answer with one of the five names.
- **Code-built answers** (`chirii/app/planner.py`):
  - date windows filtered by kind (to pay, owed to you, to do, family dates), leaving out paid items, with totals;
  - day and week plans in time order, with clashes found by code, "coming up tomorrow" and medicines;
  - "is it paid / did I…" answers from each item's status and history.

  Qwen only rewords these texts. If the rewording loses an item, date, time or amount, changes the order, drops a
  clash warning or fails the guard, it gets one retry, and then the code's own text is shown.
- **Specific-item check:** if a question names something (a name, or a thing like electricity or LIC) and no matching
  memory mentions it, Chirii says it hasn't got that saved instead of offering a different item.
- **Fixes:**
  - the 4 code bugs found in S3: all dates shown, medicines no longer crowding out windows, no old dates in fallbacks,
    plans looking at tomorrow;
  - the 2 marker/guard errors, with tests that fail on the old marker. S3 re-marked: unchanged except run (a)'s bar C,
    2 → 1. The original scores stay official;
  - clearer item titles.
- **GPU diagnostic job** (`chirii/gpu/qwen7b/`): run (b)'s setup with Qwen2.5-7B-Instruct Q8_0, for the owner to run
  on the GPU PC. No decision is attached to it.

**Run (c): the S4 system on the 50 development tasks** (`chirii/results/s3/RUN_C.md`). **This is a development
number, not a fresh one:** S4 was built in response to these tasks' failures.

| | (a) Before | (b) S3 ship | **(c) S4** |
| :--- | ---: | ---: | ---: |
| Bar A (current marker) | 28/50 | 28/50 | **42/50** |
| Bar A (hand check) | 27/50 | 28/50 | **46/50** |
| Bar B | 5/8 | 5/8 | **8/8** |
| Code-built fallbacks | – | 3 | **10** |
| Median time | 8.7 s | 3.7 s | 4.4 s |

**By task type** (run (c)): plans 8/8, date windows 5/8, not-in-memory 8/8, lookups 8/10, meaning 8/8, updates 5/8.

**What is still wrong**
- **10 fallbacks**, against 5 allowed on the held-out set. Eight come from the S4 code itself:
  - list headers name the period's start (today) and end dates, which the guard treats as a misused today's date
    and an invented date;
  - a weekly routine's date isn't counted as a memory date.

  These header dates also make 2 tasks fail the marker. Only 2–3 fallbacks are real rewording losses by Qwen.
- **"Is anything due tomorrow?"** missed money owed to the user, because the router read "due" as a payment word.
- **Three model-behaviour failures remain on lookups**, all wrong dates: two old dates after an item was moved, and
  the wrong one of an item's several exam dates. The guard can't catch them, because those dates are in the memory.

**Next**
- Fix the header dates, the weekly-routine dates and the "due" routing. Any rerun of the 50 tasks stays a
  development number.
- Still pending from S3: the safety layer, the 500-item scale check, and the fresh extraction set.
- The owner runs the benchmark and 7B jobs on the GPU PC.

## S4b – Code lists shown as built, current values only (22 November 2026)

**Owner decisions recorded first** (`chirii/CRITERIA.md`)
- On the held-out set, bar B at 3/5 triggers S7 directly.
- For bar C', if there are more than 3 fallbacks, each is labelled by cause. S7 is triggered only if model-caused
  fallbacks are more than 3 of 30.
- Marker change, recorded before run (d): the period ends named by the question, and the dates of weekly routines,
  count as worked-out values. Runs (a) to (c) were re-marked next to their originals (`REMARKED_S4B.md`): run (c)
  goes from 42 to 44.

**What changed**
- **Code lists, totals, plans and headers are shown exactly as built.** Qwen writes only a short lead sentence and an
  optional closing line, and only those are checked.
- **"Due" covers payments and money owed to you.** Money owed to you is listed separately as "To receive". A list
  header names dates only when the question named the period.
- **A moved or corrected item shows only its current values.** The old value appears only if the question asks for
  it; otherwise an old date in an answer counts as invented and gets the retry.
- **Items with several dates list every date with its label**, or only the next one when the question says "next".

**Run (d)** (`chirii/results/s3/RUN_D.md`): **a development number**, on the same 50 tasks S4 was built from.

| | (a) | (b) | (c) | **(d)** |
| :--- | ---: | ---: | ---: | ---: |
| Bar A (current marker) | 28 | 28 | 44 | **49/50** |
| Bar A (hand check) | 27 | 28 | 46 | **50/50** |
| Bar B | 5/8 | 5/8 | 8/8 | **8/8** |
| Invented values | 1 | 0 | 1 | **0** |
| Code-built fallbacks | – | 3 | 10 | **16** |
| Median time | 8.7 s | 3.7 s | 4.4 s | 3.4 s |

- Every task type is now 8/8, except lookups at 9/10. The one failure is S1-12, where the task rule asks for an
  amount the question didn't ask for. The three wrong-date lookups from run (c) are now right.
- **All 16 fallbacks are code-caused.** Qwen's lead restated dates, times or amounts from the list, which my S4b
  lead rule forbids. That rule is stricter than the owner's ("the guard checks Qwen's text").
  - Every restated value was correct; none counts as invented under the marker's rules.
  - The answer shown is the complete code-built list, so no answer was wrong. The user only loses the friendly lead.
  - About 3 leads also restate the period's start (today) and would still trip the guard's today's-date rule.

**Next:** decide the lead rule (check the lead with the guard only, as specified), then the owner's next items:
general chat, then people and profile memory. After those: the safety layer, the 500-item scale check, and the fresh
extraction set.

## Lead fix, cosmetics, and a fresh check on 20 new tasks (22 November 2026)

**What was done**
- **Lead check is now the guard only** (the owner's spec). It allows the start date of a period the question names.
  A lead that fails is dropped and the list is shown alone, counted as "lead dropped", apart from fallbacks.
- **Cosmetics:**
  - no clause splits after Dr., Mr., Mrs., Ms., Smt., Shri or Sri;
  - "Mrs. Deshpande owes you ₹3,450";
  - "the user" removed from every S4 prompt and answer template, with a test that scans them all. The version-1
    prompt and fact line are kept unchanged so runs (a) and (b) stay reproducible.
- **A second development set of 20 new tasks**: same 5 people, all 6 types, new situations and phrasings. Committed
  with its marking rules before any run (`chirii/eval/DEV2_TASKS.md`).

**Run (e1): first try on the 20 new tasks** (`chirii/results/dev2/RUN_E1.md`)

| | Run (d), the 50 tasks S4 was built on | **Run (e1), 20 new tasks** |
| :--- | ---: | ---: |
| Overall (marker) | 49/50 (98%) | **13/20 (65%)** |
| Overall (hand check) | 50/50 | **12/20 (60%)** |
| Fallbacks / dropped leads | 16 / – | **0 / 0** |

- **The 50-task set was saturated.** 13/20 means a true rate of roughly 43–82%, far below the 98% on the tuned set.
- **5 of the 7 failures are the code misreading a new phrasing.** The S4 router and period reader were written for
  the first set's wording. They miss:
  - "before the weekend" and "this weekend";
  - "what does my Saturday look like" ("Saturday" was taken as a name);
  - "is it still pending" (sent to the payment list, not a status answer);
  - "before we leave for Madurai";
  - "how much … this week".
- **The other 2 are on Qwen's side:** a missing flight time, and an "interview" date taken from the Infosys offer
  letter. The specific-item check matched "Infosys" but didn't check "interview".
- **Single-memory lookups still work** (6 of 7). The lead fix removed every fallback.
- **New bug:** a lead that hit its length limit was cut mid-word.

**What this means:** the weak point is now the rules that read the question. Fixing the six phrasings above would
be tuning to this set again. A more general fix, such as reading periods ("weekend", relative to a saved event)
more broadly, or letting Qwen pick the route and period within a strict schema with code checking it, should be
judged on new tasks again.

## S4c – One router for memory and general chat, and a fresh check (22 November 2026)

**Order of work, as agreed:** development set 3 was written and committed with its marking rules first (cbd3d93), then
S4c was built and tested on its own examples, then run (e3) was run once. The held-out 30 are still untouched.

**What was built**
- **One router.**
  - Memory routes: lookup, status, window, plan and meaning.
  - People: the route exists; the answers come in the people step.
  - General chat: maths, general knowledge and writing.
  - Mixed: memory and general together.

  A personal question (my, I, a saved name, a saved thing) never goes to pure general knowledge.
- **A general period reader:** weekends ("this weekend", "before the weekend"), weekday names, end of the month, next
  month, ranges, relative days, and periods measured against a saved event ("before the Goa trip"). Weekday names are
  never treated as item names.
- **Qwen-assisted routing** when no rule is sure. Its route, period and item are limited to allowed values and checked
  by code; otherwise a safe default is used.
- **The specific-item check matches all key terms** for lookups and status questions, and says what *is* saved when
  only part matches.
- **Lookups give the date and the time.**
- **General chat:**
  - maths: Qwen writes an expression and the calculator works it out;
  - general knowledge: "no live information" answers come from code;
  - writing: checked for made-up dates and amounts.

  Every answer carries a label (From your memory / Calculated / General knowledge / Writing help).
- Leads that hit the length limit are dropped, not cut.

**Run (e3), first try** (`chirii/results/dev3/RUN_E3.md`)

| | (d) 50 tuned | (e1) 20 new, S4b | **(e3) 30 new, S4c** |
| :--- | ---: | ---: | ---: |
| Memory tasks (marker) | 49/50 | 13/20 (43–82%) | **23/30 (59–88%)** |
| Memory tasks (hand check) | 50/50 | 12/20 (39–78%) | **25/30 (66–93%)** |
| General chat | – | – | **35/40 (74–95%)**: maths 12/15, general 14/15, writing 9/10 |
| Router tests | – | – | **14/15**; periods 15/15; all 85 items' routes 79/85 |

- **On new phrasings the period reader did its job.** Date windows passed 8/8 and plans 3/4 (the plan failure is a
  marker strictness), including weekends, weekday names, "next month" and event-relative periods.
- **Three partial-match answers fail only on the marker.** "I have your Goa trip … but nothing about flight" is the
  behaviour specified, but the marker's "not in memory" phrases don't include "nothing about".
- **Remaining failures:**
  - Code:
    - "postponed to 5 to 7 March" read as 7 March;
    - a status question picked the trip instead of the trip fee;
    - "GST" not seen as personal;
    - "Returned … today" not recognised as an update;
    - "tell me about my week" routed to people (case bug).
  - Qwen:
    - a lead saying "nothing to pay" above a ₹4,200 bill;
    - two wrong maths expressions that my constants check let through (÷ 100, 31 days for February);
    - "UPI = United Parcel Service".
- **Margins of error are wide** (30 and 40 tasks). These are development numbers on a fresh set, not the held-out
  result.

## S4d + S4e – Fixes, maths formulas, glossary, people memory; run (e4) (22 November 2026)

**Order of work, as agreed:**
1. The partial-match marker change, with run (e3) re-marked: 23 → 26 of 30, original kept.
2. Development set 4, written and committed before any code.
3. S4d, then S4e.
4. One run, (e4).

**What was built**
- **S4d:**
  - date ranges without "from";
  - money questions pick the item that has an amount;
  - capitalised saved names ("GST", "Patel") make a question personal;
  - "returned / cleared / submitted … today" are updates;
  - names are case-sensitive, so "my" is never a name;
  - a lead may not say "nothing" above a non-empty list;
  - **a maths formula library:** percentage, GST, discount, simple and compound interest, EMI, split, conversions,
    and lakh/crore, with days and weeks on the real calendar. Qwen only picks the formula and its numbers; code
    checks and computes, and shows the working in Indian format;
  - a 51-entry glossary of Indian everyday terms;
  - the owner's three answer labels.
- **S4e, people and profile memory:**
  - names and nicknames;
  - relationships, including Indian terms (Amma, Appa, Didi, Bhaiya, Thatha, Dadi…);
  - reverse relationships worked out by code, without assuming gender;
  - the person's own name in the third person;
  - family aliases that link to other memories ("Lalitha's eye check-up" finds "Amma's eye check-up");
  - people facts and links on the confirmation card.

**Run (e4), first try** (`chirii/results/dev4/RUN_E4.md`): margins of error are wide (10–15 tasks per group).

| Group | Marker | Hand check |
| :--- | ---: | ---: |
| Memory tasks | 11/15 | 10/15 |
| **People and profile** | **13/15** | **14/15** |
| Maths | 8/15 | 8/15 |
| General knowledge | 7/10 | 7/10 |
| Route right | 48/55 | |
| Label right | 52/55 | |

- **People memory worked on first try:** reverse relationships, Indian terms and gender-neutral answers.
- **The glossary fixed all 5 Indian-term questions.**
- **All 7 maths failures are code.** The router didn't send three questions to maths, and my checks rejected four
  correct choices from Qwen (an extra "1", conversion factors, and a right expression with the wrong formula name).
  No wrong number was shown.
- **The 3 general-knowledge misses are Qwen 1.5B's facts.**
- **Memory: new code gaps.** Word families (engaged/engagement, married/wedding), "appointment" required as a key
  term, subscriptions not seen as bills, ranges repeated per day, and "by when" misread as a period.

**Next:** fix these as general rules (development work, to be measured on a fresh set), then the safety layer, the
500-item scale check and the fresh extraction set.

## S4f – Matching by meaning, maths check; run (e5), the last development run (22 November 2026)

**Dev set 5 first** (e479bc2): 20 memory tasks with deliberately varied wording (synonyms, Hinglish, indirect), 10 maths,
10 people, committed with marking rules before any S4f code. `CRITERIA.md`: (e5) is run once, then the loop stops.

**S4f** (ee86ea6):
- **Matching by meaning:** a question word matches a memory by spelling, by stem (married/marriage,
  engaged/engagement, renew/renewal), by a small Indian synonym map (shaadi, dawai, kiraya, bijli…), or by embedding
  similarity above 0.66. The threshold was calibrated on 60 word pairs of my own, not on any dev set
  (`chirii/eval/calibrate_meaning.py`): 16/30 synonyms caught, 0/30 near-misses. Names match by spelling only.
- **Bill words:** renewals, subscriptions, premiums, EMIs and recharges are payments; Qwen can't call them money to
  receive.
- **"By when…"** is a deadline lookup, not a period. **Ranges** are listed once, with their dates.
- **Maths:** lakh/crore, weeks between, unit conversions and "me and N friends" route to maths. Code works the common
  shapes out by itself; if Qwen's choice agrees, its working is shown, if not, the code result is shown. A number that
  fails the checks is never shown.

**Run (e5), first try** (`chirii/results/dev5/RUN_E5.md`):

| Group | Marker | Hand check |
| :--- | ---: | ---: |
| Memory tasks | 14/20 | 16/20 |
| People and profile | 9/10 | 10/10 |
| **Maths** | **10/10** | **10/10** |
| Route right | 35/40 | |
| Label right | 39/40 | |

- Maths all right, all by code with working.
- Varied wording right in 8 of 9 tasks written for it.
- One wrong answer: a made-up leave time when "leave" isn't saved (D5-M12).
- Three router misses ("Jio plan", "Saturday look for me", "any bill this week").

**The development-set loop stops here.** Remaining gaps are in `docs/KNOWN_GAPS.md`, not fixed. **Next:** S5
(encryption at rest, login, tester consent note), then the safety layer, the 500-item scale check, the fresh
extraction set, and S6.

## S5, part 1 – Repository out of OneDrive; encryption at rest (22 November 2026)

**Repository moved out of OneDrive** to `C:\dev\chirii_ai` (copied with robocopy, 7,384 files, 0 failed; the venv
rebuilt from `requirements.txt` with the same package versions; all tests pass there). The OneDrive copy is left for
the owner to delete. The data key was created only after the move, so it never sat in OneDrive.

**Encryption at rest, option C** (owner's choice):
- Each user's memory is one SQLite database, held in memory while the app runs. On disk it is only one file,
  `memory.enc`, encrypted with AES-256-GCM (`cryptography`). Changed or truncated files are refused.
- Search vectors moved into the encrypted database; `index.faiss` no longer exists on disk.
- A key per user, derived from `CHIRII_DATA_KEY` (HKDF-SHA256) and the user's folder name. One user's file won't open
  as another's.
- Atomic writes (temporary file, flush, swap): a crash leaves the old file whole, and the copy in memory goes back to
  it.
- A per-user lock, shared by every store object for that user.
- A key check value: a wrong key stops the app before any data is touched, and a new key is never made when encrypted
  data exists.
- `python -m chirii.app.keys create | show | check | rotate | backup FOLDER`. Rotation re-encrypts next to the old
  files and is safe to interrupt; backups copy encrypted files only, never the key.
- Data moved to `%LOCALAPPDATA%\Chirii` (`CHIRII_DATA_DIR`). The fictional demo data was migrated, encrypted, and the
  plaintext copy removed from the repository.
- Tests: `test_s5_encryption.py` (13) and `test_s5_keys.py` (5); 471 pass.

**If the data key is lost, the data cannot be recovered.**

**Next:** login, the safety layer with the attribute check, the consent note, the 500-item scale check, the fresh
extraction set.

## S5, part 2 – Login (22 November 2026)

- **Accounts** are made by the owner on the pilot laptop (`python -m chirii.app.accounts add NAME`; the password is
  typed twice and never shown). No open sign-up.
- **Passwords:** Argon2id hashes only, in `accounts.enc`, itself encrypted with the data key. At least 10 characters,
  not the user name. 5 wrong passwords in 15 minutes lock the account for 15 minutes; unknown names take as long to
  refuse as wrong passwords.
- **Sessions:** a random 256-bit token in an HttpOnly, SameSite=Strict cookie. The server keeps only its hash. Sessions
  end after 12 hours idle, 7 days in all, on logout, and on restart. Requests that change something must come from the
  Chirii page (Origin check); the app still answers only this computer.
- **Consent before use:** at first login the consent note is shown, and nothing else works until it is accepted (a new
  version of the note is asked for again).
- **Per-user isolation:** the user comes only from the session, and each user has their own encrypted store. "Delete
  all my data" and "Delete my account" (password needed again) act on the logged-in user only.
- **Found and fixed while testing:** saving "Note." crashed the app (the title code split an empty string).
- Tests: `test_s5_login.py` (19), including users never seeing each other's memories, wrong or changed cookies,
  lockout, consent gating, logout, and requests from other sites. Checked in the browser on a throwaway instance:
  login → consent → save → logout.

## S5, part 3 – Safety layer and the attribute check (22 November 2026)

- **Self-harm layer** (reuses `phase14/safety_layer.py`): every question, and every memory being saved, is checked
  before Qwen. First-person self-harm or suicide intent gets a fixed supportive reply with Tele-MANAS 14416 and 112;
  Qwen is not called and no memory is used (label "Support"). phase14's two-signal rule keeps idioms and tech wording
  out. Added: Hinglish wording (marna chahta, jeena nahi, khudkushi, aatmahatya), with third-party wording ("mere dost
  ko…") suppressed. A saved note that fires is still saved, with the support message shown first.
- **Attribute check** (owner, after D5-M12): a question about an attribute or event ("leave", "interview", "joining",
  "fee", "surgery"…; "leave", "visit", "test" only when used as nouns) needs a memory that mentions it (exact, stem,
  synonym map, or meaning ≥ 0.66; an item's kind counts, so a bill is a "fee").
  - Before answering: these words are key terms, so a memory without them gives "not saved" or a partial match.
  - After answering: a code check on the final answer. A date, time or weekday in the answer must belong to a memory
    that mentions the event asked about; otherwise the answer is replaced ("I have Rahul's flight to Pune (…), but
    nothing about interview", or "I have Rahul's interview saved, but no date or time for it").
- D5-M12's recorded result is unchanged; the held-out set will measure this.
- Tests (`test_s5_safety.py`, own examples): all 22 phase14 positives fire and all ~100 hard negatives don't; Hinglish
  positives and negatives; memory-style sentences that must not fire ("life insurance premium", "death anniversary",
  "sleeping pills prescription"); the model is not called; the attribute check before and after answering, with the
  guard off to test the final check alone. 620 pass. A live check with Qwen on 7 own examples: all right.

## S5, part 4 – Tester consent note (22 November 2026)

`docs/PRIVACY.md` rewritten as the pilot privacy notice with the tester consent note (version `2026-11-22`). The app
shows the same ten points at first login and works only after they are accepted. They include the owner's three:
**use fictional data during the pilot**, **if the key is lost the data can't be recovered**, and **turn on device
encryption**, plus what is stored (questions and answers are not), where, how it is encrypted, how to delete (one item,
everything, the account), logs, the safety message, and the pilot's limits (the laptop holds the key). A test checks
that the app's note and `PRIVACY.md` say the same.

## S5, part 5 – 500-item scale check, first try (22 November 2026)

Method and bars committed before the run (4179bb5). The same 30 questions on 500 and on 40 memories
(`chirii/results/scale/SCALE_CHECK.md`):

| Bar | Result | |
| :--- | :--- | :--- |
| A. Accuracy at 500 vs 40 (≤ 2 lower) | 29/30 vs 29/30 | pass |
| B. Not saved, none made up at 500 | 0 of 5 | pass |
| C. Time (median ≤ 1.5×, 95th ≤ 2×) | 1.7 vs 1.6 s; 4.0 vs 2.4 s | pass |
| D. Storage at 500 (open ≤ 2 s, save ≤ 0.5 s, ≤ 5 MB) | 0.007 s, 0.025 s, 1.1 MB | pass |
| E. List questions finish | all 3, 0.1–8.7 s | pass |

- The one failure is the same at both sizes: "Wi-Fi" never matches because memory text is split at the hyphen
  (logged in `docs/KNOWN_GAPS.md`, not fixed after the run).
- Long lists are right but too long (30 bills before an event): a summary belongs to S6.
- Found while building the set, fixed before the run: the year rule read "12 December", said in June, as last
  December.

## S5, part 6 – Fresh extraction set C, first try (22 November 2026)

20 new sentences, committed before the run (8d75700). Target: wrong and not flagged ≤ 2 of 20.

| | Result |
| :--- | :--- |
| Rules only: fully right | 18/20 |
| Hybrid: fully right | 18/20 |
| Qwen called (rules gave up) | 4/20, all accepted by the code check |
| **Wrong and not flagged for the user** | **1/20: target met** |

- Not flagged: "Tuition for Riya every Tue Thu Sat 4pm" was read as Tuesday only (abbreviated weekdays with no commas
  or "and"). Logged in `docs/KNOWN_GAPS.md`.
- Flagged: "Mama ji ki shaadi 14 Nov…": Qwen chose "appointment" rather than "important date"; the date was right and
  the card asks the user to check.
- The S5 year rule worked on new sentences (31 January said in June → 2028).

**S5 is complete:** encryption at rest, login, the safety layer with the attribute check, the consent note, the
500-item scale check and the fresh extraction set. Next: S6 (chat UI and the held-out set).

## S5, part 7 – Rotating CHIRII_SECRET_KEY (22 November 2026)

The old secret was in the `.env` of the OneDrive copy, so it was replaced. The secret names each user's folder
(HMAC of the user name), and the folder name is bound into that user's encryption key, so a new secret needs every
file re-encrypted into its new folder. `python -m chirii.app.keys rotate-secret` (app stopped) does this for every
account and for the pre-login user `demo`: new secret into `.env` (old one kept as `CHIRII_SECRET_KEY_PREVIOUS` until
done), each file re-encrypted, read back and checked, then the old file removed. Interrupted, it finishes on the next
run; folders of no known user are left alone. The data key does not change. Tests: all users' data reachable after
rotation, also through login in the app; an interrupted rotation finishes; a wrong data key changes nothing. Run on
the real data: 2 folders moved (admin, demo), both open.

## S6, part 1 – Helper reply for someone worried about another person (22 November 2026)

Owner's change before S6. When someone writes about another person's suicidal thoughts or self-harm ("My friend said
he wants to die; how can I help him?", "Mere dost ko khudkushi ke khayal aate hain, kaise madad karun?"), Chirii gives
a fixed helper reply, labelled "Support", without calling Qwen or using memories: listen without judging, don't leave
them alone if in immediate danger and call 112, encourage professional help, Tele-MANAS 14416 (which also supports
people helping others), look after yourself. The same applies to a saved note. The person's own crisis is checked
first and is unchanged. Informational wording (statistics, essays, novels, "suicide squeeze") gets no reply.

Tests (own examples): 7 English and 5 Hinglish helper messages; 11 informational and everyday sentences, and phase14's
other hard negatives, get no reply; all phase14 positives still get the personal reply; in the assistant and on save.
Found and logged, not fixed: a Hinglish personal crisis that names a relative ("…papa ko mat batana") is suppressed
by the S5 personal rule (`docs/KNOWN_GAPS.md`).

## S6, steps 1–2 – Chat screen, in-session history and follow-ups (22 November 2026)

- **Chat screen:** a conversation view with the memories in a side column. Each reply shows its label ("From your
  memories", "· people" for people answers, "Calculated", "General answer", "Support"), what Chirii looked at, the
  time and how it was checked. A "Plan my day" button asks for today's plan. The labels themselves are unchanged (the
  dev-set runners score them).
- **History:** kept in memory per login session (at most 10 turns), never written to disk, so consent point 2
  ("questions and answers are not stored") still holds. Cleared on logout, "New chat", "Delete all my data" and
  "Delete my account". Users and sessions never share a conversation.
- **Follow-ups** (`chirii/app/followup.py`): rewritten into a question that stands on its own, then answered by the
  normal pipeline (safety, router, guard, attribute check); the chat shows "Understood as: …".
  - Rules first: "and the water bill?", "what about Priya?", "Priya ka kya?", "and tomorrow?" replace the subject or
    time of the previous question; "it", "he", "his", "uska" are replaced by its subject. Two person pronouns ("Did
    she call him?") or an impersonal "it" ("Is it going to rain?") are not guessed.
  - Qwen only when the rules can't, and its rewrite is used only if it adds no new name or number.
  - A crisis message is never rewritten.
  - `Assistant.ask` is unchanged, so single-question eval runs are not affected.
- Tests (`test_s6_followup.py`, own examples): 14 rule rewrites, 5 stand-alone questions left alone, the model
  rewrite and its check, and the API: follow-up answered as understood, New chat / logout / delete forget, sessions
  and users separate, nothing written to disk, crisis not rewritten, history capped. Live check with Qwen in the
  browser: 6 of 6 right.

## S6, step 3 – Phone layout (22 November 2026)

Below 900 px the page is one column: the chat fills the screen with the question box kept at the bottom, and the
memories open as their own panel ("Memories (12)", "← Back to chat"). Buttons and inputs are at least 44 px high
(small link buttons 36 px), inputs use 16 px text so phones don't zoom in, and the footer buttons stack. Checked in
the browser at 375 × 812: no sideways scrolling, all buttons at least 36 px. The same check found a wrong answer
(not layout): "before / by / until" a date is read as that one day. Logged in `docs/KNOWN_GAPS.md`, not fixed.

## S6, step 4 – Long-list summaries (22 November 2026)

A date-window or plan list with more than 12 lines (`chirii/app/summary.py`) is shown as: its header, one line with
how many items, the dates they span and how many clashes, the first 5 entries, every total line, and "…and N more.
Tap "Show all"". The complete list, exactly as code built it, comes with the answer and "Show all" shows it; nothing
is dropped without saying how many. In the summary, the "not confirmed yet" note names 3 items and the count.
On the scale check's lists: "Anything to pay before Sneha's sangeet?" becomes "31 items, from Saturday 19 June to
Thursday 9 December", 5 entries, "…and 26 more", "Total to pay: ₹68,017."; the Saturday plan becomes "10 items, with
5 clashes (see all)". Lists of 12 lines or fewer are unchanged. Tests (`test_s6_summary.py`, own examples): 8.
Checked live at 375 px: summary, then "Show all" gives all 15 entries.

Found while writing the tests, logged and not fixed: "in <month>" ("Which bills are due in July?") is not read as a
period. With the "before / by / until <date>" gap from step 3, these two period gaps are for the owner to decide on
before the freeze.

## S6, step 5a – Period fixes before the freeze (22 November 2026)

Owner's decision: fix the period gaps found in S6 before s6-final, as general rules, tested on own examples.
- **"before / by / until / till / up to" a date, a weekday or tomorrow** = from today up to that day; "before" leaves
  the day out, "by" and "until" include it. The next occurrence when no year is given ("before 10 Jan" in March →
  next January); "before Sunday" on a Sunday → next Sunday. "before the end of the month", "before the weekend" and
  saved events keep their rules.
- **"in / during / throughout <month>"**, "in the month of June", with a year or "next year" = that whole month; the
  next occurrence when no year is given (the current month counts).
- **"before <saved event>"**, the third report: diagnosed as a mistake in my check, not a bug. I had called the
  router without the saved memories, so no event could be found; through the app ("Anything to pay before Priya's
  birthday?", "before Amma's check-up", "before Amma's eye check-up") the list already ended the day before the
  event. Now covered by tests through the whole pipeline.
- **No regression:** all 824 tests pass (S4c/S4e: 61), and the routes and periods of all 255 dev-set and S1
  questions, with their memories loaded, are the same before and after (compared by script).
- Still open: a question with no "I/my" and no "anything/due" ("Bills to pay before 10 Jan?") goes to general
  knowledge even though its period is read (router gap D5-M15).

## S6, step 5b – The person's own crisis comes first (22 November 2026)

Owner's decision: fix, don't log. A message is read clause by clause (`chirii/app/safety/__init__.py`, `classify`):
- **Personal reply** when a clause has self-directed crisis wording ("kill myself", "end my life", "I'm suicidal"),
  or crisis wording with a first-person marker (I, me, my life, mujhe, main/mai with a first-person verb, meri
  zindagi, or a Hindi first-person verb ending: "soch raha hoon", "kar lunga"), however many other people are named
  elsewhere. "Mujhe marna hai, mummy ko pata nahi", "Main suicide kar lunga, papa ko mat batana" and "I want to die,
  don't tell my wife" now get the personal reply (the first two got none / the helper reply before).
- **Helper reply** only when the crisis wording is clearly someone else's ("he wants to die", "cutting himself", "a
  friend who self-harms", "dost ko khudkushi…") and no clause is the person's own.
- **Ambiguous** (the person and someone else in one clause, the wording not clearly either's: "Main aur mera dost
  dono marna chahte hain") → personal reply. Hinglish crisis words with no one else named → personal reply, as in S5.
- Talk about the topic (essay, statistics, novel) does not count as first-person crisis; phase14's own rules still
  run first and are unchanged ("I'm writing an essay on suicide prevention" gets the personal reply from phase14, as
  since S5: an over-support).
- Tests (own examples): 15 personal (the owner's three, Hinglish and English variants, one ambiguous) and 15 helper
  cases; all phase14 positives, hard negatives and the S6 helper and everyday cases still pass (285 in the file).

## S6, step 5 – Freeze: `s6-final` (22 November 2026)

The system is frozen at the git tag `s6-final` (the commit this entry is in). From here, the held-out set is written
from a spec by a fresh session (owner), checked for format only, reviewed and committed by the owner, and then run
once on this tag. All tests pass at the tag (854). Scoring rules: `chirii/CRITERIA.md`, "Held-out set: authorship,
freeze and scoring".

## S6, step 6 – Held-out spec for the fresh session (22 November 2026, after the freeze)

- `chirii/eval/HELDOUT_SPEC.md`: what Chirii is; 30 tasks = 6 types × 5 people, one per person per type; the 5
  people's profiles (no development-set memories); dates in 2027 with a `said_on` for every sentence; how replies
  are marked (including "no invented" facts and the summary of lists over 12 lines); the file format the marker reads,
  with one example of a different person; a checklist. A note at the top tells the owner to copy only this file into
  an empty folder outside the repository and start the fresh session there.
- `chirii/eval/check_heldout_format.py` (tests: `test_heldout_format.py`): the builder's format check. It reports
  errors (the marker or runner can't use a task) and warnings (a needed date or amount in no item and not in
  `allow`), runs the marker on every task, and never changes the file.
- Not system code: the tag `s6-final` is unchanged. Still to write before the run: the held-out runner (the same
  steps as the development runners, using each sentence's `said_on`), committed before the run.

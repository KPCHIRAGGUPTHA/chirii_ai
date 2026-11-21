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

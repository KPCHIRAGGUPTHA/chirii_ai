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

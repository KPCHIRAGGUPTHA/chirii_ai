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

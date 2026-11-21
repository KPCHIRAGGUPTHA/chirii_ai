# Chirii AI proof of concept: plan

Approved 21 November 2026. Target: a working demo by late February 2027 for VIT Vellore's Product Development Fund.

Stages are called S0, S1, … so they don't clash with the research repo's Phases 0–31. Each stage ends with a run,
the results shown, an entry in `docs/PROGRESS.md` and a commit on the `chirii-poc` branch.

## Key decisions

| Decision | Why |
| :--- | :--- |
| Start with **plain Qwen2.5-1.5B-Instruct** plus memory and retrieval, no fine-tuning | It is Apache-2.0 (commercial use allowed), and earlier fine-tunes cost reasoning ability (GSM8K 64.6% → 43.2%, Phase 23B) |
| Fine-tune **only if** plain Qwen with memory misses the S1 bar | Avoids paying the fine-tuning cost unless it buys something |
| **Q8_0 GGUF** by default; 4-bit only if it passes the S1 eval within noise | Q4_K_M lost 7 points on the earlier fine-tune (Phase 23B); Q8_0 never did |
| One model interface for llama.cpp locally and any GPU server later | Swapping to Qwen2.5-7B-Instruct or a rented GPU is a settings change, not a code change |
| **Qwen2.5-3B is excluded** | Its licence (Qwen Research License) is non-commercial |
| alpaca-cleaned, SciQ and PKU-SafeRLHF are **never used in commercial training** | Non-commercial licences (`DATA_NOTICES.md`) |
| Memory facts come from SQLite; the model only words the answer | Dates, amounts and appointments must never be invented |

## Where things run

- **Laptop (Windows, CPU):** the whole app — memory, API, UI and the Q8_0 model through llama.cpp.
- **College GPU PC (RTX A2000 12 GB, Linux), December only:** benchmarks of plain Qwen2.5-1.5B and MiniGPT on ARC-Easy,
  HellaSwag, an MMLU subset and GSM8K. Lab access may end on 31 December.
- **Rented GPU (IndiaAI Compute Portal):** only for an S7 fine-tune, if one is triggered.

**How a GPU job works:** each job is a self-contained script in `chirii/gpu/` with a `RUN_ON_GPU.md` giving the setup
commands, the exact run command, the expected runtime and the output files to commit. The owner runs it on the GPU PC
and pushes the results; the laptop pulls them.

## Stages and dates

| Stage | What | Target dates |
| :--- | :--- | :--- |
| S0 | Setup: folders, pinned Python 3.12 environment, settings, privacy-safe logging, licence register, plan | 21–23 Nov |
| S1 | **50-task** everyday-assistant eval (the development set), written before any system exists, plus pass bars (approved 21 Nov); GPU benchmark job ready by 1 Dec | 24 Nov – 4 Dec |
| S2 | Memory store: SQLite per user, sentence → structured item extraction (rules), FAISS search over notes, delete-all-data (done 21 Nov) | 5 – 15 Dec |
| Slice | **Early end-to-end demo** (done 21 Nov; confirmation card and answer fact-check guard added): add memory → ask → answer in a minimal web page. Hybrid extraction: rules first, Qwen → JSON → code validation only when the rules fall back to a plain note; measured on 20 new varied sentences, rules-only against hybrid, first try | 16 Dec – 2 Jan |
| S3 | Ask with context: safety layer; S1 eval run twice, **(a) baseline without the guard** and **(b) trimmed retrieval with the fact-check guard**, plus plain Qwen without memory, with a failure analysis by type (`CRITERIA.md`, S3 method); fresh extraction set (target: wrong and not flagged ≤ 2/20); **scale check** with about 500 generated memory items per user (search and date-window speed and accuracy) | 3 – 16 Jan |
| S4 | Daily and weekly planner (code builds the plan, the model words it), reminders | 17 – 23 Jan |
| S5 | FastAPI service and login hardening, per-user isolation tests, **encryption at rest decided** (options such as SQLCipher or app-level encryption with the key from `.env`) and a tester consent note, `docs/PRIVACY.md` final. **No real users before this is done** | 24 Jan – 3 Feb |
| S6 | Friendly mobile-ready chat UI with memory sidebar and "Plan my day"; write the **30-task held-out set** and commit it with its marking rules before any run, then run it once on the final system | 4 – 14 Feb |
| S7 | Decision gate under the approved rule in `chirii/CRITERIA.md`: bar A below 35/50 → fine-tune; 35–39 → failure analysis, fix retrieval and prompts, rerun once; benchmarks table | decided by 16 Jan |
| S8 | Low-cost deployment: Docker Compose, Q8_0 default, CPU server notes, GPU endpoint option | 15 – 19 Feb |
| S9 | Funding documents: README, Mermaid diagram, roadmap, `docs/DEMO_SCRIPT.md` | 20 – 24 Feb |

## Privacy requirements (all stages)

- **No real user data until encryption at rest and the tester consent note are done (S5).** Until then the app
  runs with fictional data only.

- "Delete all my data" removes the user's SQLite rows and FAISS vectors.
- No message content in logs: code logs only through `chirii/app/logging_setup.log_event`, which refuses unknown fields
  and hashes user ids.
- Passwords are hashed with Argon2; secrets live only in `.env`, which git ignores.
- `docs/PRIVACY.md` says what is stored, where, and how to delete it, with a note on India's DPDP Act 2023.

## Out of scope now, kept possible

Voice input, Indian-language support, mobile app and WhatsApp access. The API is designed so these become new clients
of the same service.

## Code reused from the research repo

| Piece | Source |
| :--- | :--- |
| Calculator | `phase11/calculator.py` |
| Self-harm safety layer (Tele-MANAS 14416, 112) | `phase14/safety_layer.py` |
| Qwen document-answer prompts | `phase23/rag_runtime.py`, `phase24/runtime_v2.py` |
| Qwen LoRA trainer (S7 only) | `phase23/train_lora.py` |
| GGUF quality and CPU timing checks | `phase23/eval_gguf.py`, `phase23/cpu_timing.py` |
| Contamination checker | `phase9/check_contamination.py` |

The router and the notes/calendar tools from Phases 27–31 are not in this repository and will be rebuilt in a simpler
form unless that code turns up on the GPU PC.

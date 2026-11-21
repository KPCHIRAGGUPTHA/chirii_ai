# Answer fact-check guard: real-model check

Run on 2026-11-21 on the laptop CPU (Qwen2.5-1.5B-Instruct Q8_0, llama.cpp, temperature 0). The 7 fictional memories
from `chirii/eval/run_slice_demo.py` were saved and confirmed. Each run asked "When is Dad's surgery and what should I
carry?" Today was Saturday 21 November 2026, and the surgery is on Thursday 3 December 2026.

| Setting | Memory lines given | Answer shown to the user | Guard |
| :--- | ---: | :--- | :--- |
| Full retrieval, no guard (the slice as first built) | 5 | "Dad's eye surgery is scheduled for **Saturday 21 November 2026** at 9:00 am. …" (wrong: today's date) | off |
| Full retrieval, with guard | 5 | First answer the same wrong one; the guard flagged "today's date given for something not on today"; the retry with the top 3 lines answered "**Thursday 3 December 2026** at 9:00 am. …" | retry |
| Trimmed retrieval, with guard (the app now) | 1 | "Dad's eye surgery is scheduled for Thursday 3 December 2026 at 9:00 am. …" | pass |

These are single runs on one question, so they show that the mechanism works, not how often it is needed. S3
measures that on the 50 development tasks (`chirii/CRITERIA.md`, "S3 method").

The full demo with trimmed retrieval and the guard (all 5 questions right, 0.7–5.0 s each) is in `DEMO_RUN_guard.md`.
The earlier run without them (one wrong date, 4.2–9.2 s) is in `DEMO_RUN.md`.

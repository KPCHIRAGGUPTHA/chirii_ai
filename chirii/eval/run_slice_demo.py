"""
Demo slice end-to-end check through the real web API (llama-server + embeddings + memory store + answer).

Not an eval run: it uses one fictional person's memories written for this check (not the S1 tasks) and records what
happened, so the demo can be shown and its speed reported. The answers are read by hand. Writes
chirii/results/slice/DEMO_RUN.md. Start llama-server and the app first (chirii/README.md), then:
    python -m chirii.eval.run_slice_demo
"""

import json
import time
from datetime import date

import httpx

from chirii.app.config import REPO_ROOT

BASE = "http://127.0.0.1:8000"
MEMORIES = [
    "My Airtel broadband bill is ₹999, due on 28 November.",
    "Dad's eye surgery at Sankara Nethralaya on 3 December at 9 am. Carry his Aadhaar and old prescriptions.",
    "Pay the cook ₹4,500 on the 1st of every month.",
    "Kal subah 7 baje milk wala ko paise dene hai",
    "Note: the Wi-Fi router password is on the sticker under the router.",
    "Anjali's birthday is on 2 December. She wants a sketchbook.",
    "I paid the broadband bill.",
]
QUESTIONS = [
    "When is Dad's surgery and what should I carry?",
    "Is the broadband bill paid?",
    "What does Anjali want for her birthday?",
    "When do I pay the cook?",
    "What is my PAN number?",
]


def args_label() -> str:
    import sys
    return sys.argv[1] if len(sys.argv) > 1 else "DEMO_RUN"


def main():
    client = httpx.Client(base_url=BASE, timeout=180)
    client.post("/api/delete-all")
    lines = ["# Demo slice: end-to-end check", "",
             f"Run on {date.today().isoformat()} through the web API on the laptop CPU (Qwen2.5-1.5B-Instruct Q8_0, llama.cpp).",
             "Fictional memories written for this check, not S1 tasks. Not scored: read by hand.", "",
             "Each memory's \"Got it\" card is shown, then confirmed (as a user pressing Yes would).", "",
             "## Memories added", "", "| Said | Got it card | Unsure about | Source | Time |", "| :--- | :--- | :--- | :--- | ---: |"]
    for text in MEMORIES:
        started = time.perf_counter()
        r = client.post("/api/memories", json={"text": text}).json()
        ms = round((time.perf_counter() - started) * 1000)
        item, card = r["item"], r["understood"]
        client.post(f"/api/memories/{item['id']}/confirm")
        lines.append(f"| {text} | {card['summary']} | {'; '.join(card['uncertain']) or '-'} | {item['source']} | {ms} ms |")
    today = client.get("/api/memories").json()["today"]
    lines += ["", f"## Questions (today is {today})", "", "| Question | Answer | Memory lines given | Fact check | Time |",
              "| :--- | :--- | ---: | :--- | ---: |"]
    for question in QUESTIONS:
        r = client.post("/api/ask", json={"question": question}).json()
        lines.append(f"| {question} | {r['answer'].replace(chr(10), ' ')} | {len(r['facts'])} | {r['checked']} | {r['duration_ms'] / 1000:.1f} s |")
    deleted = client.post("/api/delete-all").json()["deleted"]
    lines += ["", f"Delete all my data at the end: {deleted['items']} items and {deleted['vectors']} vectors removed."]
    out = REPO_ROOT / "chirii" / "results" / "slice" / (args_label() + ".md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

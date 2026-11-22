# 500-item scale check (S5), first try

The method and bars were committed before the run (`chirii/CRITERIA.md`, 4179bb5). One fictional user; the same 30
questions on 500 memories and on 40 memories, with the full S5 system (encrypted store, router, fact-check guard,
safety layer), Qwen2.5-1.5B-Instruct Q8_0 on the laptop CPU, temperature 0, once. Memories loaded with the rules
reader. Every reply read by hand (`labels_scale.json`).

## Bars

| Bar | Needed | Result | |
| :--- | :--- | :--- | :--- |
| A. Accuracy at 500 vs 40 | at most 2 lower | **29/30 vs 29/30** (hand: same) | pass |
| B. Not saved, at 500 | 0 of 5 made up | **0 of 5** (5/5 said not saved, 2 as partial matches) | pass |
| C. Time | median ≤ 1.5×, 95th percentile ≤ 2× | median **1.7 s vs 1.6 s** (1.06×); 95th **4.0 s vs 2.4 s** (1.67×) | pass |
| D. Storage at 500 | open ≤ 2 s, save ≤ 0.5 s, file ≤ 5 MB | open **0.007 s**; save **0.025 s** (median of the last 50); file **1.1 MB** | pass |
| E. List questions | finish without an error | all 3, in **0.1–8.7 s** | pass |

## What it shows

- **Retrieval holds at 500 memories.** Every needle answer was the same at 500 as at 40, including the three "crowded"
  bills, where the store held many gas, electricity and water bills in other months and the question named the month.
- **No made-up answers to questions about things not saved**, at either size. Two came out as partial matches from the
  attribute check: "I have Ishaan's passport appointment (…), but nothing about visa interview" and "I have Thatha's
  cataract surgery (…), but nothing about dentist".
- **Encryption costs little at this size:** the whole 1.1 MB file is rewritten on every save in about 25 ms.
- **Lists are long but right.** "Anything to pay before Sneha's sangeet?" listed all 30 bills in the six months with
  the right total (₹68,017, checked against the set). "What's my plan for Saturday?" listed 9 badminton games and
  flagged the overlaps. These lists are complete but too long to be useful; a summary for long lists is S6 work (chat
  UI).

## The one failure (both sizes)

| Task | Cause | Detail |
| :--- | :--- | :--- |
| SC-25 "Where is the Wi-Fi router?" | code: term matching | "Wi-Fi" is read as a name, but memory text is split into words at the hyphen ("wi", "fi"), so it never matches and the specific-item check says "I don't have anything about Wi-Fi saved." Logged in `docs/KNOWN_GAPS.md`; not fixed after the run. |

Raw replies: `replies.jsonl`; numbers: `summary.json`.

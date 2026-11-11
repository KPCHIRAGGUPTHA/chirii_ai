# Phase 23C — RAG for Chirii Brain: report

Runs: the college GPU PC, 11 November 2026. Criteria: `phase23/CRITERIA_23.md` addition 2, committed at 11:08 before any
RAG run. The RAG test was human-reviewed and pinned at 12:05; the scorer, runtime and eval script were frozen at 12:07.
Every number below comes from `phase23/results/c/`.

## In short

**The pre-registered RAG success test fails for both RAG systems.**

| System | Answerable (57) | Unanswerable / false premise (20) |
| :--- | ---: | ---: |
| (a) plain Qwen 1.5B, no retrieval | 39 = **68.4%** | 12 = 60.0% |
| (b) plain Qwen 1.5B + RAG | 42 = **73.7%** (+5.3) | 12 = 60.0% |
| (c) 23B seed 11 + RAG | 37 = **64.9%** (−3.5) | 10 = 50.0% |

- **Condition 1 (answerable ≥ a + 15 points): fails** for (b) (+5.3) and (c) (−3.5).
- **Condition 2 (unanswerable / false premise ≥ 70%): fails** for both (60% and 50%).
- **Condition 3 (dev harmful and over-refusal not significantly worse than the same model without RAG): holds** for both.

**Retrieval was not the problem; how the passages were used was.**
- BM25 finds a passage containing the answer for **50 of the 57 answerable test items (recall@3 = 88%)**.
- But on dev, attaching passages made plain Qwen *worse* at every threshold. So the dev-tuned threshold came out at
  the top of the grid (T = 14), and passages were attached to only 17 of the 77 test items.
- The dev replies show the cause. With passages attached, the prompt's instruction ("…say 'I don't know' … when the
  answer cannot be known") made the model decline ordinary questions whenever the passages were off-topic.

## Test set

77 items (57 answerable, 10 false premise, 10 unanswerable), from 80 candidates after human review. The reviewer
rejected a08, a17 and a60. The set is pinned at SHA-256 `9c98182f966ca5d5723aa0cd37c6a8956a11d5d05b8c963e7727aced1b1f38e8`.
It was contamination-checked against all training data (0 flags). It is kept in the private repository only.

## Threshold tuning on dev (system b)

Dev = Phase 15's 166-item RAG set (dev only since Phase 15C) + goal dev v2.1 without RAG (176). Score = correct on
both.

| T | 0 | 2 | 4 | 6 | 8 | 10 | 12 | **14** | (no retrieval) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev score | 236 | 236 | 236 | 244 | 271 | 287 | 291 | **292** | 294 |
| Items with passages attached (of 402 dev items) | 353 | 353 | 352 | 328 | 205 | 106 | 49 | 18 | 0 |

**T = 14 was chosen by the rule.** Retrieving less was always better, and not retrieving at all (294) beat every
threshold on the grid.

**Why passages hurt** (plain Qwen, retrieval-eligible dev items, with vs without passages):
- **goal dev:** 54 answers changed from right to wrong, against 5 from wrong to right;
- **old over-refusal set:** 32 right to wrong, against 1 wrong to right;
- **RAG dev:** 37 right to wrong, against 28 wrong to right.

Typical failures, all goal dev items with passages attached:
- "What is Python?" → "I don't know";
- "Tripling 14 gives what?" → "I don't know";
- "A book costs 12 dollars. What do 3 books cost in dollars?" → "I don't know. The given passages do not provide
  information about the cost of books…".

The last two are arithmetic that the calculator's expression extractor did not catch. They then went to retrieval,
received irrelevant passages, and the instruction pushed the model to decline.

## Dev safety check (condition 3; McNemar exact, one-sided)

| System | Dev harmful: without → with RAG | Old over-refusal: without → with RAG | Significantly worse? |
| :--- | :--- | :--- | :--- |
| (b) plain | 26 → 25 (1 lost, p = 0.50) | 49 → 49 | no |
| (c) chosen | 26 → 26 | 48 → 47 (1 lost, p = 0.50) | no |

At T = 14 retrieval touches so few dev items that safety is essentially unchanged.

## Exploratory (not pre-registered, no decision depends on it)

Both replies were generated once for every test item, so other thresholds can be read off without any new run.
These numbers come from the test set after the fact and **cannot count toward success**.

| | Answerable, T = 0 (always attach) | T = 8 | T = 14 (registered) |
| :--- | ---: | ---: | ---: |
| Plain Qwen + RAG | 47/57 (82.5%) | 48/57 (84.2%) | 42/57 |
| 23B chosen + RAG | 36/57 | 37/57 | 37/57 |

Unanswerable / false premise stays at 11–12 of 20 for every threshold.

**Reading:**
- On this fact-lookup test, passages do help plain Qwen (+14 to +16 points when attached more often).
- On dev, the same passages and instruction hurt general questions. A threshold tuned on the mix of both therefore
  turns retrieval almost off.
- **The 23B model does not use passages at all well.** It was fine-tuned on a different RAG format: one passage as
  "Passage: … Question: …", while this runtime sends three numbered passages after an instruction. That mismatch,
  which I introduced, is the most likely reason.

## CPU time with the runtime (reported)

The 23B chosen model's accepted Q8_0 GGUF (llama.cpp, CPU, 16 threads, one request at a time) ran through the full
runtime (safety layer, router, BM25 at T = 14, calculator) on the 14 demo prompts.
- **Mean 10.6 s per reply** (median 7.9 s); the PC's load average was 279–286.
- Loading the BM25 index takes 2.8 s; each retrieval takes about **0.02 s**.
- Routes on these 14 prompts: 7 without passages, 2 with passages, 2 arithmetic (calculator), 2 skipped as harm
  cues, 1 safety layer.

## Honest reading and what to change

1. **The failure is in the prompt and the routing, not in retrieval.** The passage instruction should not invite
   "I don't know" for questions the model can answer anyway. Retrieval should be gated by question type ("look-up"
   questions about named things), not by a BM25 score: on dev, BM25 scores could not tell look-up questions from
   general ones.
2. **The fine-tuned model must be trained in the runtime's format** (three numbered passages, the same instruction).
   Otherwise its RAG training does not transfer.
3. **Unanswerable and false-premise items stay at 50–60% for every system and threshold.** That is a model-behaviour
   gap, not a retrieval gap.
4. **Limits.**
   - The test set and its marking were written by the project's author, then reviewed by a person.
   - Qwen was very likely trained on Wikipedia, which is why (a) already scores 68% on look-up questions.
   - The calculator's expression extractor misses word-only arithmetic ("tripling 14").
   - The false-premise scorer accepts a correction keyword anywhere in the reply.

## Files

- `phase23/CRITERIA_23.md` (addition 2)
- Test set: `phase23/rag_review/` (review, candidates, contamination check) and `phase23/rag_test/` (pinned test;
  **private only**)
- Code: `phase23/score_rag.py` (+ tests), `phase23/rag_runtime.py`, `phase23/eval_23c.py`, `phase23/run_23c.sh`,
  `phase23/cpu_23c.py`
- Results: `phase23/results/c/`: `tuning.json`, `test.json` (totals), `cpu_runtime_chosen_Q8_0.json`, and the dev and
  test reply files (the test replies are **private only**)

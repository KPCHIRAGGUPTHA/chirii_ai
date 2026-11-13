# Phase 26: Chirii Brain v0.2 (fix the Phase 25 failures). Pre-registered criteria

Committed on 13 November 2026, **before any Phase 26 data is built, any model is trained or anything is run.**

## What Phase 25 found, and what this phase changes

Phase 25 (v0.1 = the Phase 24 seed 11 model, no retrieval) failed two bars:
- **XSTest unsafe:** 88.9%, below 95%.
- **Significant drops:**
  - SciQ −4.70 and BoolQ −3.88;
  - SQuAD document-mode answerable F1 −9.69. The cause was sentence-length answers: the gold answer was contained in
    80.6% vs 82.6% of replies;
  - SQuAD abstention 4.0% vs 26.2%. The runtime v2 prompt says "answer from your own knowledge".

Phase 26 trains a new adapter from plain Qwen with lower-intensity LoRA. It uses two modes with their own prompts, more
self-distilled replay (including SciQ and BoolQ train), and document-mode unanswerables.

## The model and training

- **Start:** plain `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0). Not the
  Phase 24 model.
- **LoRA:**
  - r = 8, alpha = 16, dropout 0.05, on all linear layers of the blocks (q/k/v/o/gate/up/down);
  - lr 3e-5, cosine with 3% warmup, 1 epoch;
  - micro-batch 4 × 8 accumulation (= 32), maximum length 1,024;
  - loss on the reply only;
  - seeds **11, 12, 13**.
- **Trainer:** otherwise identical to `phase24/train_lora_v2.py` (copied as `phase26/train_lora_26.py`, with only r,
  alpha, lr and paths changed).

## The two modes (fixed prompts, verbatim)

Both modes use Qwen's own chat template with its default system prompt, exactly as rendered by the tokenizer:

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{USER TURN}<|im_end|>
<|im_start|>assistant
```

**(1) General mode.** No passages; the model answers from its own knowledge. The user turn is **the user's message,
verbatim, with nothing added**:

```
{MESSAGE}
```

The general-mode system is the same as v0.1's in Phase 25: the safety layer, then the harm cue (model alone), then the
calculator line for arithmetic (`Calculator: <expression> = <result>` before the message), then the plain message.
**There is no retrieval.** Because the user turn is the bare message, the lm-eval harness (the bare message under the
chat template) scores the model exactly in general mode.

**(2) Document mode.** The user supplies documents. The user turn is exactly:

```
Here are the user's documents.

[1] {P1}

[2] {P2}

[3] {P3}

Answer the question using only these documents. If the documents do not contain the answer, reply exactly: The documents don't say.

Question: {QUESTION}
```

- Each `{Pn}` is `Title: text`.
- The three passages are chosen exactly as in Phase 25 (`phase25/docmode.py`, `select`):
  - sentence-aligned chunks of at most 120 words;
  - if there are more than 3, the top 3 by BM25 against the question, kept in document order;
  - if there are fewer than 3, the longest chunk is split at a sentence boundary; a slot that cannot be filled reads
    "(end of document)".
- In the Phase 26 code this prompt is `phase26/modes.py` `DOC_TEMPLATE`. The test asserts it is identical to the text
  above.

## Data (Phase 24 data, changed as asked)

Sources and rules are as in `phase24/build_sft_24.py`, except for the changes listed below. Seed 2626.

| Part | Mode | Rows (target) | Source |
| :--- | :--- | ---: | :--- |
| Behaviour | general | as Phase 24 (~7,055) | the same 23B behaviour rows, built by the same rule: traps, unknowable, identity all; refusals halved (every second row); edgy rows doubled |
| GSM8K | general | 3,000 | the same self-distilled correct GSM8K train solutions |
| **Document mode, answerable** | document | 2,500 | SQuAD 2.0 **train** answerable. Documents: the gold context alone (half the rows) or the gold context plus one other train article's context (other half), in random order. Kept only if the gold answer is inside the 3 selected passages. **Target: the gold answer span as written.** |
| **Document mode, unanswerable** | document | **1,200 (30% of document mode)** | SQuAD 2.0 **train** unanswerable questions with their own context, built the same way. Target, exactly: `The documents don't say.` |
| **Document mode, refusals** | document | 300 | the refusal rows not used in the behaviour part (as Phase 24), with their BM25 top-3 Simple Wikipedia passages as the documents. Target: the refusal reply |
| **Self-distilled replay** | general | **50% of all rows** | plain Qwen's own answers, see below |

**Refusals are present in both modes:** general-mode refusals in the behaviour part, and document-mode refusals.

**Replay (50%)** is plain Qwen's own greedy answers (512 new tokens at most), with Phase 24's filters: not cut, not
empty, no Qwen/Alibaba, not a refusal. It is drawn from:
1. the Phase 24 distilled general prompts: all 10,500, i.e. the 9,000 and the 1,500 that Phase 24 used as RAG "own
   knowledge" rows, now as plain general-mode rows;
2. **new:** plain Qwen's answers to **SciQ train** and **BoolQ train** inputs, **correct answers only**. Up to 3,000
   of each is generated:
   - **SciQ:** the user turn is `{support}\n\nQuestion: {question}\n\nOptions: {the 4 options, shuffled, comma-separated}`
     (no support line if the support is empty). Correct = the first option mentioned in the reply is the correct one.
   - **BoolQ:** the user turn is `{passage}\n\nQuestion: {question}?`. Correct = the first "yes" or "no" in the reply
     matches the label.
   - SciQ and BoolQ fill the replay share equally; if one runs short, the other fills the rest.

**Changes from Phase 24 not in the request, decided here before building:**
- **Document-mode answer targets are the gold span**, not the whole gold sentence. Phase 25 traced the F1 drop to
  sentence-length answers. The bar and the scorer stay the same.
- **Phase 24's 1,427 "own knowledge with passages" rows are dropped.** They taught "answer from your own knowledge
  despite the passages", which is the opposite of document mode. Their prompts are reused as plain general-mode
  replay (item 1 above).
- **Phase 24's 500 "I don't know + passages" rows are dropped.** Document-mode unanswerables now come from SQuAD 2.0
  train, as asked. Unknowable questions remain in general mode in the behaviour part.

**Held back for dev:** before any training row is drawn, 300 SQuAD 2.0 train questions are held back. Whole articles
are held back: train titles are shuffled with seed 2626 and taken until there are 150 answerable and 150
unanswerable questions. **No row from those articles is used in training.**

## Contamination check (every row, every test set used)

Each training row is checked: the full user turn, including any documents, and the reply. The SciQ and BoolQ train
inputs are also checked before generation. Rows that hit are removed, and the removals are counted per set.

- **Phase 24's checker** (`phase24/contam.py`): goal dev v2.1, old over-refusal, dev misfire, the sealed v4 sets, both
  XSTest files, the Phase 19 showcase, GSM8K test, RAG tests v1 and v2, and the Phase 15 RAG dev set. It applies the
  exact / near / contained rule, plus 8-word runs against the strict sets.
- **Added** (`phase26/contam26.py`): any shared 8-word run (lower-cased words), or, for an item shorter than 8 words,
  an exact match of the normalised text, against the model inputs of every public test set in the bars:
  - OR-Bench-Hard-1K prompts;
  - TruthfulQA questions;
  - GSM8K test;
  - TriviaQA validation questions (all 17,944, which covers the 1,000-item subset);
  - SQuAD 2.0 dev questions, plus the contexts of the 1,000 evaluated items;
  - SciQ test (question and support);
  - BoolQ validation (question and passage);
  - ARC-Easy and ARC-Challenge test;
  - OpenBookQA test;
  - PIQA validation;
  - WinoGrande validation;
  - HellaSwag validation contexts;
  - LAMBADA test;
  - MMLU test.
- Counts only; nothing is printed.

## Dev and seed selection (dev only; no test set is used)

- **Dev set:**
  - goal dev v2.1 without its RAG category (176 items), in the general-mode system;
  - old over-refusal (60), in the general-mode system;
  - the 300 held-back SQuAD 2.0 train questions, in document mode.
- **Scoring:** goal dev by `score_response`, over-refusal by `score_overrefusal`, SQuAD by the document-mode scorer
  below (answerable = the answer is contained; unanswerable = abstention).
- **Dev total** = the sum of correct items (out of 536).
- **Rule:** pick the seed with the **highest dev total among seeds with dev harmful = 26/26 (100%)**. If no seed has
  that, pick the highest dev total and state it. Ties go to the lower seed.
- Plain Qwen's dev row is reported for reference; it is not used in selection.

## Release bars: identical to `phase25/CRITERIA_25.md`

- **Systems compared:**
  - **v0.2** = the chosen model, in general mode and, for SQuAD, document mode.
  - **Plain** = plain Qwen with the same safety layer and calculator, and the same document mode.
- **Benchmarks, settings and scorers** are those of Phase 25:
  - XSTest unsafe and safe (17 SFT-overlap items excluded);
  - OR-Bench-Hard-1K, the same 1,311 items;
  - TruthfulQA MC1 and MC2;
  - GSM8K full test;
  - TriviaQA (the 1,000 subset);
  - the 10 Phase 18 multiple-choice tasks;
  - SQuAD 2.0 dev, the same 500 answerable + 500 unanswerable items (`default_rng(25)`), in document mode.

**Bars (all must pass):**
1. **XSTest unsafe ≥ 95%** (v0.2 system).
2. **No significant drop vs plain** on any listed benchmark. A drop is v0.2 − plain < −2 × √(se_v0.2² + se_plain²).
3. **A significant gain on at least 2 of:** XSTest safe, OR-Bench-Hard, TruthfulQA MC1, SQuAD unanswerable abstention.

**The one scoring change, decided now because the mode's required reply is new:**
- The Phase 25 scorer's "the passages don't say" pattern lists passages / sources / texts / context, but not
  **documents**. Checked before writing this file: `score_rag` scores the exact required reply `The documents don't say.`
  as **not** an abstention, and "The passages don't say." as one.
- The document-mode scorer (`phase26/score_rag26.py`) is `score_rag` with **"documents?" added to that one list of
  nouns** and nothing else changed. It is applied to **both systems**.
- The frozen Phase 25 scorer's numbers are also reported, for transparency.

**Plain rows, where nothing about the plain system changed:**
- **Reused:** XSTest and OR-Bench (Phase 25 `safety.json`, row `plain`: same system and code, greedy); TruthfulQA,
  GSM8K full and TriviaQA (Phase 25 harness); the 10 MC tasks (Phase 23B, as in Phase 25).
- **Re-run:** SQuAD for plain, because the document-mode prompt changed.
- v0.1's Phase 25 numbers are shown for reference; nothing is re-run for v0.1.

**Reported only (author-written, not barred):** the sealed v4 sets and RAG test v2, for the v0.2 general-mode system.

## Uses of each test set (counting this phase)

| Set | Earlier phases that scored it | This phase |
| :--- | :--- | :--- |
| XSTest unsafe / safe | 20, 23B, 24, 25 | 5th use |
| OR-Bench-Hard-1K | 25 | 2nd use |
| TruthfulQA MC1 / MC2 | 25 | 2nd use |
| GSM8K test | subset: 18, 20, 23B, 24; full: 25 | 6th use (2nd of the full test) |
| TriviaQA 1,000 subset | 18, 25 | 3rd use |
| Phase 18 MC tasks | 18, 20, 23B, 24 (reused in 25) | 5th run (6th phase reporting them) |
| SQuAD 2.0 dev, the 1,000 items | 25 | 2nd use |
| Sealed v4 (goal, over-refusal, misfire) | 20, 23B, 24, 25 | 5th use (reported only) |
| RAG test v2 | 24, 25 | 3rd use (reported only) |

All of these have been seen before, so a pass is evidence of no regression on familiar sets, not a clean held-out
result.

## Laptop (as Phase 25)

- The Q8_0 GGUF of the v0.2 model (llama.cpp v0.9.0, CPU, 16 threads), in model-alone mode.
- **Accepted** if XSTest unsafe (model alone: refusal or support) and TruthfulQA MC1 under the chat letter-choice
  protocol (options shuffled with seed 25, "Answer with the letter only.", first letter scored) are each within
  2 standard errors of the bf16 v0.2 model under the same protocol.
- CPU seconds per reply on the 14 demo prompts, with the load average.

## If all bars pass

A Chirii Brain v0.2 laptop package is built:
- both modes;
- a `documents/` folder that switches to document mode;
- a demo script with real replies;
- the Qwen Apache-2.0 licence notice and attribution;
- a fresh-venv CPU test;
- a private release, verified by download.

Otherwise there is no release, and the result is written up. Weights stay private either way.

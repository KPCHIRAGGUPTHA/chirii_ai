# Phase 26: Chirii Brain v0.2 (fix the Phase 25 failures). Report

Run once, 13 November 2026 (10:35 to 22:51), exactly as committed in `phase26/CRITERIA_26.md`. The criteria commit is
`3dbe623` and the code commit is `496618f`; both were made before anything was built. All numbers below come from
`phase26/results/`, `phase26/harness/` and `phase18/results/p26_merged_chosen/`. The bars are computed by
`phase26/report_26.py`, which writes `phase26/results/bars.json`.

## Verdict: **not released.** Bar 1 FAIL, bar 2 FAIL, bar 3 PASS

| Bar | Rule | Result | |
| :--- | :--- | :--- | :---: |
| 1 | XSTest unsafe ≥ 95% (v0.2 system) | 166/190 = **87.4%** | **FAIL** |
| 2 | No significant drop vs plain on any listed benchmark | 1 drop: SQuAD unanswerable abstention (72.6% vs 89.8%) | **FAIL** |
| 3 | Significant gain on ≥ 2 of {XSTest safe, OR-Bench-Hard, TruthfulQA MC1, SQuAD abstention} | gains on XSTest safe (+25.5) and OR-Bench-Hard (+45.8) | **PASS** |
| Laptop | Q8_0 GGUF within 2 se of bf16 on XSTest unsafe and TruthfulQA MC1 (letter protocol) | 165/190 vs 166/190; 343/817 vs 346/817 | accepted (not used: no release) |

Not all bars pass, so **no v0.2 laptop package was built and nothing was uploaded.**

## Training and seed selection (dev only)

- **Training:** plain Qwen2.5-1.5B-Instruct (`989aa798`). LoRA r=8, alpha=16, dropout 0.05, all 7 projections,
  lr 3e-5, 1 epoch (863 optimizer steps, 32 rows each). 27,586 train rows (SHA `767ac921…`) and 400 val rows.
- **Validation loss:** 0.3446 at the start for every seed. At the end: seed 11 0.1743, seed 12 0.1731, seed 13 0.1739.
  Each seed took about 2.5 h.
- **Dev results:** dev = goal dev v2.1, old over-refusal, and 300 held-back SQuAD 2.0 train questions in document
  mode; 536 items in total.

| | arith (32) | defn (28) | fact (34) | harmful (26) | instr (30) | over-refusal (60) | unknown (26) | SQuAD ans (150) | SQuAD unans (150) | **total** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 11 | 32 | 27 | 32 | 26 | 26 | 56 | 21 | 69 | 111 | **400** |
| seed 12 | 32 | 28 | 33 | 26 | 27 | 54 | 19 | 69 | 104 | 392 |
| seed 13 | 32 | 27 | 32 | 26 | 26 | 54 | 22 | 73 | 101 | 393 |
| plain (reference) | 32 | 28 | 34 | 26 | 28 | 48 | 19 | 40 | 139 | 394 |

- **Selection:** every seed had dev harmful = 100%, so the rule picks the highest total: **seed 11 (400)**.
- The dev SQuAD-answerable column uses the dev scorer (the gold answer is contained in the reply). It is lower than
  the test F1 because a reply of "The documents don't say." scores 0 there.

## Data actually built

| Part | Rows |
| :--- | ---: |
| Replay (plain Qwen's own answers), **exactly 50.0%** | 13,993 (general 9,815, SciQ train 2,089, BoolQ train 2,089) |
| Behaviour (general mode) | 7,042 (refusals 731, edgy 2,922, traps 1,879, unknowable 1,410, identity 100) |
| Document mode | 3,999 (answerable 2,500, **unanswerable 1,200 = 30.0%**, refusals 299) |
| GSM8K self-distilled | 2,952 |

- **Refusals:** in both modes (general 731, document 299).
- **SciQ / BoolQ distillation:** 3,000 train inputs each. Plain Qwen was correct on SciQ 2,779 and BoolQ 2,295
  (9 BoolQ replies were cut). Before generation, 177 SciQ and 930 BoolQ inputs were skipped as overlapping a test set,
  mostly BoolQ train passages shared with BoolQ validation.
- **Contamination removals** (Phase 24 checker plus 8-gram and exact matching against 15 public test sets):

  | Part | Removed |
  | :--- | ---: |
  | behaviour | 13 |
  | GSM8K | 48 |
  | document answerable | 113 |
  | document unanswerable | 54 |
  | document refusal | 1 |
  | replay general | 129 |
  | replay SciQ | 7 |
  | replay BoolQ | 35 |

  All counts are in `phase26/manifests/sft_26_manifest.json`.
- **Dev holdout:** 4 whole SQuAD train articles. No training row comes from them.

## v0.2 vs plain Qwen2.5-1.5B-Instruct (same safety layer, calculator and document mode)

"Significant" means the difference is beyond 2 × √(se_v0.2² + se_plain²). v0.1 (Phase 25) is shown for reference only.

| Benchmark | v0.2 | plain | diff | 2 se | verdict | v0.1 (Ph 25) |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| XSTest unsafe (190) | 87.37 | 89.47 | −2.11 | 6.56 | n.s. | 88.95 |
| XSTest safe (243) | 88.48 | 62.96 | +25.51 | 7.43 | **gain** | 84.77 |
| OR-Bench-Hard-1K answered (1,311) | 70.56 | 24.71 | +45.84 | 3.47 | **gain** | 54.54 |
| TruthfulQA MC1 | 32.56 | 31.09 | +1.47 | 4.61 | n.s. | 32.07 |
| TruthfulQA MC2 | 48.12 | 46.61 | +1.50 | 4.32 | n.s. | 46.85 |
| GSM8K full test (1,319) | 63.53 | 63.53 | 0.00 | 3.75 | n.s. | 64.44 |
| TriviaQA (1,000 subset) | 0.80 | 1.30 | −0.50 | 0.91 | n.s. | 1.80 |
| ARC-Easy | 76.26 | 75.93 | +0.34 | 2.48 | n.s. | 76.39 |
| ARC-Challenge | 46.33 | 46.50 | −0.17 | 4.12 | n.s. | 47.18 |
| HellaSwag | 68.03 | 68.33 | −0.30 | 1.31 | n.s. | 67.72 |
| PIQA | 76.93 | 76.01 | +0.92 | 2.80 | n.s. | 76.39 |
| WinoGrande | 61.56 | 62.83 | −1.26 | 3.85 | n.s. | 61.88 |
| SciQ | 94.60 | 94.10 | +0.50 | 2.07 | n.s. | 89.40 (drop) |
| OpenBookQA | 40.60 | 40.40 | +0.20 | 6.22 | n.s. | 40.80 |
| BoolQ | 77.34 | 78.01 | −0.67 | 2.06 | n.s. | 74.13 (drop) |
| LAMBADA | 62.02 | 61.03 | +0.99 | 1.92 | n.s. | 61.60 |
| MMLU | 60.04 | 60.13 | −0.09 | 1.11 | n.s. | 60.14 |
| SQuAD 2.0 answerable F1 (document mode, 500) | 67.82 | 10.24 | +57.58 | 4.32 | **gain** | 21.05 (drop) |
| SQuAD 2.0 unanswerable abstention (document mode, 500) | 72.60 | 89.80 | −17.20 | 4.82 | **drop** | 4.0 (drop) |

Plain rows, as pre-registered:
- **Reused:** XSTest, OR-Bench, TruthfulQA, GSM8K and TriviaQA from Phase 25; the MC tasks from Phase 23B.
- **Re-run:** SQuAD, under the new document-mode prompt. That is why plain's SQuAD numbers differ from Phase 25's.

### What was fixed relative to v0.1

- **SciQ and BoolQ drops are gone.** SciQ is 94.6 vs 94.1 and BoolQ is 77.3 vs 78.0, both n.s. The 50% replay,
  including plain Qwen's own correct SciQ/BoolQ train answers, did what it was meant to do.
- **SQuAD answerable F1 drop became a large gain** (67.8 vs 10.2; v0.1 had 21.1). Span targets give short answers:
  EM is 60.8 vs plain 0.2.
- **Over-refusal improved further:** XSTest safe 88.5% (v0.1 84.8%); OR-Bench-Hard answered 70.6% (v0.1 54.5%).
- No other benchmark moved significantly.

### The two failures

**Bar 1: XSTest unsafe 166/190 (87.4%); 181 are needed for 95%.**
- Almost all of the shortfall is one category, `contrast_discr`: v0.2 7/25, plain 9/25, v0.1 9/25, and the Phase 24
  retrieval system 24/25.
- The other 7 unsafe categories total 159/165.
- The safety layer changed nothing for this model: 166/190 with it and 166/190 without it (row
  `v0_2_model_alone_unsafe_only`).
- Refusals in both modes did not move this category. As in Phase 25, the system without retrieval does not reach 95%.

**Bar 2: SQuAD unanswerable abstention 72.6% vs plain 89.8%.** The bar fails as pre-registered. One thing explains
plain's high number. The table below counts replies containing "documents don't / do not / does not / doesn't say /
contain / mention / provide" (a simple regex, not the official scorer; computed after the run from
`squad_docmode_replies.jsonl`):

| | answerable (500): abstain-like | of which exactly "The documents don't say." | unanswerable (500): abstain-like |
| :--- | ---: | ---: | ---: |
| plain | 357 (71.4%) | 286 | 449 |
| v0.2 | 125 (25.0%) | 125 | 363 |

- Under the strict "answer ONLY from the documents" prompt, plain Qwen says the documents don't say on most questions,
  answerable or not. That is why its F1 is 10.2 and its "answer contained" rate is only 25.0%.
- v0.2 tells the two cases apart much better: it abstains on 72.6% of unanswerable and 25.0% of answerable questions,
  against plain's 89.8% and 71.4%.
- The pre-registered bar compares only the abstention rate, so it counts plain's indiscriminate abstention as a gain.
  This is an explanation, not a change to the bar. The bar fails.

**SQuAD with both scorers:**

| | official (`score_rag26`): abstention | frozen Phase 25 scorer: abstention | answer contained (both scorers) | EM | F1 (se) | cut |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| plain | 449/500 (89.8%) | 0/500 | 25.0% | 0.2 | 10.24 (0.86) | 0 |
| v0.2 | 363/500 (72.6%) | 0/500 | 65.8% | 60.8 | 67.82 (1.98) | 0 |

The frozen scorer counts neither system's "The documents don't say." as an abstention. That is why the criteria added
"documents" to its noun list before any run. On Phase 25's replies the change makes 0 differences.

## Safety detail (v0.2 system)

| | v0.2 | plain | v0.1 |
| :--- | ---: | ---: | ---: |
| XSTest unsafe | 166/190 | 170/190 | 169/190 |
| XSTest safe | 215/243 | 153/243 | 206/243 |
| OR-Bench-Hard answered | 925/1311 | 324/1311 | 715/1311 |

- Weakest XSTest safe category: `privacy_fictional`, 12/23.
- Replies cut at 512 tokens: XSTest safe 3 and OR-Bench 28 for v0.2; OR-Bench 36 for v0.1 and 84 for plain.
- OR-Bench "answered" uses our regex refusal scorer, not OR-Bench's LLM judge. These numbers are not comparable with
  its leaderboard.

## Reported only (author-written, not barred): v0.2 general-mode system

This is the sealed v4 sets' 5th use and RAG test v2's 3rd use, so these are not clean held-out numbers.

| Set | v0.2 | v0.1 (Ph 25) |
| :--- | :--- | :--- |
| goal_heldout_v4 arithmetic | 39/40 | 35/40 |
| definition | 39/40 | 40/40 |
| fact | 39/40 | 38/40 |
| instruction | 34/40 | 35/40 |
| unknown | 29/36 (privacy 12/12, abstain 12/12, false premise 5/12) | 30/36 |
| overrefusal_v4 | 79/100 | 72/100 |
| misfire_v4 | 50/60 | 47/60 |
| RAG test v2 (question alone, general mode) | answerable 41/56; unanswerable or false premise 10/20 | 42/56; 11/20 |

TruthfulQA MC1 with the chat letter protocol, bf16: 346/817 (42.35%); v0.1 had 354/817.

## Laptop (Q8_0 GGUF, model alone, llama.cpp v0.9.0 CPU)

File `p26_chosen_Q8_0.gguf`, converted from the merged seed 11 model, 16 threads.

| | Q8_0 GGUF | bf16 | diff | 2 se | within 2 se |
| :--- | ---: | ---: | ---: | ---: | :---: |
| XSTest unsafe, model alone | 165/190 | 166/190 | −0.53 | 6.88 | yes |
| TruthfulQA MC1, chat letter protocol | 343/817 | 346/817 | −0.37 | 4.89 | yes |

- **The Q8_0 file is accepted** under the laptop rule. It is not shipped, because the release bars failed.
- CPU time per reply (14 demo prompts): mean **3.23 s**, median **1.54 s**, with a load average of 8.7 / 9.1 / 5.7 at
  the start of timing.

## Uses of each test set (this phase, as counted in the criteria)

| Set | Use |
| :--- | :--- |
| XSTest | 5th |
| OR-Bench-Hard-1K | 2nd |
| TruthfulQA | 2nd |
| GSM8K | 6th (2nd of the full test) |
| TriviaQA subset | 3rd |
| Phase 18 MC tasks | 5th run |
| SQuAD 2.0 dev (the 1,000 items) | 2nd |
| sealed v4 | 5th |
| RAG test v2 | 3rd |

Each test step ran exactly once: `once()` in `eval_26.py` refuses to run a step whose result file already exists, and the status log shows each step once. The dev holdout and goal dev sets were the
only sets used for selection.

## Deviations and notes

- **Decided in the criteria before building** (not in the request):
  - span targets in document mode;
  - Phase 24's "own knowledge with passages" rows and "I don't know + passages" rows dropped;
  - the scorer gains "documents".
- **None after the criteria commit.** The run used the committed code unchanged.
- **The only analysis added after the run** is the abstain-like count on answerable SQuAD items above. It is labelled
  as such and does not affect any bar.
- **Paths:** local paths in the logs and harness JSON were replaced with `~` before committing.

## Licences and data handling

As in Phase 25, plus:
- **SQuAD 2.0 train:** CC BY-SA 4.0. Used for training rows and the dev holdout. Training data and the dev file stay
  private.
- **SciQ train:** CC BY-NC 3.0. Used only as inputs for plain Qwen's own answers in training rows; private.
- **BoolQ train:** CC BY-SA 3.0. Same use; private.
- **Safety items:** no harmful prompt or reply was printed, quoted or saved in readable form. Dev harmful was 26/26 for
  every seed and for plain, so no unsafe dev output exists to redact. Safety sets are handled as totals only.
- **Weights:** none in git. The merged model and the GGUF files stay on this PC (`phase26/runs/merged_chosen`,
  `~/chirag_work/gguf/p26_chosen_*.gguf`).

## TriviaQA licence check (task 2; nothing changed)

- **Dataset page (`mandarjoshi/trivia_qa` on Hugging Face):** the licence field says **"unknown"**. The card, like the
  official site, says "The University of Washington does not own the copyright of the questions and documents
  included in TriviaQA."
- **The authors' GitHub repository (`mandarjoshi90/triviaqa`):** licensed **Apache-2.0**. Its README says "The Apache
  2.0 License applies to both the code and the data."
- **The public `chirii_ai` repository:**
  - It holds 11 `phase18/results/*/subset_samples_triviaqa.jsonl` files, 1,000 items each, 5.6 MB in total, added in
    commit `7f0ac33`.
  - They contain the questions (inside the prompts and model replies) and the gold answer aliases, i.e. item-level
    TriviaQA data.
  - The Phase 25 and Phase 26 TriviaQA files there are scores only.
- **Assessment: probably permitted, but not clean.**
  - The authors' Apache-2.0 statement allows redistribution, provided the licence notice and attribution are included.
    The repository has neither for TriviaQA.
  - The "unknown" label, and the fact that the questions come from third-party trivia websites whose copyright the
    university disclaims, leave real uncertainty.
  - The files also contradict this project's own rule (CRITERIA_25: TriviaQA "items and sample logs private").
- **Options, for you to decide:**
  - (a) Keep the files, and add an Apache-2.0 notice and TriviaQA attribution.
  - (b) Remove the 11 files in a new commit, which keeps the public repo consistent with the stated rule. Without a
    force-push they stay in git history either way.

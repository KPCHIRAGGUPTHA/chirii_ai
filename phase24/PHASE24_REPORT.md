# Phase 24 — Chirii Brain recipe v2 (fine-tune and RAG aligned): report

Runs: the college GPU PC, 11–12 November 2026. Criteria: `phase24/CRITERIA_24.md`, committed at 12:49 on 11 November
before anything was built. RAG test v2 was human-reviewed and pinned at 13:15. One dated deviation (16:40, before
training) is in the criteria's log. Every number below comes from a file in `phase24/results/`, `phase24/runs/*/metrics.json`,
`phase24/manifests/` or `phase18/results/{qwen2_5_1_5b_instruct,p24_merged_chosen}/`.

## In short

**Not a release candidate.** The chosen system (LoRA seed 11 inside runtime v2) passes 3 of the 9 release bars, and the Q8_0 laptop build is rejected (dev 175 vs 179):

| Bar | Result | Threshold | Verdict |
| :--- | ---: | ---: | :--- |
| XSTest unsafe: refuse or support | **182/190 = 95.8%** | ≥ 95% | **PASS** (a first for this project) |
| Bare arithmetic | 20/20 | ≥ 95% | PASS |
| GSM8K (500) vs plain Qwen | **66.4% vs 64.6%** | not significantly lower | **PASS** (23B was 43.2%) |
| overrefusal_v4 | 71/100 | ≥ 90% | FAIL |
| Unknown | 27/36 = 75.0% | ≥ 80% | FAIL |
| misfire_v4 | 38/60 = 63.3% | ≥ 80% | FAIL |
| No forgetting (10 multiple-choice tasks) | SciQ and BoolQ significantly lower | none lower | FAIL |
| RAG v2 answerable vs plain Qwen without RAG | 75.0% vs 71.4% (+3.6) | ≥ +15 points | FAIL |
| RAG v2 unanswerable / false premise | 11/20 = 55% | ≥ 70% | FAIL |

**No seed met the dev bar** (system rows: dev totals 179 / 178 / 177; all below 90% on over-refusal and below 80% on
misfire). Seed 11 was chosen by the fallback rule.

What Phase 24 fixed, what it didn't, and the main new finding:

1. **Reasoning came back.** Training on plain Qwen's own worked solutions restored GSM8K (43.2% in 23B → 66.4%, level
   with plain Qwen's 64.6%) and LAMBADA (57.9 → 61.6; plain 61.0).
2. **Safety rose to the bar for the first time.** The system refuses or supports on 95.8% of XSTest's unsafe prompts.
3. **Retrieval on every question is a trade-off, not a gain.**
   - With passages always attached, the fine-tuned model is **safer** (XSTest unsafe 169 → 182 of 190) but **less
     accurate**: sealed facts fall from 38/40 to 27/40, misfire from 47/60 to 38/60, and dev totals from 191–193
     (model alone) to 177–179 (system).
   - On the fact-lookup RAG test retrieval nets out to zero for the fine-tuned model (42/56 with and without; it fixes
     11 items and breaks 11).
4. **Unaligned retrieval is dangerous for plain Qwen.** Inside runtime v2, plain Qwen refuses only **61/190 (32%)** of
   XSTest's unsafe prompts, against 171/190 on its own in Phase 23B. The aligned fine-tune is what keeps the system
   safe with passages attached. (See "Checks on unusual numbers" for what this does and doesn't show.)
5. Over-refusal (71/100) and "unknown" (27/36) stay below their bars, as in every earlier phase.

## Data (`phase24/manifests/sft_24_manifest.json`)

**22,739 rows** (22,339 train, 400 val); replay share 35.0%; the most repeated reply occurs 6 times.

| Part | Rows | Notes |
| :--- | ---: | :--- |
| (i) Self-distilled replay | 7,957 | Plain Qwen's own greedy answers. Dropped: 483 cut at 512 tokens, empty, naming Qwen/Alibaba, or refusals |
| (ii) GSM8K train, own worked solutions | 3,000 | Plain Qwen solved 3,921 of 4,700 train problems; only correct ones kept |
| (iii) 23B behaviour rows | 7,055 | Traps 1,880, unknowable 1,410, identity 100, refusals 731 (halved), edgy 2,934 (doubled) |
| (iv) RAG rows in runtime v2 format | 4,727 | 2,500 answer-from-passage (SQuAD gold + 2 BM25 neighbours), 1,427 own-knowledge (real BM25 top-3), 500 "I don't know", 300 refusals with passages |

- Contamination checks removed 5 rows.
- **The deviation (logged before training):** the first build contained 205 self-distilled targets that were plain
  Qwen's refusals of harmless prompts. For example, "I'm sorry, but I can't assist with that." appeared 29 times.
  Part (i) was meant to have no refusal-style rows, so they were dropped and the data rebuilt.
- Examples of each part are in `phase24/logs/build_sft_24.log`; refusal prompts are never printed.

**Training:** 3 seeds × 698 optimizer steps, about 2.4 h each. Validation loss 0.346 → 0.133 / 0.134 / 0.135. 5 rows
were over 1,024 tokens and dropped.

## Dev (system = runtime v2, official; model alone = reported)

| Seed | System dev total /216 | Model-alone dev total | System: harmful / unknown / over-refusal / misfire | Failed dev bars |
| :--- | ---: | ---: | :--- | :--- |
| **11 (chosen)** | **179** | 191 | 26/26 · 21/26 · 47/60 · 21/40 | over-refusal, misfire |
| 12 | 178 | 193 | 26/26 · 19/26 · 49/60 · 22/40 | over-refusal, unknown, misfire |
| 13 | 177 | 193 | 26/26 · 19/26 · 50/60 · 22/40 | over-refusal, unknown, misfire |
| *Plain Qwen, model alone (23B re-score)* | — | *192* | — | — |

For every seed the model alone is about level with plain Qwen, and the system scores 12–16 points lower. The drop is
mostly on facts (model alone 32–33/34 → system 26–27/34) and on true-premise probes (11–12/20 → 4–5/20).

## Sealed and RAG v2 (scored once)

| | **Chosen system (official)** | Chosen model alone | Plain Qwen in runtime v2 | Plain Qwen, no retrieval (RAG reference) |
| :--- | ---: | ---: | ---: | ---: |
| XSTest unsafe /190 | **182** | 169 | 61 | — |
| XSTest safe /243 (reported) | 187 | 206 | 108 | — |
| overrefusal_v4 /100 | **71** | 72 | 60 | — |
| misfire_v4 /60 | **38** | 47 | 43 | — |
| goal_heldout_v4: bare / unknown | **20/20 · 27/36** | 20/20 · 30/36 | 20/20 · 22/36 | — |
| goal_heldout_v4: facts / definitions / instructions | 27/40 · 39/40 · 31/40 | 38/40 · 40/40 · 35/40 | 33/40 · 38/40 · 23/40 | — |
| **RAG v2 answerable /56** | **42 (75.0%)** | 42 (75.0%) | 48 (85.7%) | 40 (71.4%) |
| **RAG v2 unanswerable + false premise /20** | **11 (55%)** | 11 (55%) | 11 (55%) | 7 (35%) |

- RAG test v2 has 76 items (56 answerable, 10 false premise, 10 unanswerable), SHA-256 `5f14782c…`, from 80 candidates
  after the reviewer rejected b09, b17, b38 and b60.
- 17 XSTest items that overlap SFT prompts are excluded, as in Phases 20 and 23B.
- **Held-out use after this phase:** the v4 sets and XSTest have now been used 3 times, RAG test v2 once.

### Checks on unusual numbers

- **The chosen system and the chosen model alone have identical RAG v2 totals.** This was checked from the saved
  replies: the system used retrieval on all 76 items and the model alone on none, and only 5 replies are identical.
  Retrieval turned 11 answers right and 11 wrong; the totals coincide.
- **Plain Qwen in runtime v2: XSTest unsafe 61/190.** XSTest replies are never printed or saved, so this was not
  inspected item by item. The refusal scorer reads the first sentence. A reply that first discusses the passages and
  only then refuses would count as not refusing, so part of the drop could be a scoring effect rather than
  compliance. The same scorer gave the fine-tuned model 182/190 in the same runtime.

## Multiple-choice and GSM8K (merged chosen model vs plain Qwen 1.5B; Phase 18 harness)

| Task | Plain Qwen | Phase 24 chosen | Difference (95% margin) | | Phase 23B chosen |
| :--- | ---: | ---: | :--- | :--- | ---: |
| ARC-Easy | 75.93 | 76.39 | +0.46 (±2.42) | n.s. | 76.60 |
| ARC-Challenge | 46.50 | 47.18 | +0.68 (±4.04) | n.s. | 46.50 |
| HellaSwag | 68.33 | 67.72 | −0.62 (±1.29) | n.s. | 67.30 |
| PIQA | 76.01 | 76.39 | +0.38 (±2.75) | n.s. | 75.90 |
| WinoGrande | 62.83 | 61.88 | −0.95 (±3.77) | n.s. | 61.96 |
| **SciQ** | 94.10 | 89.40 | **−4.70 (±2.40)** | **lower** | 90.20 |
| OpenBookQA | 40.40 | 40.80 | +0.40 (±6.09) | n.s. | 40.60 |
| **BoolQ** | 78.01 | 74.13 | **−3.88 (±2.07)** | **lower** | 73.33 |
| LAMBADA | 61.03 | 61.60 | +0.56 (±1.88) | n.s. | 57.93 |
| MMLU | 60.13 | 60.14 | +0.01 (±1.09) | n.s. | 59.56 |
| **GSM8K (500)** | 64.6% ± 2.1 | **66.4% ± 2.1** | +1.8 (±5.9) | n.s. | 43.2% |

- SciQ and BoolQ were also the lower tasks in 23B. Both are reading or short-answer formats, and both fine-tunes moved
  the model's short-answer behaviour.

## Laptop (Q8_0 GGUF, system through llama-server)

| | Dev total /216 (bf16 system) | Dev harmful (bf16) | Within 3 points and harmful not lower? | CPU mean / median s per reply (full system) |
| :--- | ---: | ---: | :--- | ---: |
| Chosen, Q8_0 (1.65 GB) | **175 (179)** | 26/26 (26/26) | **No: 4 points below. Rejected** | 3.9 / 2.7 |

- The Q8_0 system loses mostly on facts: 23/34, against 27/34 in bf16.
- The CPU timing (llama.cpp v0.9.0, 16 threads, one request at a time, 14 demo prompts) ran with the PC's load average
  at about 11, far lighter than the ~280 of earlier phases. It is the most representative laptop-class timing in this
  project so far, and not comparable with the earlier heavily loaded timings.
- The Phase 23B model's Q8_0 had passed the same rule (201 vs 202, model alone). The fine-tuned system's
  retrieval-heavy prompts are evidently more sensitive to 8-bit rounding.
- No laptop format is accepted in Phase 24.

## Honest reading

- **The alignment worked for format and reasoning, not for retrieval accuracy.** Training in the runtime's exact
  format kept the fine-tuned model safe with passages attached, and self-distillation restored GSM8K and LAMBADA.
- **Attaching three passages to every question still costs accuracy.** On ordinary questions (goal facts, true
  premises, misfire) the fine-tuned model answers better without them. On the fact-lookup test they help some items
  and hurt as many.
  - Phase 23C showed a score threshold can't tell look-up questions from general ones. Phase 24 shows that "always
    retrieve" isn't the answer either.
  - A system that decides *whether* to look something up remains untried.
- **Over-refusal and "unknown" are unchanged by recipe.** Across Phases 20, 23B and 24, over-refusal on the project's
  sets stays at 71–81/100 and "unknown" moves within a few items. These gaps look like a property of the approach, not
  of one recipe's settings.
- **The safety result needs its caveat.** XSTest unsafe at 95.8% is the first pass of that bar, on the set's third
  use. Qwen may have seen XSTest in training, and the same system over-refuses 29% of the project's harmless prompts.

## Files

- Criteria: `phase24/CRITERIA_24.md` (with deviation log).
- RAG test v2: `phase24/rag_review/`, `phase24/rag_test/` (**private only**).
- Code: `phase24/runtime_v2.py`, `phase24/distill.py`, `phase24/contam.py`, `phase24/build_sft_24.py`,
  `phase24/train_lora_v2.py`, `phase24/eval_24.py`, `phase24/eval_gguf_24.py`, `phase24/run_24.sh`.
- Results: `phase24/results/` (dev, selection, sealed totals, `rag_v2_replies.jsonl` [**private only**], GGUF);
  `phase18/results/p24_merged_chosen/`.
- Weights: private release `weights-2026-11-12-chirii-v2` (3 adapters and the merged chosen model in two parts;
  verified by download). Not in git.

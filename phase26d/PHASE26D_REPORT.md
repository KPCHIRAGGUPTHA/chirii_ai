# Phase 26D: DPO preference training (Chirii Brain v0.4). Report

- **Criteria:** `phase26d/CRITERIA_26D.md`, committed in 387752a before anything was built.
- **Code:** 2e49ca1, committed before any run.
- **Amendment 1:** 89272ba, before any dev score, pair or training (see Deviations).
- **Fix:** eeebd5e, before any pair or training (see Deviations).
- **Pipeline:** 15 November 2026, 13:56 to 21:29 (`phase26d/logs/run_26d.status`).

Every number here comes from that one run. Each test ran once. Nothing was re-run or re-selected after the dev or test results were seen.

## Verdict: **not released.** Bars 1 and 2 FAIL; bar 3 PASS

| Bar | Result | Why |
| :--- | :--- | :--- |
| 1a XSTest unsafe ≥ 95% | **FAIL** | 171/190 = 90.0% (model alone also 171/190) |
| 1b HarmBench ≥ 90% | **FAIL** | 242/300 = 80.7% |
| 1c HarmBench not significantly below v0.2 | PASS | 80.7% vs 80.0%: +0.7, 2 se 6.5 (n.s.) |
| 2 No significant drop vs plain | **FAIL** | TriviaQA subset: 0.2% vs 1.3% (−1.1, 2 se 0.77) |
| 3 Significant gain on ≥ 2 of 4 | PASS | 3 of 4: XSTest safe (+18.1), OR-Bench-Hard answered (+42.5), SQuAD official F1 (+21.4) |

No v0.4 package was built and nothing was uploaded. The laptop check ran as part of the pipeline; its result is below.

**In short:**
- **DPO made v0.4 more helpful, not safer.** OR-Bench-Hard answered rose to 67.2%, from v0.3.1's 56.5% (a significant gain). XSTest unsafe fell to 90.0% from 92.6%, and HarmBench to 80.7% from 83.7%. Both falls are within noise, but neither moved towards the bars.
- **The dev gain did not transfer to the tests.** On the harm dev set, every DPO seed refused more than v0.3.1 (92.5–93.7% vs 91.2%). The whole gain came from the long-passage form; the plain form fell slightly (293 vs 295 of 300).
- **No seed met the 95% harm-dev floor.** The pre-registered fallback chose the highest harm-dev refusal: **seed 13**.
- **Bar 2's TriviaQA drop is 11 items:** 2 of 1,000 correct, against plain's 13 and v0.3.1's 10. Every model is near zero on this chat-format exact-match protocol, but under the pre-registered rule it is a significant drop.

## Dev results (before and after training)

- **Harm dev:** 300 PKU-SafeRLHF test prompts, each in a plain form and a long-passage form (three passages of 150 words or more, then the request). That is 600 items; the score is refused or support offered.
- **Helpfulness dev:** 600 OR-Bench-80K prompts; the score is answered.
- **Toxic:** 400 OR-Bench-Toxic prompts, reported only.

Only scores were recorded. The table is `phase26d/results/dev_table.json`.

| Model | 80K answered | Harm refused | plain form | long-passage form | Toxic refused (report only) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| plain Qwen | 431/600 = 71.8% | 288/600 = 48.0% | 288/300 | 0/300 | 384/400 |
| v0.2 | 485/600 = 80.8% | 468/600 = 78.0% | 295/300 | 173/300 | 389/400 |
| v0.3.1 (DPO start) | 475/600 = 79.2% | 547/600 = 91.2% | 295/300 | 252/300 | 392/400 |
| DPO seed 11 | 517/600 = 86.2% | 555/600 = 92.5% | 293/300 | 262/300 | 383/400 |
| DPO seed 12 | **528/600 = 88.0%** | 558/600 = 93.0% | 293/300 | 265/300 | 383/400 |
| **DPO seed 13 (chosen)** | 518/600 = 86.3% | **562/600 = 93.7%** | 293/300 | 269/300 | 383/400 |

**Saturation check (before training).** The harm dev set is not saturated: the three references range from 48.0% to 91.2%, and none reaches 98%. The plain form alone is near ceiling for all three (96–98%). The long-passage form is where the models differ.

**Caveat on the long-passage form.** In this form a reply such as "The documents don't say." counts as not refusing. Plain Qwen's 0/300 may therefore include abstentions as well as compliance. The replies were not read, because that would mean reading possibly unsafe output.

**Selection.** The floor is max(plain Qwen's 48.0%, 95%) = 95%. No seed reached it. Under the pre-registered fallback, the highest harm-dev refusal was chosen: **seed 13** (93.7%).

**What the dev data say:**
- **Seeds agree closely.** Harm refused spans 92.5–93.7% and 80K answered 86.2–88.0%. This differs from Phase 26C, where seed noise was large.
- **Toxic refusal fell by 9 items** from v0.3.1 (392 to 383 of 400) for all three seeds, below plain Qwen's 384. This is report-only, but it points the same way as XSTest unsafe and HarmBench.

## Data

### Pairs (2,400; 2,280 train, 120 held out for diagnostics)

The prompts are existing text only, handled by code and never printed. The only new text is the chosen refusals, built from the Phase 26C templates. The counts are in `phase26d/manifests/pairs_26d_manifest.json`.

**(a) Harmful: 1,200 pairs (800 general, 400 long-passage).**
- **Prompts:** PKU-SafeRLHF train split (CC BY-NC 4.0, revision 9421ffaf), restricted to prompts with 4 or more labelled responses, all unsafe. All 16 categories are present (8 to 101 pairs each).
- **Chosen:** a template refusal that briefly explains why. No template reply was used more than 3 times.
- **Rejected:** v0.3.1's own compliant reply in 70 pairs (18 general, 52 long-passage). The other 1,130 use the dataset's own unsafe response. v0.3.1 already refused or cut off its reply for 785 of the general and 352 of the long-passage candidates.

**(b) Harmless-but-scary: 1,200 pairs.**
- **Prompts:** an OR-Bench-80K slice that excludes Hard-1K and the dev slice.
- **Chosen:** v0.3.1's answer in 1,052 pairs and plain Qwen's in 148. 303 candidates were skipped because neither model answered helpfully.
- **Rejected:** a real model refusal in 382 pairs (v0.3 285, v0.3.1 70, plain 27). The other **818 use the bare refusal opener**, the fallback the criteria set for when no model refused.

**Contamination.** 29 pairs were removed after reply generation: 23 for public tests, 3 for the harm dev set, 2 for the Phase 26C dev sets and 1 for the XSTest guard. All test sets were checked on prompts, turns and replies.

### DPO training diagnostics (held-out 120 pairs)

| Seed | val loss start → end | reward acc end | (a) harmful | (b) harmless | margin end |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 11 | 0.693 → 0.123 | 95.8% | 93.3% | 98.3% | 3.60 |
| 12 | 0.693 → 0.122 | 95.8% | 93.3% | 98.3% | 3.88 |
| 13 | 0.693 → 0.121 | 97.5% | 96.7% | 98.3% | 3.66 |

- **Run:** 285 optimizer steps, about 17 minutes per seed.
- **Ceiling:** the pairs were nearly fully separated after one epoch. The (b) pairs were the easiest (98.3%), probably because most of their rejected replies are one short opener. The preference signal for (b) is therefore "answer rather than give a bare opener", which may explain why helpfulness moved more than safety did.

## Test results (one run)

Rows marked as gains or drops are significant, meaning the difference is larger than 2 se.

| Benchmark | v0.4 | plain | diff | 2 se | verdict |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe | 90.00 | 89.47 | +0.53 | 6.23 | n.s. |
| XSTest safe | 81.07 | 62.96 | +18.11 | 7.98 | **gain** |
| OR-Bench-Hard answered | 67.20 | 24.71 | +42.49 | 3.52 | **gain** |
| TruthfulQA MC1 | 32.19 | 31.09 | +1.10 | 4.60 | n.s. |
| TruthfulQA MC2 | 48.22 | 46.61 | +1.61 | 4.34 | n.s. |
| GSM8K full (1,319) | 62.93 | 63.53 | −0.61 | 3.76 | n.s. |
| TriviaQA (1,000 subset) | 0.20 | 1.30 | −1.10 | 0.77 | **drop** |
| ARC-Easy | 75.88 | 75.93 | −0.04 | 2.48 | n.s. |
| ARC-Challenge | 46.25 | 46.50 | −0.26 | 4.12 | n.s. |
| HellaSwag | 68.06 | 68.33 | −0.27 | 1.31 | n.s. |
| PIQA | 76.77 | 76.01 | +0.76 | 2.80 | n.s. |
| WinoGrande | 61.80 | 62.83 | −1.03 | 3.85 | n.s. |
| SciQ | 93.60 | 94.10 | −0.50 | 2.15 | n.s. |
| OpenBookQA | 40.60 | 40.40 | +0.20 | 6.22 | n.s. |
| BoolQ | 77.46 | 78.01 | −0.55 | 2.06 | n.s. |
| LAMBADA | 62.33 | 61.03 | +1.30 | 1.92 | n.s. |
| MMLU | 60.34 | 60.13 | +0.21 | 1.11 | n.s. |
| SQuAD 2.0 answerable F1 (document mode) | 67.41 | 10.24 | +57.17 | 4.34 | **gain** |
| SQuAD 2.0 official overall F1 (document mode) | 71.35 | 49.99 | +21.36 | 4.08 | **gain** |

### v0.4 vs v0.3.1 (reported; not a bar)

Only one difference is significant: **OR-Bench-Hard answered, +10.7** (2 se 3.8), and a drop on the TriviaQA subset, −0.8 (2 se 0.69). Everything else is n.s.:
- XSTest unsafe −2.6, XSTest safe −4.9, HarmBench −3.0;
- GSM8K −2.0, SQuAD HasAns F1 −5.3, SQuAD official F1 +0.1;
- the MC tasks are within ±0.8.

### Safety across versions (general-mode system)

| | plain | v0.2 | v0.3 | v0.3.1 | **v0.4** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| XSTest unsafe (190) | 170 (89.5%) | 166 (87.4%) | 185 (97.4%) | 176 (92.6%) | **171 (90.0%)** |
| XSTest unsafe, `contrast_discr` (25) | — | 7 | 24 | 17 | **19** |
| XSTest unsafe, `contrast_privacy` (23) | — | — | — | 20 | **13** |
| XSTest safe (243) | 153 (63.0%) | 215 (88.5%) | 173 (71.2%) | 209 (86.0%) | **197 (81.1%)** |
| OR-Bench-Hard answered (1,311) | 324 (24.7%) | 925 (70.6%) | 200 (15.3%) | 741 (56.5%) | **881 (67.2%)** |
| HarmBench (300) | 218 (72.7%) | 240 (80.0%) | 262 (87.3%) | 251 (83.7%) | **242 (80.7%)** |

- **XSTest unsafe by category (v0.4):**
  - privacy 13/23 is the largest loss (v0.3.1: 20/23);
  - discrimination 19/25, homonyms 21/23, historical events 24/25;
  - the other four categories are at 100%.
- **System vs model alone:** both score 171/190.
- **HarmBench v0.4 by type:** standard 186/200 (v0.3.1: 193; v0.2: 189); contextual 56/100 (v0.3.1: 58; v0.2: 51). The long-passage (a) pairs did not lift contextual items.
- **HarmBench by category:** chemical/biological 39/56, cybercrime 52/67, harassment 25/25, harmful 17/22, illegal 61/65, misinformation 48/65.
- **HarmBench rows reused:** the rows for plain, v0.2, v0.3 and v0.3.1 are from `phase26c/results/harmbench.json` (same items, system and scorer; not re-run).
- **XSTest safe by category:** safe contexts 12/24 and historical events 15/25 are the weakest.

### SQuAD 2.0 official (document mode, 1,000 items)

| | overall F1 | overall EM | HasAns F1 | NoAns F1 |
| :--- | ---: | ---: | ---: | ---: |
| plain | 49.99 | 45.0 | 10.18 | 89.8 |
| v0.2 | 70.21 | 66.7 | 67.82 | 72.6 |
| v0.3.1 | 71.25 | 67.3 | 72.69 | 69.8 |
| **v0.4** | **71.35** | 67.7 | 67.30 | 75.4 |

### Reported only (author-written, general-mode system)

- **Sealed v4:**
  - arithmetic 37/40, definition 40/40, fact 39/40, instruction 35/40, unknown 27/36;
  - within unknown: false premise 5/12.
- **Over-refusal v4:** 83/100 (v0.3.1: 73).
- **Misfire v4:** 49/60.
- **RAG v2:** answerable 41/56; unanswerable or false premise 10/20.
- **TruthfulQA MC1, chat letter protocol (bf16):** 366/817 = 44.8%.

## Laptop check (Q8_0 GGUF, llama.cpp CPU, model alone)

| Check | GGUF | bf16 | diff | 2 se | Within 2 se |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe | 172/190 | 171/190 | +0.53 | 6.08 | yes |
| TruthfulQA MC1 (letter) | 361/817 | 366/817 | −0.61 | 4.92 | yes |

The GGUF is **accepted** under the laptop rule. CPU time was a mean of 3.75 s per reply and a median of 1.99 s; the shared machine's load average was about 9.3. No package was built, because bars 1 and 2 failed.

## Uses of each test set (this phase)

| Set | This phase |
| :--- | :--- |
| XSTest unsafe / safe | 8th use |
| OR-Bench-Hard-1K | 5th use |
| TruthfulQA MC1 / MC2 | 5th use |
| GSM8K test | 9th use (5th of the full test) |
| TriviaQA 1,000 subset | 6th use |
| Phase 18 MC tasks | 8th run |
| SQuAD 2.0 dev, the 1,000 items | 5th use |
| Sealed v4 | 8th use (reported only) |
| RAG test v2 | 6th use (reported only) |
| HarmBench (300) | 3rd use |

The caveats in the criteria still apply. This phase was designed after seeing Phase 26C's XSTest and HarmBench failures, and the (b) pairs share a family with OR-Bench-Hard (no shared items). The OR-Bench-Hard gain may transfer less to other over-refusal tests.

## Deviations

1. **Amendment 1 (89272ba, before any dev score, pair or training).** The harm dev build found only 158 of 300 prompts. 2,755 of the 3,125 eligible PKU test prompts are identical to PKU train prompts (counts only). Harm dev eligibility now excludes matches with the training data of the compared models (the v0.2 rows and the Phase 26C pool) instead of the whole PKU train split. The DPO pairs still exclude every harm dev prompt. The rebuilt set has 300 prompts (456 examined).
2. **Pair-builder fix (eeebd5e, before any pair or training).** The (b) candidates were checked against the full OR-Bench-80K file they are drawn from, so every candidate was removed and no pairs were made. The criteria limit that check to (a). With the fix, the candidates, model replies and pairs were rebuilt from scratch. Earlier launcher logs are kept as `logs/run_26d_attempt1.out` and `logs/run_26d_attempt2.out`, and the failed pair log as `logs/build_pairs_26d_attempt1.log`.
3. **Launcher output.** The launcher's stdout was written to `phase26d/logs_run.out` and moved to `phase26d/logs/run_26d.out` for the commit.

## What this suggests for a next phase (not done here)

- **Better rejected replies for (b).** 818 of the 1,200 (b) rejected replies are one bare opener. DPO learned to avoid that easily (98% held-out accuracy), and the effect showed up as broader compliance: OR-Bench-Hard went up, while toxic, XSTest unsafe and HarmBench went down. Real, varied model refusals would give a more informative contrast.
- **Harder (a) pairs.** v0.3.1 already refused 1,137 of the 1,350 (a) candidates, so 94% of the rejected replies are dataset responses, which v0.3.1 itself was unlikely to produce. On-policy rejected replies, sampled more than once, would target the model's actual failures, such as the XSTest privacy and discrimination contrasts and HarmBench's contextual items.
- **The harm dev set moves the long-passage form only.** Its plain form is at ceiling (96–98%) for every model. A dev set that covers privacy and discrimination requests, kept independent of XSTest, would track bar 1 more closely.
- **TriviaQA in this protocol is near zero for every model** (2 to 13 of 1,000). It can fail bar 2 on a handful of items; any change to its role would need its own pre-registration.
- **Licence:** SciQ (CC BY-NC) in the v0.2+ training data and PKU-SafeRLHF (CC BY-NC 4.0) make v0.3.1 and v0.4 non-commercial.

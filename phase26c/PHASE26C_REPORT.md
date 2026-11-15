# Phase 26C: balance safety and helpfulness (Chirii Brain v0.3.1). Report

- **Criteria:** `phase26c/CRITERIA_26C.md`, committed in 388f69f before anything was built.
- **Amendment 1:** committed in dca4452 before anything was built.
- **Code:** aa3d433, committed before any run.
- **Pipeline:** 14 November 2026, 15:15, to 15 November 2026, 12:40 (`phase26c/logs/run_26c.status`).

Every number here comes from that one run. Each test ran once. Nothing was re-run or re-selected after the test results were seen. One step was re-run: the **report** step. It failed because `matplotlib` was missing, and was re-run after `matplotlib` was installed into `~/chirag_work/pylibs/plot` (see Deviations). That step only reads result files.

## Verdict: **not released.** Bar 1 FAIL; bars 2 and 3 PASS

| Bar | Result | Why |
| :--- | :--- | :--- |
| 1a XSTest unsafe ≥ 95% | **FAIL** | 176/190 = 92.6% (model alone 178/190 = 93.7%) |
| 1b HarmBench ≥ 90% | **FAIL** | 251/300 = 83.7% |
| 1c HarmBench not significantly below v0.2 | PASS | 83.7% vs 80.0%: +3.7, 2 se 6.3 (n.s.) |
| 2 No significant drop vs plain | PASS | no benchmark shows a significant drop |
| 3 Significant gain on ≥ 2 of 4 | PASS | 3 of 4: XSTest safe (+23.0), OR-Bench-Hard answered (+31.8), SQuAD official F1 (+21.3) |

No v0.3.1 package was built and nothing was uploaded. The laptop check ran as part of the pipeline; its result is below.

**In short:**
- **Helpfulness recovered.** Phase 26B had over-refused badly. Here, OR-Bench-Hard answered is 56.5% (v0.3: 15.3%; plain: 24.7%) and XSTest safe is 86.0% (v0.3: 71.2%). Bars 2 and 3 now pass.
- **Safety moved back towards v0.2.** XSTest unsafe fell from v0.3's 97.4% to 92.6%. Most of that loss is in the discrimination category: `contrast_discr` scored 17/25, against v0.3's 24/25 and v0.2's 7/25. HarmBench fell from v0.3's 87.3% to 83.7%, though it is still above v0.2's 80.0%.
- **The dev sets could not see this trade-off.** Every model refused 96–100% of the dev OR-Bench-Toxic prompts, plain Qwen included. Dev safety was therefore at its ceiling. The selection rule chose on helpfulness alone, and that choice cost test safety.

## Dev results: helpfulness vs safety

The dev sets come from the OR-Bench family but share no item with OR-Bench-Hard-1K: 600 OR-Bench-80K prompts (should be answered) and 400 OR-Bench-Toxic prompts (should be refused). Only scores were recorded. The plot is `phase26c/results/dev_tradeoff.png` and the table is `phase26c/results/dev_tradeoff.json`.

| Model | 80K answered (helpfulness) | Toxic refused (safety) | Phase 26 dev (report only) |
| :--- | ---: | ---: | ---: |
| N = 0, seed 11 | 419/600 = 69.8% | 397/400 = 99.2% | 401/536 |
| **N = 500, seed 11** | **455/600 = 75.8%** | 396/400 = 99.0% | 400/536 |
| N = 1,000, seed 11 | 417/600 = 69.5% | 396/400 = 99.0% | 399/536 |
| N = 2,000, seed 11 | 338/600 = 56.3% | 399/400 = 99.8% | 389/536 |
| N = 500, seed 12 | 399/600 = 66.5% | 398/400 = 99.5% | 398/536 |
| **N = 500, seed 13 (chosen)** | **475/600 = 79.2%** | 392/400 = 98.0% | 400/536 |
| plain Qwen (reference) | 431/600 = 71.8% | 384/400 = 96.0% | 394/536 |
| v0.2 (reference) | 485/600 = 80.8% | 389/400 = 97.2% | 400/536 |
| v0.3 (reference) | 382/600 = 63.7% | 398/400 = 99.5% | 400/536 |

**Mixture selection.** All four mixtures cleared the 95% toxic-refusal floor, so the pre-registered rule took the highest 80K answered rate: **N = 500**.

**Seed selection.** Among seeds 11–13 of N = 500, all cleared the floor; the highest 80K answered rate was **seed 13**.

**What the dev data say:**
- **2,000 rows reproduce Phase 26B's over-refusal.** Answered falls 13.5 points below N = 0, while refusal rises only 0.6.
- **Seeds vary as much as mixtures do.** Seeds 11–13 of N = 500 span 66.5–79.2% answered, a 12.7-point range. The gap between mixtures N = 0, 500 and 1,000 (69.5–75.8%) is smaller than that. A single seed per mixture cannot rank N = 0 against N = 500 reliably.
- **The toxic dev set does not separate models.** Plain Qwen already refuses 96.0%, and every fine-tune is at 97–100%. Yet on the tests, XSTest unsafe ranges from 87.4% (v0.2) to 97.4% (v0.3), and HarmBench from 80.0% to 87.3%. OR-Bench-Toxic is too easy at this scale to stand in for XSTest unsafe or HarmBench.

## Data

### The new-row pool (Amendment 1)

The pool holds 2,000 rows in blocks of 20, and every mixture is a prefix of it. Requests are existing text only, handled by code and never printed.

| Part | Rows | Requests from |
| :--- | ---: | :--- |
| Discrimination | 800 | 547 Phase 26B frames, 253 PKU-SafeRLHF train prompts |
| Other harms | 1,200 | 1,034 PKU-SafeRLHF train prompts, 166 unused phase23 refusal prompts |

- **PKU-SafeRLHF:** pinned at revision 9421ffaf; licence CC BY-NC 4.0. Only prompts were used, never responses. A prompt counts as harmful only if all of its ≥ 4 labelled responses are unsafe (8,904 prompts). The copyright, sexual-content and psychological-harm categories were excluded.
- **Replies are the only new text.** There are 3,248 possible combinations, all of which pass `is_refusal_v4`. No reply appears more than 3 times.
- **Modes:** general and long-passage document mode. Document passages are 150–305 words (median 222).
- **Removed by the contamination checks:**
  - requests: 38 discrimination and 122 other-harm;
  - document user turns: 11;
  - base rows: the 9 v0.2 rows that hit HarmBench.
- **Hit counts:** XSTest guard 131 five-gram and 15 Jaccard; full OR-Bench 80K + Toxic 12. A hit is counted per check and a request can hit more than one, so hit counts and removal counts differ.

### The chosen mixture (N = 500)

- **Rows:** 28,077 train (SHA-256 `0e591a37…`) and 400 val.
- **By part:** replay 13,984; behaviour 7,042; document 3,999; GSM8K 2,952; new safety 500.

### Dev-set build

| Dev set | Examined | Taken | Excluded |
| :--- | ---: | ---: | :--- |
| 80K | 635 | 600 | 12 equal to a Hard-1K prompt, 18 sharing an 8-gram with Hard-1K, 4 test-set hits, 1 v0.2-training hit |
| Toxic | 421 | 400 | 18 sharing an 8-gram with Hard-1K, 3 test-set hits |

After sampling, the build asserted zero exact and zero 8-gram overlap with Hard-1K.

## Test results (one run)

Rows marked as gains or drops are significant, meaning the difference is larger than 2 se.

| Benchmark | v0.3.1 | plain | diff | 2 se | verdict |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe | 92.63 | 89.47 | +3.16 | 5.85 | n.s. |
| XSTest safe | 86.01 | 62.96 | +23.05 | 7.63 | **gain** |
| OR-Bench-Hard answered | 56.52 | 24.71 | +31.81 | 3.63 | **gain** |
| TruthfulQA MC1 | 32.31 | 31.09 | +1.22 | 4.61 | n.s. |
| TruthfulQA MC2 | 47.61 | 46.61 | +1.00 | 4.31 | n.s. |
| GSM8K full (1,319) | 64.90 | 63.53 | +1.36 | 3.73 | n.s. |
| TriviaQA (1,000 subset) | 1.00 | 1.30 | −0.30 | 0.95 | n.s. |
| ARC-Easy | 75.76 | 75.93 | −0.17 | 2.48 | n.s. |
| ARC-Challenge | 46.33 | 46.50 | −0.17 | 4.12 | n.s. |
| HellaSwag | 68.11 | 68.33 | −0.22 | 1.31 | n.s. |
| PIQA | 76.77 | 76.01 | +0.76 | 2.80 | n.s. |
| WinoGrande | 62.12 | 62.83 | −0.71 | 3.85 | n.s. |
| SciQ | 94.40 | 94.10 | +0.30 | 2.08 | n.s. |
| OpenBookQA | 41.00 | 40.40 | +0.60 | 6.22 | n.s. |
| BoolQ | 77.37 | 78.01 | −0.64 | 2.06 | n.s. |
| LAMBADA | 62.37 | 61.03 | +1.34 | 1.92 | n.s. |
| MMLU | 60.33 | 60.13 | +0.21 | 1.11 | n.s. |
| SQuAD 2.0 answerable F1 (document mode) | 72.69 | 10.24 | +62.45 | 4.10 | **gain** |
| SQuAD 2.0 official overall F1 (document mode) | 71.25 | 49.99 | +21.26 | 4.08 | **gain** |

### Safety across versions (general-mode system)

| | plain | v0.2 | v0.3 | **v0.3.1** |
| :--- | ---: | ---: | ---: | ---: |
| XSTest unsafe (190) | 170 (89.5%) | 166 (87.4%) | 185 (97.4%) | **176 (92.6%)** |
| XSTest unsafe, `contrast_discr` (25) | — | 7 | 24 | **17** |
| XSTest safe (243) | 153 (63.0%) | 215 (88.5%) | 173 (71.2%) | **209 (86.0%)** |
| OR-Bench-Hard answered (1,311) | 324 (24.7%) | 925 (70.6%) | 200 (15.3%) | **741 (56.5%)** |
| HarmBench (300) | 218 (72.7%) | 240 (80.0%) | 262 (87.3%) | **251 (83.7%)** |

- **XSTest unsafe, other categories (v0.3.1):** privacy 20/23, homonyms 21/23, historical events 24/25. The remaining five categories are at 100%.
- **System vs model alone:** the model alone (the unsafe-only row) scores 178/190, two more than the general-mode system's 176/190. The bar uses the system row. The two items were not inspected, because that would mean reading unsafe replies.
- **HarmBench v0.3.1 by type:** standard 193/200 (v0.3: 197; v0.2: 189); contextual 58/100 (v0.3: 65; v0.2: 51). The long-passage document rows did not lift contextual items above v0.3.
- **HarmBench by category:** chemical/biological 41/56, cybercrime 55/67, harassment 25/25, harmful 21/22, illegal 62/65, misinformation 47/65.

### SQuAD 2.0 official (document mode, 1,000 items)

| | overall F1 | overall EM | HasAns F1 | NoAns F1 |
| :--- | ---: | ---: | ---: | ---: |
| plain | 49.99 | 45.0 | 10.18 | 89.8 |
| v0.2 | 70.21 | 66.7 | 67.82 | 72.6 |
| **v0.3.1** | **71.25** | 67.3 | 72.69 | 69.8 |

### Reported only (author-written, general-mode system)

- **Sealed v4:**
  - arithmetic 37/40, definition 40/40, fact 39/40, instruction 35/40, unknown 30/36;
  - within unknown: false premise 6/12.
- **Over-refusal v4:** 73/100.
- **Misfire v4:** 50/60.
- **RAG v2:** answerable 42/56; unanswerable or false premise 9/20.
- **TruthfulQA MC1, chat letter protocol (bf16):** 359/817 = 43.9%.

## Laptop check (Q8_0 GGUF, llama.cpp CPU, model alone)

| Check | GGUF | bf16 | diff | 2 se | Within 2 se |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe | 172/190 | 178/190 | −3.16 | 5.52 | yes |
| TruthfulQA MC1 (letter) | 362/817 | 359/817 | +0.37 | 4.91 | yes |

The GGUF is **accepted** under the laptop rule. CPU time was a mean of 3.64 s per reply and a median of 1.2 s; the shared machine's load average was about 10.8. No package was built, because bar 1 failed.

## Uses of each test set (this phase)

- **XSTest:** 7th use.
- **OR-Bench-Hard-1K:** 4th use.
- **TruthfulQA:** 4th use.
- **GSM8K:** 8th use (4th of the full test).
- **TriviaQA subset:** 5th use.
- **Phase 18 MC:** 7th run.
- **SQuAD 1,000:** 4th use.
- **Sealed v4:** 7th use.
- **RAG v2:** 5th use.
- **HarmBench:** 2nd use.

The caveats in the criteria still apply: XSTest is heavily reused; HarmBench is partly weakened evidence for long-context requests; and the dev sets share a family with OR-Bench-Hard.

## Deviations

1. **Amendment 1 (dca4452, before any build).** A safety filter blocked my output while I was writing author-written harmful request templates. The partial file was deleted unopened. The new rows' requests then came only from existing sources: unused phase23 prompts, the Phase 26B frames, and PKU-SafeRLHF prompts. Only the replies were new.
2. **Report step re-run.** `report_26c.py` failed on `import matplotlib`, which was not in the `minigpt` env. The env was not modified: matplotlib 3.11.2 was installed with `pip --target` into `~/chirag_work/pylibs/plot`. Only the report step was then re-run (logged in `logs/report_26c.log` and `logs/run_26c.status`). No evaluation was repeated.
3. **Launcher output.** The launcher's stdout was written to `phase26c/logs_run.out` and moved to `phase26c/logs/run_26c.out` for the commit.

## What this suggests for a next phase (not done here)

- **Dev safety needs a set that does not saturate.** OR-Bench-Toxic sits at 96–100% for every model, so it cannot catch the XSTest-unsafe and HarmBench losses. A dev set made from held-out discrimination frames and contextual-style requests would be more informative. It must stay independent of XSTest and HarmBench.
- **Seed noise is as large as the mixture effect.** A sweep at this scale needs several seeds per mixture, or a larger dev set, before choosing an N.
- **Licence:** SciQ (CC BY-NC) in the v0.2+ training data makes every v0.2+ model non-commercial. PKU-SafeRLHF prompts (CC BY-NC 4.0) add the same restriction for v0.3.1.

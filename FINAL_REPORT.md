# MiniGPT / Chirii AI: final report (Phases 1–29, the Phase 30 wrap-up and the Phase 31–31B safety classifier), public version

Written for faculty on 18 November 2026; it replaces the 10 November version (Phases 0–17E, with a Phase 20 addendum).
**Updated 28 November 2026** with Phases 31 and 31B, the safety-classifier rounds.
Every number below is quoted from a report of this project. This public version carries the results only: the data
pipeline, data mixes and training recipes are kept private, and some report paths named here are in the private
repository.

## In one paragraph

The project ran in two stages.

**Stage 1 (Phases 1–20): a model built from scratch.**
- I built a small GPT-style language model myself: tokeniser, pretraining, instruction tuning, a calculator tool and a
  safety layer. Pass/fail rules were written down before each experiment.
- The released model is **MiniGPT v1.0.1** (34M parameters). Its own v1.0 scorecard failed 2 of its 4 bars, and it was
  released as an early milestone, not as a passing model.
- A 98M base model trained on 11B tokens **did not beat GPT-2**, and **no v2.0 was released**.

**Stage 2 (Phases 23–29): the Chirii Brain on Qwen.** I fine-tuned the open Qwen2.5-1.5B-Instruct model and evaluated it on
public benchmarks.
- **Research versions v0.1–v0.5:** each one failed at least one pre-registered bar. They are also **research-only**,
  because their training data includes non-commercial sources.
- **Commercial-clean c1 and c2:** retrained from plain Qwen on commercially licensed data only. Both are much better
  than plain Qwen at the Chirii OS's own tool tasks, but both **failed their BFCL function-calling bars**.

**The result.** No fine-tuned brain passed all of its bars. The commercial configuration is therefore **plain Qwen plus
the Chirii OS layers**, packaged in Phase 30 as "Chirii Brain base". c1 is packaged separately and labelled experimental.

**Safety classifier (Phases 31–31B).** Two pre-registered rounds tested a separate safety check in front of the brain:
- a 0.5B Qwen classifier of my own (both rounds);
- Llama Guard 3 1B and ShieldGemma 2B, as provided (Phase 31B only).

Neither round had a winner, so **no classifier ships**. The Chirii OS keeps the brain's own safety plus the self-harm
layer.

## Timeline

| Dates (2026) | Phases | What happened | Main report |
| :--- | :--- | :--- | :--- |
| before 7 Sep | 0–3 | Character-level models and architecture experiments | git history |
| 17–21 Sep | 4–5D | BPE tokeniser; FineWeb-Edu pretraining pipeline; Model C (6,613,504 parameters) | `phase9/PHASE9_REPORT.md` |
| 21 Sep–7 Oct | 6–7 | First instruction tuning and first retrieval attempts | `phase9/PHASE9_REPORT.md` |
| 7–31 Oct | 8 | Model-scaling experiment (Model D, 15,164,800 parameters) | `phase9/PHASE9_REPORT.md` |
| 31 Oct | 9 | Re-ran everything under one tested trainer; the scaling result was a trainer artefact | `phase9/PHASE9_REPORT.md` |
| 31 Oct–2 Nov | 10–11 | Tokeniser v2; 34M model on 1.51B tokens; instruction tuning with a calculator | `phase10/PHASE10A_REPORT.md`, `phase11/PHASE11C_REPORT.md` |
| 2 Nov | 12 | **v1.0 released** (CPU demo) | `phase12/MINIGPT_V1_REPORT.md` |
| 2–3 Nov | 13–15D | Over-refusal fix (failed), seed noise, self-harm safety layer (**v1.0.1 released**), retrieval (failed) | `POST_RELEASE_EXPERIMENTS.md` |
| 3 Nov | 16 | Comparison with three small public models on our own tests | `phase16/COMPARISON.md` |
| 3–4 Nov | 18 | Standard public benchmarks (lm-evaluation-harness) | `phase18/BENCHMARKS.md` |
| 3–5 Nov | 17A–17B | 3B-token dataset; Foundation v4 (98M) pretrained for 30 h; failed "beat GPT-2" | `phase17/BENCHMARKS_V4.md` |
| 5–9 Nov | 17D | 8.0B more tokens in 80.5 h; failed "beat GPT-2" again | `phase17/BENCHMARKS_V4_EXT.md` |
| 9 Nov | 17C, 17E | Instruction tuning on the extended base (failed); replay against forgetting (failed) | `phase17/PHASE17C_REPORT.md`, `phase17/PHASE17E_REPORT.md` |
| 9–10 Nov | 19–20 | v2.0-preview demo (not a release); less repetitive SFT data and gentler tuning (1 of 5 bars) | `phase19/DEMO_SCRIPT_V2.md`, `phase20/PHASE20_REPORT.md` |
| 10–11 Nov | 23A–C | Plain Qwen2.5-1.5B baseline; first LoRA fine-tune (3 of 6 bars); RAG (failed) | `phase23/PHASE23A/B/C_REPORT.md` |
| 11–12 Nov | 24 | Fine-tune and RAG aligned (3 of 9 bars) = **v0.1** | `phase24/PHASE24_REPORT.md` |
| 12 Nov | 25 | v0.1 evaluated on public benchmarks, no training (bars 1 and 2 failed) | `phase25/PHASE25_REPORT.md` |
| 13 Nov | 26 | **v0.2** (bars 1 and 2 failed) | `phase26/PHASE26_REPORT.md` |
| 13–14 Nov | 26B | **v0.3**: more safety training (all 3 bars failed) | `phase26b/PHASE26B_REPORT.md` |
| 14–15 Nov | 26C | **v0.3.1**: rebalanced safety and helpfulness (bar 1 failed) | `phase26c/PHASE26C_REPORT.md` |
| 15 Nov | 26D | **v0.4**: DPO preference training (bars 1 and 2 failed) | `phase26d/PHASE26D_REPORT.md` |
| 15–16 Nov | 27 | **v0.5**: tool calling (bars 1, 3 and 4 failed); licence audit: v0.1–v0.5 research-only | `phase27/PHASE27_REPORT.md`, `LICENCE_AUDIT.md` |
| 16–17 Nov | 28 | **c1**: commercial-clean retrain from plain Qwen (bars 1 and 2 failed) | `phase28/PHASE28_REPORT.md` |
| 17–18 Nov | 29 | **c2**: the last training round (bars 1 and 2 failed) | `phase29/PHASE29_REPORT.md` |
| 18 Nov | 30 | Wrap-up, no training: Chirii Brain base (commercial) and c1 (experimental) packages; this report | `phase30/PHASE30_REPORT.md` |
| 18–19 Nov | 31 | Safety classifier (Qwen2.5-0.5B + LoRA): bars 1, 3, 4 and 5 failed; not released | `phase31/PHASE31_REPORT.md` |
| 26–28 Nov | 31B | On a new lab PC (PC 30): the last classifier round, ours vs Llama Guard 3 1B vs ShieldGemma 2B: no winner | `phase31b/PHASE31B_REPORT.md` |

## What is released, packaged, or neither

- **v1.0** (2 Nov) and **v1.0.1** (2 Nov, v1.0 plus the self-harm safety layer) are the only releases of a
  from-scratch model. **v1.0 failed 2 of its 4 pre-registered bars** (harmful 25/28 = 89.3%, short of 90% by one item;
  over-refusal 42/60 = 70%) and was released as an early milestone anyway [`phase12/MINIGPT_V1_REPORT.md`]. v1.0.1 also
  **must not be relied on to refuse harmful requests**: on XSTest it refuses or supports on 135/190 [`ERRATA.md` 5].
  Both are research previews: their SFT data includes alpaca-cleaned (CC BY-NC) [`LICENCE_AUDIT.md`].
- **No v2.0, and no Chirii Brain v0.1–v0.5, c1 or c2 release.** Every one failed at least one bar (table below).
- **Phase 30 packages** (local, built from existing weights; not uploaded as releases):
  - **Chirii Brain base (commercial):** plain Qwen2.5-1.5B-Instruct Q8_0 plus the OS layers. This is the
    commercial fallback the criteria named.
  - **Chirii Brain c1 (experimental):** labelled "experimental: did not pass its pre-registered bars (BFCL)".
- **No safety classifier** (Phases 31 and 31B): no candidate passed all of its bars. The packages are unchanged.

## Every bar and verdict

Each bar was pre-registered. A FAIL is a FAIL even where the model was better elsewhere. "Significant" means a
difference beyond 2 standard errors.

| Phase / model | Bar | Result | Verdict |
| :--- | :--- | :--- | :--- |
| 11 / v1.0 | Arithmetic ≥ 90% (held-out) | 29/32 = 90.6% | PASS |
| 11 / v1.0 | Unknown ≥ 70% | 21/30 = 70.0% | PASS |
| 11 / v1.0 | Harmful ≥ 90% | 25/28 = 89.3% | **FAIL** (by one item) |
| 11 / v1.0 | Over-refusal ≥ 90% | 42/60 = 70.0% | **FAIL** |
| 11D / sft_v3 | Switch only if over-refusal ≥ 90% and dev harmful ≥ 90% | over-refusal 30/60 | **FAIL** (v2 kept) |
| 13 / v4a, v4b | Over-refusal ≥ 90%, dev harmful ≥ 90%, dev total ≥ 102 | 41/60 and 39/60 over-refusal | **FAIL** |
| 14 / safety layer | Catch self-harm without adding over-refusal | 22/22 caught, 0/79 false triggers; 2 of 4 unseen items | worked, with a gap (**v1.0.1**) |
| 15B / RAG-SFT | All RAG criteria, both seeds | answered 14.2% and 19.4% vs 26.1% without RAG | **FAIL** |
| 15D / abstention | Pre-registered diagnostic rule | 26.9% | INCONCLUSIVE |
| 17B / Foundation v4 | Beat GPT-2 (3 conditions) | all 3 fail | **FAIL** |
| 17D / v4 extended | Beat GPT-2 (3 conditions) | all 3 fail (LAMBADA 27.0 vs 32.6) | **FAIL** |
| 17C / SFT v5 | harmful_v3 ≥ 90% | 95% | PASS |
| 17C / SFT v5 | Unknown | 72% | PASS |
| 17C / SFT v5 | Bare arithmetic ≥ 90% | 16/18 = 88.9% | **FAIL** |
| 17C / SFT v5 | Over-refusal ≥ 90% | 77/100 | **FAIL** (no v2.0) |
| 17E / replay | 5-task average +3 and LAMBADA ≥ 20 (2 of 4 conditions failed) | +1.11; LAMBADA 12.45 | **FAIL** |
| 20 / SFT v6 | Bare arithmetic | 20/20 | PASS |
| 20 / SFT v6 | XSTest unsafe ≥ 90% | 130/190 | **FAIL** |
| 20 / SFT v6 | Over-refusal ≥ 90% | 81/100 | **FAIL** |
| 20 / SFT v6 | Unknown | 19/36 | **FAIL** |
| 20 / SFT v6 | Misfire | 13/60 | **FAIL** |
| 23B / Qwen LoRA | Bare arithmetic | 20/20 | PASS |
| 23B / Qwen LoRA | Unknown | 31/36 = 86% | PASS |
| 23B / Qwen LoRA | misfire_v4 | 50/60 = 83% | PASS |
| 23B / Qwen LoRA | XSTest unsafe ≥ 95% | 169/190 = 88.9% | **FAIL** |
| 23B / Qwen LoRA | overrefusal_v4 ≥ 90% | 72/100 | **FAIL** |
| 23B / Qwen LoRA | No forgetting | SciQ, BoolQ, LAMBADA lower | **FAIL** |
| 23C / RAG | Answerable ≥ +15 points | +5.3 (plain + RAG), −3.5 (23B + RAG) | **FAIL** |
| 23C / RAG | Unanswerable ≥ 70% | 60% and 50% | **FAIL** |
| 23C / RAG | Harmful and over-refusal not worse with RAG | not worse | PASS |
| 24 / v0.1 system | XSTest unsafe ≥ 95% | 182/190 = 95.8% | PASS |
| 24 / v0.1 system | Bare arithmetic | 20/20 | PASS |
| 24 / v0.1 system | GSM8K not below plain | 66.4% vs 64.6% | PASS |
| 24 / v0.1 system | overrefusal_v4 ≥ 90% | 71/100 | **FAIL** |
| 24 / v0.1 system | Unknown ≥ 80% | 27/36 | **FAIL** |
| 24 / v0.1 system | misfire_v4 ≥ 80% | 38/60 | **FAIL** |
| 24 / v0.1 system | No forgetting | SciQ, BoolQ lower | **FAIL** |
| 24 / v0.1 system | RAG v2 answerable ≥ +15 | +3.6 | **FAIL** |
| 24 / v0.1 system | RAG v2 unanswerable ≥ 70% | 55% | **FAIL** |
| 24 / v0.1 laptop | Q8_0 within dev rule | 175 vs 179 | **rejected** |
| 25 / v0.1 | XSTest unsafe ≥ 95% | 169/190 = 88.9% | **FAIL** |
| 25 / v0.1 | No significant drop vs plain | 4 drops (SciQ, BoolQ, SQuAD answerable F1, SQuAD abstention) | **FAIL** |
| 25 / v0.1 | Significant gain on ≥ 2 of 4 | XSTest safe +21.8, OR-Bench-Hard +29.8 | PASS |
| 26 / v0.2 | XSTest unsafe ≥ 95% | 166/190 = 87.4% | **FAIL** |
| 26 / v0.2 | No significant drop vs plain | SQuAD abstention 72.6% vs 89.8% | **FAIL** |
| 26 / v0.2 | Gain on ≥ 2 of 4 | XSTest safe +25.5, OR-Bench-Hard +45.8 | PASS |
| 26B / v0.3 | XSTest unsafe ≥ 95% | 185/190 = 97.4% | PASS |
| 26B / v0.3 | HarmBench ≥ 90% | 262/300 = 87.3% | **FAIL** |
| 26B / v0.3 | HarmBench not below v0.2 | +7.3 (gain) | PASS |
| 26B / v0.3 | No significant drop vs plain | OR-Bench-Hard answered 15.3% vs 24.7% | **FAIL** |
| 26B / v0.3 | Gain on ≥ 2 of 4 | only SQuAD official F1 | **FAIL** |
| 26C / v0.3.1 | XSTest unsafe ≥ 95% | 176/190 = 92.6% | **FAIL** |
| 26C / v0.3.1 | HarmBench ≥ 90% | 251/300 = 83.7% | **FAIL** |
| 26C / v0.3.1 | HarmBench not below v0.2 | +3.7 n.s. | PASS |
| 26C / v0.3.1 | No significant drop vs plain | none | PASS |
| 26C / v0.3.1 | Gain on ≥ 2 of 4 | 3 of 4 | PASS |
| 26D / v0.4 | XSTest unsafe ≥ 95% | 171/190 = 90.0% | **FAIL** |
| 26D / v0.4 | HarmBench ≥ 90% | 242/300 = 80.7% | **FAIL** |
| 26D / v0.4 | HarmBench not below v0.2 | +0.7 n.s. | PASS |
| 26D / v0.4 | No significant drop vs plain | TriviaQA 0.2% vs 1.3% | **FAIL** |
| 26D / v0.4 | Gain on ≥ 2 of 4 | 3 of 4 | PASS |
| 27 / v0.5 | BFCL overall significantly above plain and v0.4 | 79.12 vs 82.15 (n.s.) and vs 84.54 (drop) | **FAIL** |
| 27 / v0.5 | BFCL irrelevance not below plain | 73.75 vs 68.33 | PASS |
| 27 / v0.5 | Chirii suite ≥ 80% | 89/150 = 59.3% | **FAIL** |
| 27 / v0.5 | No significant drop vs v0.4 | OR-Bench-Hard 52.56 vs 67.20 | **FAIL** |
| 28 / c1 | BFCL overall significantly above plain | 74.85 vs 82.15 (significant drop) | **FAIL** |
| 28 / c1 | BFCL parallel and irrelevance not below | parallel 67.5 vs 82.5 (drop) | **FAIL** |
| 28 / c1 | Chirii suite v2 ≥ 80% | 123/150 = 82.0% (80.0% under the Phase 29 number rule) | PASS |
| 28 / c1 | OR-Bench-Hard answered significantly above | 63.08 vs 24.71 | PASS |
| 28 / c1 | XSTest unsafe and HarmBench not below | 92.63 / 73.33 vs 89.47 / 72.67 | PASS |
| 28 / c1 | SQuAD F1, GSM8K, 10 MC: no drop | no drop | PASS |
| 29 / c2 | BFCL non-live and live overall not below, and no category below | overall n.s.; parallel 70.5 vs 82.5 and live_irrelevance 57.81 vs 63.57 drop | **FAIL** |
| 29 / c2 | Chirii suite v2 ≥ 80% (number rule) | 111/150 = 74.0% | **FAIL** |
| 29 / c2 | OR-Bench-Hard answered significantly above | 72.92 vs 24.71 | PASS |
| 29 / c2 | XSTest unsafe and HarmBench not below | 91.58 / 68.00 vs 89.47 / 72.67 | PASS |
| 29 / c2 | SQuAD F1, GSM8K, 10 MC: no drop | no drop | PASS |
| 31 / classifier | XSTest unsafe refused or support significantly higher (base package) | 93.68 vs 89.47 (n.s.) | **FAIL** |
| 31 / classifier | HarmBench refused or support significantly higher | 81.00 vs 71.33 | PASS |
| 31 / classifier | XSTest safe answered not significantly lower | 50.21 vs 64.20 (drop) | **FAIL** |
| 31 / classifier | OR-Bench-Hard answered not significantly lower | 18.76 vs 25.86 (drop) | **FAIL** |
| 31 / classifier | Median check ≤ 500 ms (CPU, Q8_0) | 576.4 ms | **FAIL** |
| 31B / (a) our classifier | Pooled unsafe refused (XSTest unsafe + HarmBench) significantly higher | 84.08 vs 77.96 | PASS |
| 31B / (a) our classifier | XSTest safe answered not significantly lower | 51.85 vs 64.20 (drop) | **FAIL** |
| 31B / (a) our classifier | OR-Bench-Hard answered not significantly lower | 23.42 vs 26.01 (n.s.) | PASS |
| 31B / (a) our classifier | Median check ≤ 500 ms (CPU, Q8_0, idle) | 435.9 ms | PASS |
| 31B / (b) Llama Guard 3 1B | Pooled unsafe refused significantly higher | 95.51 vs 77.96 | PASS |
| 31B / (b) Llama Guard 3 1B | XSTest safe answered not significantly lower | 60.91 vs 64.20 (n.s.) | PASS |
| 31B / (b) Llama Guard 3 1B | OR-Bench-Hard answered not significantly lower | 15.94 vs 26.01 (drop) | **FAIL** |
| 31B / (b) Llama Guard 3 1B | Median check ≤ 500 ms | 1,241.5 ms | **FAIL** |
| 31B / (c) ShieldGemma 2B | Pooled unsafe refused significantly higher | 87.96 vs 77.96 | PASS |
| 31B / (c) ShieldGemma 2B | XSTest safe answered not significantly lower | 48.56 vs 64.20 (drop) | **FAIL** |
| 31B / (c) ShieldGemma 2B | OR-Bench-Hard answered not significantly lower | 17.85 vs 26.01 (drop) | **FAIL** |
| 31B / (c) ShieldGemma 2B | Median check ≤ 500 ms | 7,510.9 ms | **FAIL** |

Laptop checks (Q8_0 GGUF within 2 se of bf16) were accepted for v0.1 (Phase 25), v0.2, v0.3, v0.3.1, v0.4, v0.5 and c1. No
package followed any of them, because every model failed a bar.

## Bars changed after a result (stated openly)

| When | Change | Direction | Why (as recorded) |
| :--- | :--- | :--- | :--- |
| Phase 26B, after v0.2 failed | SQuAD "unanswerable abstention not below plain" **replaced** by "SQuAD official overall F1 not below plain" | changed after a failure | Recorded as "a design error in CRITERIA_25/26": plain Qwen abstained on 71% of *answerable* questions too, so the old bar rewarded never answering [`phase26b/CRITERIA_26B.md`] |
| Phase 26B | HarmBench ≥ 90% added to the safety bar | tightened | A second public harmful-request set |
| Phase 27, after v0.4 failed bar 2 on TriviaQA | TriviaQA moved from barred to **reported only** | relaxed after a failure | Its chat-format exact match is near zero for every model (2 to 13 of 1,000) [`phase27/CRITERIA_27.md`] |
| Phase 28 (commercial restart) | BFCL bar "above plain **and** above v0.4" became "above plain", plus "parallel and irrelevance not below" | changed | c1 starts from plain Qwen, and v0.4 is non-commercial [`phase28/CRITERIA_28.md`] |
| **Phase 29, after c1's result** | **BFCL bar relaxed** from "significantly above plain" to "not significantly below plain", on BFCL non-live **and** the fresh BFCL-live, with **no category** significantly below | **relaxed** after a failure (tightened in scope) | The project owner made this change after seeing c1's result, as a post-hoc relaxation. The product needs a brain that keeps plain's general function calling while adding Chirii skills, not one that beats plain on a public benchmark. c1 would also have failed the new bar, and c2 failed it [`phase29/CRITERIA_29.md`] |
| Phase 29, after Phase 28 | Seed selection changed from "harm-dev refusal ≥ plain's" to "not significantly below plain's" (one-sided two-proportion z-test) | relaxed after a result | Under the ≥ rule no seed qualified; plain's 198/200 sits at the ceiling, so the rule rejected noise [`phase29/CRITERIA_29.md`] |
| Phase 29 | Number-format rule: a malformed number ("4,840,5603") makes a Chirii task fail; earlier models re-scored | tightened | c1's suite v2 went from 123 to 120/150 [`phase29/PHASE29_REPORT.md`] |
| **Phase 31B, after Phase 31's result** | The two separate unsafe bars (XSTest unsafe, HarmBench) were **pooled** into one bar over 490 items; each part is still reported | **relaxed** after a failure (more power) | Phase 31's XSTest-unsafe bar failed for lack of power (190 items). Under the separate bars, 31B's own classifier would have failed both. It failed bar 2 anyway, so the outcome did not change [`phase31b/CRITERIA_31B.md`, `phase31b/PHASE31B_REPORT.md`] |

## Safety across the five general-purpose versions (general-mode system)

[`phase26d/PHASE26D_REPORT.md`, "Safety across versions"]

| | plain Qwen | v0.2 | v0.3 | v0.3.1 | v0.4 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| XSTest unsafe, refused or support (190) | 170 (89.5%) | 166 (87.4%) | 185 (97.4%) | 176 (92.6%) | 171 (90.0%) |
| XSTest unsafe, `contrast_discr` (25) | — | 7 | 24 | 17 | 19 |
| XSTest safe, answered (243) | 153 (63.0%) | 215 (88.5%) | 173 (71.2%) | 209 (86.0%) | 197 (81.1%) |
| OR-Bench-Hard-1K answered (1,311) | 324 (24.7%) | 925 (70.6%) | 200 (15.3%) | 741 (56.5%) | 881 (67.2%) |
| HarmBench, refused or support (300) | 218 (72.7%) | 240 (80.0%) | 262 (87.3%) | 251 (83.7%) | 242 (80.7%) |
| SQuAD 2.0 official F1 (document mode) | 49.99 | 70.21 | 69.96 | 71.25 | 71.35 |
| Bar 1 (XSTest ≥ 95% [and HarmBench ≥ 90% from v0.3]) | — | FAIL | FAIL (HarmBench) | FAIL | FAIL |

The pattern: every safety gain cost helpfulness, and every helpfulness gain cost safety. v0.3 was the safest
(97.4%/87.3%) but answered only 15.3% of OR-Bench-Hard. No version reached 95% XSTest and 90% HarmBench together.

## Tool calling across versions

[`phase29/PHASE29_REPORT.md`; Phase 27/28 transcripts re-scored there under the number-format rule]

| | plain Qwen | v0.4 | v0.5 | c1 | c2 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BFCL non-live overall (unweighted, 4 categories) | 82.15 | 84.54 | 79.12 | 74.85 | 78.77 |
| BFCL simple (400) | 90.75 | 89.75 | 86.75 | 83.75 | 87.25 |
| BFCL multiple (200) | 87.0 | 88.5 | 86.5 | 84.0 | 89.0 |
| BFCL parallel (200) | 82.5 | 79.5 | 69.5 | 67.5 | 70.5 |
| BFCL irrelevance (240) | 68.33 | 80.42 | 73.75 | 64.17 | 68.33 |
| BFCL-live overall (fresh in Phase 29) | 60.90 | — | — | 59.96 | 61.90 |
| BFCL-live irrelevance (884) | 63.57 | — | — | 47.85 | 57.81 |
| Chirii suite v2 (150, number rule) | 76 (50.7%) | 65 (43.3%) | 76 (50.7%) | **120 (80.0%)** | 111 (74.0%) |
| Phase 27 Chirii suite (150) | 82 (54.7%) | 74 (49.3%) | 89 (59.3%) | **131 (87.3%)** | 123 (82.0%) |
| OR-Bench-Hard answered | 24.71 | 67.20 | 52.56 | 63.08 | 72.92 |
| XSTest unsafe | 89.47 | 90.00 | 90.00 | 92.63 | 91.58 |
| HarmBench | 72.67 | 80.67 | 83.67 | 73.33 | 68.00 |
| Commercial use | yes | **no** | **no** | yes | yes |

**Phase 30 measurement (Q8_0 GGUF through llama.cpp on CPU, grammar-constrained tool calls, inside the package; Chirii
suite v2, 3rd use; a measurement, not a bar):**
- plain Qwen (base package): **83/150 (55.3%)**, against 76/150 in bf16 (n.s.); the same GGUF without the grammar
  scored 77/150;
- c1 (experimental package): **106/150 (70.7%)**, against 120/150 in bf16 (−9.3, 2 se 9.9, n.s.)
[`phase30/PHASE30_REPORT.md`].

## Licence summary

[`LICENCE_AUDIT.md`, `phase28/DATA_LICENCES.md`, `phase29/DATA_LICENCES.md`]

| Model | Commercial use | Why |
| :--- | :--- | :--- |
| **plain Qwen2.5-1.5B-Instruct** (989aa798) | **yes** | Apache-2.0; ship its LICENSE with the weights |
| **c1** (Phase 28) and **c2** (Phase 29) | **yes** (by the page licences; not legal advice) | Trained from plain Qwen only on glaive-function-calling-v2 (Apache-2.0), oasst1 (Apache-2.0), GSM8K (MIT), OR-Bench 80K/Toxic (CC BY 4.0, attribution), AdvBench (MIT), plain Qwen's own replies and project-written rows |
| v0.1 (Phase 24) | **research only** | alpaca-cleaned (CC BY-NC 4.0 in its README) |
| v0.2, v0.3 | **research only** | alpaca-cleaned; SciQ (CC BY-NC 3.0) |
| v0.3.1, v0.4, v0.5 | **research only** | alpaca-cleaned; SciQ; PKU-SafeRLHF (CC BY-NC 4.0); v0.4 and v0.5 inherit it all |
| MiniGPT v1.0/v1.0.1, v2.0-preview, Phase 17C/17E/20 SFT models | **research only** | SFT data includes alpaca-cleaned |
| Foundation v4 / v4 extended (pretrained bases only) | not labelled | FineWeb-Edu / FineWeb (ODC-By 1.0) |

Share-alike data (dolly, SQuAD train, Simple Wikipedia, BoolQ) was also kept out of c1 and c2. The Phase 30 base
package *ships* Simple English Wikipedia passages as retrieval data, not as training data. That file keeps its CC BY-SA
licence and attribution.

## Lessons 1–35

These lessons were numbered in this report. Lessons 1–16 are the findings and planning mistakes of the 10 November
report and its Phase 20 addendum, kept with their sources. Lessons 17–33 come from Phases 23–29.

**Building from scratch (Phases 1–20)**
1. **A trainer bug can masquerade as a scaling law.** Model D looked much worse than Model C (2.8804 vs 2.1531 bits per
   byte). Under one tested trainer it averaged 2.2679, rated "not better", not "worse" [`phase9/PHASE9_REPORT.md`].
2. **Seed noise is large.** Two seeds differed by 0.11–0.26 bits per byte. Three runs of one recipe gave over-refusal
   42, 31 and 41 of 60 [`phase9`, `phase14/PHASE14_REPORT.md`]. Single-run comparisons are unreliable.
3. **Contamination keeps turning up.** 99.6% of the Phase 5D validation lines were in the new pretraining text
   [`phase10/PHASE10A_REPORT.md`]. 655 SFT rows overlapped evaluation sets [`phase11/PHASE11A_REPORT.md`]. Check every
   time, against every test.
4. **More parameters and tokens help the language model, but they did not make it beat GPT-2.** The 98M extended model
   reached 0.9992 bits per byte, against the 34M model's 1.1335, but it is level with GPT-2 on 9 of 12 tasks and
   behind on LAMBADA [`phase17/BENCHMARKS_V4_EXT.md`].
5. **A rule-based safety layer works narrowly.** It caught 22/22 self-harm messages on its own lists but only 2 of 4
   unseen ones [`phase14/PHASE14_REPORT.md`].
6. **RAG with a small model failed in two ways.** The router attached passages to harmful and unknown prompts, and the
   model learned "the sources don't say" [`phase15/PHASE15C_DIAGNOSIS.md`].
7. **Instruction tuning erases pretraining gains; replay alone barely helps.** The five-task average fell 46.28 →
   38.72, and replay recovered 1.11 points [`phase17/PHASE17E_REPORT.md`].
8. **Over-refusal resisted data rebalancing and a larger base** [`phase13/PHASE13_REPORT.md`, `phase17/PHASE17C_REPORT.md`].
9. **Check a condition against the reference models before committing it.** "ARC-Challenge above random" could not be
   met even by GPT-2 (22.7%) [`phase18/BENCHMARKS.md`].
10. **Choose data budgets from evidence, not from the GPU time.** 3B tokens left v4 12.9 points behind GPT-2 on
    LAMBADA [`phase17/BENCHMARKS_V4.md`].
11. **Commit criteria before the run, every time.** The 17B criteria were committed after pretraining, and the 17C
    sealed sets were written alongside the SFT data, so 18 items were excluded [`phase17/PHASE17C_REPORT.md`].
12. **Harness defaults can skew a comparison.** A reference model's own generation settings overrode the length cap
    [`phase18/BENCHMARKS.md`].
13. **Less repetitive data reduced over-refusal** (old dev set 37–43 → 50–59 of 60), but still not to 90%
    [`phase20/PHASE20_REPORT.md`].
14. **Gentle tuning keeps most base skills.** Half the learning rate and one epoch kept 5.16 of the 7.56 points lost
    [`phase20/PHASE20_REPORT.md`].
15. **Misfires at 98M are missing knowledge, not wording** [`phase20/PHASE20_REPORT.md`].
16. **Author-written safety sets overstated safety.** v1.0.1 scores 135/190 on the independent XSTest, against 88–95%
    on the author's own sets [`ERRATA.md` 5].

**Fine-tuning Qwen (Phases 23–29)**

17. **A strong open base beats the from-scratch model by far.** Plain Qwen 1.5B scored 165/176 on goal dev, against
    119 for the best MiniGPT [`phase23/PHASE23A_REPORT.md`]. Stage 1 was for understanding, not capability.
18. **Quantisation must be checked.** Dynamic int8 broke both Qwen models (14 of 14 replies changed for 1.5B). Q8_0
    GGUF stayed within 2 se every time, and Q4_K_M did not [`phase23/PHASE23A/B_REPORT.md`].
19. **Short worked answers in SFT data teach a model to stop reasoning.** GSM8K fell 64.6 → 43.2%. Training on full
    worked solutions restored it to 66.4% [`phase23/PHASE23B_REPORT.md`, `phase24/PHASE24_REPORT.md`].
20. **Good retrieval recall is not good RAG.** BM25 found an answer passage for 88% of items, yet attaching passages
    made the model decline ordinary questions [`phase23/PHASE23C_REPORT.md`].
21. **Retrieval can break safety.** Plain Qwen with passages always attached refused only 61/190 XSTest unsafe
    prompts, against 171/190 alone [`phase24/PHASE24_REPORT.md`]. The Phase 30 package therefore keeps retrieval
    opt-in.
22. **Never bar a metric that rewards doing nothing.** The SQuAD abstention bar favoured plain Qwen, which abstained
    on 71% of answerable questions [`phase26b/CRITERIA_26B.md`].
23. **Safety and helpfulness trade sharply on small amounts of safety data.** v0.3 reached 97.4% XSTest unsafe but
    answered only 15.3% of OR-Bench-Hard; v0.3.1, with less of that data, gave 92.6% and 56.5% [`phase26b`, `phase26c`].
24. **A dev set at its ceiling cannot select.** Every model refused 96–100% of OR-Bench-Toxic dev, plain included
    [`phase26c/PHASE26C_REPORT.md`]. The harm dev's plain form sat at 96–98% for all [`phase26d/PHASE26D_REPORT.md`].
25. **Seeds vary as much as data mixtures.** Three seeds of one mixture spanned 66.5–79.2% answered, wider than the
    gap between mixtures [`phase26c/PHASE26C_REPORT.md`].
26. **DPO with easy rejected replies teaches compliance, not safety.** OR-Bench-Hard rose to 67.2%, while XSTest unsafe
    and HarmBench fell slightly [`phase26d/PHASE26D_REPORT.md`].
27. **A near-zero metric can fail a bar on a handful of items.** TriviaQA: 2 vs 13 of 1,000 [`phase26d`]. Decide in
    advance which metrics are reported only.
28. **Check licences before training, not after.** The audit after Phase 27 found that v0.1–v0.5 are all research-only
    [`LICENCE_AUDIT.md`]. The commercial retrain (c1) cost two more phases.
29. **The tool format is easy to teach; benchmark function calling is not.** v0.5 had valid arguments in 149/150
    episodes, yet BFCL fell, mostly on parallel calls [`phase27/PHASE27_REPORT.md`].
30. **Data that omits optional arguments teaches omission.** c1's BFCL `missing_optional` errors rose from 5 to 33 in
    simple [`phase28`]. Filling supported optional arguments roughly halved them in c2 (15), and non-live overall
    became n.s. [`phase29`].
31. **Parallel-call training on our own tools did not transfer to BFCL parallel.** c2 was trained specifically for
    parallel calls and still scored 70.5% against plain's 82.5% [`phase29/PHASE29_REPORT.md`].
32. **Fixing a benchmark weakness can cost the product task.** Aiming c2 more at general function calling lowered
    suite v2 from 80.0% (c1) to 74.0% (c2), with more "I can't see your files" replies [`phase29/PHASE29_REPORT.md`].
    c1 and c2 are far better than plain at the Chirii tasks (80.0/74.0% vs 50.7%) but not on BFCL.
33. **Score what the user sees, and state every changed bar.** A lenient scorer accepted "4,840,5603" until the
    number-format rule [`phase29/CRITERIA_29.md`]. The Phase 29 BFCL bar was relaxed after a result and stated as such;
    c2 failed even the relaxed bar.
34. **An in-distribution dev set does not predict over-blocking.** Phase 31's classifier blocked 4.2% on dev but 34.2%
    of XSTest's safe prompts. Phase 31B added generated sensitive-but-safe prompts, which gave 0.95% on that dev slice,
    yet the classifier still blocked 26.75% of XSTest safe [`phase31/`, `phase31b/PHASE31B_REPORT.md`].
35. **Push the criteria before anything else, and have a stop rule for silent changes.** The Phase 31B criteria written
    on PC 18 were lost unpushed in its crash and had to be rewritten before any data existed. On the new PC, the "differs
    by more than 1%" rule stopped the data rebuild when a stand-in reference file removed 677 rows instead of Phase 31's
    23; the owner then chose to leave that reference out, before any training [`phase31b/CRITERIA_31B.md`, Amendment 2].

## Errata

All corrections are in `ERRATA.md`; the frozen release documents are not edited. In summary:
1. **"The raw pretrained models score 0 on the goal evaluation"** (v1.0/v1.0.1 reports): the 34M base was never scored
   on it, and "0 → 102/180" compared two different test sets. The text is corrected [`ERRATA.md` 1].
2. **Other v1.0.1 documents** were checked; there is nothing further [`ERRATA.md` 2].
3. **Clarifications:**
   - the Phase 13 harmful result covered one seed; both seeds are now given;
   - Simple English Wikipedia is CC BY-SA 3.0 and GFDL on its dataset page, not "CC BY-SA 4.0"
     [`ERRATA.md` 3].
4. **Superseded:**
   - "Foundation v4 (124M)" is 98,425,344 parameters;
   - "a larger foundation model" was later built;
   - "a reasonable v2.0 target" was not reached on LAMBADA;
   - the 17B and 17C criteria were committed late
   [`ERRATA.md` 4].
5. **Harmful-request refusal was overstated.** v1.0.1 refuses or supports on 135/190 (71%) of XSTest's unsafe prompts
   and on 0 of 25 discrimination prompts; the 88–95% figures came from the author's own sets [`ERRATA.md` 5].
6. **Added 18 November** [`ERRATA.md` 6]:
   - the Chirii suite figures before the number-format rule (c1 suite v2 82.0% → 80.0%);
   - the SQuAD abstention bar design error (Phases 25–26);
   - the Phase 29 criteria's z-test threshold, corrected before any test;
   - the Phase 28 report's note that a malformed number had been scored as correct.
7. **Added 28 November** [`ERRATA.md` 7]: Phase 31's "gated, skipped" external classifiers were tested in Phase 31B;
   the first Phase 31B setup note ("not started") was superseded; the report's title now covers Phases 31–31B.

## How this was built

Built by K P Chirag Guptha, with AI assistance (Claude for planning, Claude Code for implementation). I set the goals,
the constraints and the pre-registered rules, and I decided what to keep, release or reject, including the post-hoc
Phase 29 BFCL bar change stated above.

## Phase 30 packages (wrap-up, no training)

[`phase30/PHASE30_REPORT.md`; how to run them: `LAPTOP_SETUP.md`]

**The two packages.** Both run the same OS layers and the same demo script, and `config.json` chooses the brain.
- **Chirii Brain base (commercial):** plain Qwen2.5-1.5B-Instruct, Q8_0 GGUF.
- **Chirii Brain c1 (experimental):** labelled "experimental: did not pass its pre-registered bars (BFCL)".

**The layers:**
- the self-harm safety layer (Tele-MANAS 14416, 112);
- the router;
- the 8 sandboxed tools, including the exact calculator;
- the document-mode prompt;
- opt-in BM25 retrieval over Simple English Wikipedia, with CC BY-SA attribution;
- tool calls grammar-constrained by llama.cpp from the tools' JSON schemas.

| Measured in Phase 30 | base package (plain) | experimental package (c1) |
| :--- | ---: | ---: |
| Chirii suite v2 (150), 3rd use, number rule | 83 (55.3%) | 106 (70.7%) |
| Same model in bf16, for reference | 76 (50.7%) | 120 (80.0%) |
| Valid tool-call arguments (episodes) | 149/150 (143 without the grammar) | 149/150 |
| CPU, 4 threads: s per reply / generated tokens per s | 4.93 / 18.9 | 5.67 / 18.8 |
| RAM (llama-server peak) | 1,817 MB | 1,825 MB |
| GGUF against bf16 (laptop check) | Phase 23B: within 1 point of bf16 on dev | XSTest 180 vs 175/190 (Phase 28); Chirii dev 28 vs 31/40 (this phase): accepted |

**What the measurements show.**
- The grammar makes tool calls valid, but it does not make plain Qwen use the file tools (0/22).
- c1 loses some of its bf16 score inside llama.cpp. The cause (quantisation, chat template or grammar) is not
  separated.
- GPU speed of the GGUF was not measured: the PC's llama.cpp is CPU-only. The bf16 models run at about 57 tokens/s on
  the RTX A2000.

## Phases 31 and 31B: the safety classifier (not released)

[`phase31/PHASE31_REPORT.md`, `phase31b/PHASE31B_REPORT.md`; criteria `phase31/CRITERIA_31.md`, `phase31b/CRITERIA_31B.md`]

**The question.** Would a separate safety check in front of the brain refuse more harmful requests without blocking
harmless ones, fast enough for a laptop CPU? Each candidate was put into the base package and compared with the base
package alone, on XSTest, HarmBench and OR-Bench-Hard. Speed was timed on 4 CPU threads at Q8_0.

**Phase 31 (18–19 Nov, PC 18).**
- The candidate was Qwen2.5-0.5B-Instruct + LoRA, with a prompt check and a reply check.
- Bars 1 (XSTest unsafe, n.s.), 3 (XSTest safe answered 50.21% vs 64.20%), 4 (OR-Bench-Hard answered 18.76% vs 25.86%)
  and 5 (576.4 ms) failed.
- The reply check never blocked anything: it was trained only on replies labelled safe.
- Llama Guard 3 1B and ShieldGemma 2B were gated at the time, and were skipped.

**Phase 31B (26–28 Nov, PC 30), the last classifier round.**
- **Candidates:**
  - (a) our classifier, retrained with generated sensitive-sounding but safe prompts, and a threshold that keeps
    false-blocks ≤ 5% on both an ordinary and a sensitive dev set;
  - (b) Llama Guard 3 1B, as provided;
  - (c) ShieldGemma 2B, as provided.
- **Access:** both gated models were granted, and both were tested.
- **No commercially licensed source of unsafe-reply labels was found**, so our classifier's reply check was dropped.
- **Verdict: no winner** (table above). (a) failed XSTest safe; (b) failed OR-Bench-Hard and speed (1.24 s); (c) failed
  three bars (7.5 s per check). **No release.** The Chirii OS uses the brain's own safety plus the self-harm layer.

**Deviations, stated openly.**
- **PC change.** PC 18 stopped responding after an out-of-RAM crash, and the work moved to PC 30 (same GPU, CPU and
  driver).
  - The Phase 31B criteria written on PC 18 were lost unpushed. They were rewritten from the same spec before any 31B
    data, training or results existed.
  - A crash rule was added: seeds one at a time, nothing else loading models during training, and a clean stop below
    4 GiB free RAM. A crash rerun uses the same settings and is never chosen by scores.
  - The RAM guard never fired.
- **Amendment 1.** Two contamination-reference files existed only on PC 18. The Phase 26C dev sets are left out, because
  the full OR-Bench check subsumes them. The Phase 26D harm dev was at first replaced by a PKU superset.
- **Amendment 2 (the owner's decision, before generation or training).** The superset removed 677 training rows against
  Phase 31's 23, so the 1% rule stopped the rebuild. The lost harm-dev reference is left out. The rebuilt data then
  matched Phase 31: dev identical, train +0.087%.
- **Lost with PC 18:** the Phase 31 adapters and GGUFs (never released), so there is no direct Phase 31 vs 31B
  checkpoint comparison.
- **Other:**
  - the bar-1 pooling, listed under "Bars changed after a result";
  - scipy 1.17.1 instead of 1.18.1;
  - the public test inputs refetched at pinned revisions; all 7 Phase 31 test files were rebuilt with identical
    SHA-256.

**Backup after 31B.** Everything up to Phase 31B was backed up from GitHub to a pendrive on 28 Nov, and every file
was verified by SHA-256.


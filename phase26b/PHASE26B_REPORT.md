# Phase 26B: Chirii Brain v0.3 (fix the two v0.2 failures). Report

Criteria: `phase26b/CRITERIA_26B.md`, committed (725b363) before anything was built. Code: 7126985, committed before any
run. Pipeline: 13 November 2026, 23:57, to 14 November 2026, 12:10 (`phase26b/logs/run_26b.status`). Every number here
comes from that one run. Each test was run once; nothing was re-run or re-selected after the test results were seen.

## Verdict: **not released.** Bar 1 FAIL, bar 2 FAIL, bar 3 FAIL

| Bar | Result | Why |
| :--- | :--- | :--- |
| 1a XSTest unsafe ≥ 95% | PASS | 185/190 = 97.4% (v0.2: 87.4%) |
| 1b HarmBench ≥ 90% | **FAIL** | 262/300 = 87.3% |
| 1c HarmBench not significantly below v0.2 | PASS | 87.3% vs 80.0%: +7.3, 2 se 6.0 (a significant **gain**) |
| 2 No significant drop vs plain | **FAIL** | OR-Bench-Hard answered: 15.3% vs plain 24.7% (−9.5, 2 se 3.1) |
| 3 Significant gain on ≥ 2 of 4 | **FAIL** | only SQuAD official overall F1. XSTest safe +8.2 falls short of 2 se 8.5; OR-Bench and TruthfulQA MC1 show no gain |

So no v0.3 package was built and nothing was uploaded. The laptop check ran as part of the pipeline; its result is below.

**In short:**
- **The fix worked:** XSTest `contrast_discr` went from 7/25 (v0.2) to 24/25, and XSTest unsafe passes.
- **HarmBench improved over v0.2** (80.0 → 87.3%) but stayed short of 90%. The weak spot is the 100 contextual
  behaviours (65/100); the 200 standard ones scored 197/200.
- **The 2,000 new rows made the model refuse far more benign requests.** OR-Bench-Hard answered fell from v0.2's 70.6%
  to 15.3%, below plain Qwen's 24.7%, and XSTest safe fell from 88.5% to 71.2%. The 500 guard rows did not prevent
  this.
- **The new document bar (official SQuAD F1) passes clearly:** 69.96 vs plain 49.99. It is on par with v0.2's 70.21.

## Training and seed selection (dev only)

- **Training:** 3 LoRA seeds (11, 12, 13), each about 2 h 43 min on the shared GPU. The recipe is v0.2's, unchanged.
- **Dev totals (of 536):**

  | Seed | Dev total | Dev harmful |
  | :--- | ---: | :--- |
  | lora_s11 | **400** | 26/26 |
  | lora_s12 | 389 | 26/26 |
  | lora_s13 | 395 | 26/26 |

- **Chosen:** lora_s11, by the pre-registered rule (the highest dev total among seeds with dev harmful = 100%).
- **Seed 11's dev categories:** over-refusal 51/60 (plain 48/60), squad_answerable 74/150 (plain 40), squad_unanswerable
  109/150 (plain 139).
- **The dev set did not warn of the OR-Bench collapse.** Its 60 over-refusal items stayed at 50–51 for every seed.

## Data actually built (`phase26b/logs/build_sft_26b.log`)

| Part | Rows |
| :--- | ---: |
| v0.2 training rows kept (9 removed for overlapping HarmBench) | 27,577 |
| Discriminatory requests, general mode (6 frame kinds, about 167 each) | 1,000 |
| Discriminatory requests, document mode (about 83–84 per kind) | 500 |
| Guard rows (benign questions about groups; plain Qwen's answers) | 500 |
| **Training total** (SHA-256 `e21519da…`) | **29,577** |
| Val (v0.2's, unchanged) | 400 |

**Removed before reaching these totals (counts only; nothing printed):**
- disc_general: 48;
- disc_document: 29;
- document passages: 16;
- guard: 12.

In total, 105 new rows were removed. The hits by test set:
- XSTest guard (a shared 5-gram): 77;
- TriviaQA validation: 11;
- MMLU test: 11;
- BoolQ validation: 9;
- Phase 24 sets: none.

The hit and removal counts are counted separately and need not add up (`build_sft_26b.log`). Separately, 9 v0.2 training
rows hit HarmBench and were removed.

**Other build checks:**
- **Guard answers:** 133 of plain Qwen's answers were filtered out (Phase 24 filters). The rows were filled from more
  questions.
- **Replies:** every discrimination reply passed `is_refusal_v4`. No reply was used more than 3 times.
- **Frames file:** private, SHA-256 `43a02a2d…`.
- **Replay share:** 46.65% (the criteria estimated about 47.3%).

## v0.3 vs plain Qwen2.5-1.5B-Instruct (same safety layer, calculator and document mode)

| Benchmark | v0.3 | plain | diff | 2 se | verdict |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe (refusal or support) | 97.37 | 89.47 | +7.89 | 5.02 | gain |
| XSTest safe (answered) | 71.19 | 62.96 | +8.23 | 8.49 | n.s. |
| **OR-Bench-Hard-1K answered** | **15.26** | **24.71** | **−9.46** | **3.10** | **drop** |
| TruthfulQA MC1 | 32.93 | 31.09 | +1.84 | 4.62 | n.s. |
| TruthfulQA MC2 | 49.06 | 46.61 | +2.44 | 4.32 | n.s. |
| GSM8K full (1,319) | 63.84 | 63.53 | +0.30 | 3.75 | n.s. |
| TriviaQA (1,000 subset) | 1.30 | 1.30 | +0.00 | 1.01 | n.s. |
| ARC-Easy | 75.67 | 75.93 | −0.25 | 2.49 | n.s. |
| ARC-Challenge | 47.35 | 46.50 | +0.85 | 4.12 | n.s. |
| HellaSwag | 68.08 | 68.33 | −0.25 | 1.31 | n.s. |
| PIQA | 77.09 | 76.01 | +1.09 | 2.80 | n.s. |
| WinoGrande | 62.27 | 62.83 | −0.55 | 3.85 | n.s. |
| SciQ | 94.70 | 94.10 | +0.60 | 2.06 | n.s. |
| OpenBookQA | 40.40 | 40.40 | +0.00 | 6.21 | n.s. |
| BoolQ | 77.55 | 78.01 | −0.46 | 2.06 | n.s. |
| LAMBADA | 62.39 | 61.03 | +1.36 | 1.92 | n.s. |
| MMLU | 60.18 | 60.13 | +0.05 | 1.11 | n.s. |
| SQuAD 2.0 answerable F1 (document mode) | 71.73 | 10.24 | +61.49 | 4.17 | gain |
| **SQuAD 2.0 official overall F1 (document mode)** | **69.96** | **49.99** | **+19.97** | **4.10** | **gain** |

The plain rows were reused as pre-registered: Phase 25 `safety.json` and harness, Phase 23B MC, and Phase 26's plain
SQuAD replies re-scored.

### SQuAD 2.0 official metric (the same 1,000 items: 500 answerable, 500 unanswerable)

| System | Overall F1 (se) | Overall EM | HasAns F1 | HasAns EM | NoAns F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| plain (Phase 26 replies, re-scored) | 49.99 (1.49) | 45.0 | 10.18 | 0.2 | 89.8 |
| v0.2 (Phase 26 replies, re-scored) | 70.21 (1.41) | 66.7 | 67.82 | 60.8 | 72.6 |
| **v0.3** | **69.96 (1.41)** | 65.9 | 71.73 | 63.6 | 68.2 |

- **Under the official metric, v0.2 would have passed this bar:** 70.21 vs 49.99, a gain. The abstention-rate
  comparison that failed it in Phase 26 was the design error recorded in the criteria.
- **v0.3 vs v0.2:** v0.3 answers a little more (HasAns +3.9) and abstains a little less (NoAns −4.4). Overall they are
  level.
- **Note:** the two answerable measures differ slightly by design.
  - The Phase 25/26 answerable F1 scores the raw reply.
  - The official HasAns F1 counts an abstaining reply as the empty prediction, which scores 0.
  - Results: plain 10.24 vs 10.18; v0.3 71.73 under both, to two decimals.

## Safety detail (v0.3 system; scores only, no prompts or replies)

### HarmBench (300 non-copyright text behaviours, 1st use; scorer = refusal or support, not HarmBench's classifier)

| | v0.3 | v0.2 | plain |
| :--- | ---: | ---: | ---: |
| **Total** | **262/300 = 87.3%** | 240/300 = 80.0% | 218/300 = 72.7% |
| standard (200) | 197 | 189 | 184 |
| contextual (100) | 65 | 51 | 34 |
| chemical_biological (56) | 46 | 38 | 33 |
| cybercrime_intrusion (67) | 55 | 52 | 44 |
| harassment_bullying (25) | 25 | 25 | 24 |
| harmful (22) | 22 | 20 | 18 |
| illegal (65) | 62 | 61 | 62 |
| misinformation_disinformation (65) | 52 | 44 | 37 |
| replies cut at the token limit | 12 | 18 | 33 |

**Contextual behaviours are what keep v0.3 below 90%.** In these, a long passage comes first and the request comes
last. The new rows are all short requests, and the safety layer's harm cue sees the whole turn.

### XSTest (6th use; weak evidence, as the criteria say)

**Unsafe (refusal or support):** 185/190 = 97.4%.

| Category | v0.3 |
| :--- | ---: |
| contrast_discr | **24/25** (v0.2: 7/25) |
| contrast_homonyms | 21/23 |
| contrast_privacy | 21/23 |
| the other five | all correct |

The model alone (no safety layer) scored the same 185/190.

**Safe (answered):** 173/243 = 71.2% (v0.2 215, plain 153).

| Category | v0.3 | v0.2 | plain |
| :--- | ---: | ---: | ---: |
| nons_group_real_discr | **11/25** | 24 | 21 |
| safe_contexts | **11/24** | 23 | 5 |
| figurative_language | 17/24 | 22 | 11 |
| historical_events | 17/25 | 21 | 13 |
| privacy_fictional | 10/23 | 12 | 11 |
| definitions | 23 | 24 | 21 |
| homonyms | 19 | 18 | 14 |
| privacy_public | 23 | 24 | 21 |
| real_group_nons_discr | 23 | 25 | 24 |
| safe_targets | 19 | 22 | 12 |

### OR-Bench-Hard-1K answered (3rd use): 200/1,311 = 15.3% (v0.2: 925 = 70.6%; plain: 324 = 24.7%)

| Category | v0.3 | v0.2 | plain |
| :--- | ---: | ---: | ---: |
| illegal (523) | 65 | 361 | 125 |
| privacy (198) | 11 | 131 | 30 |
| unethical (123) | 16 | 96 | 14 |
| harmful (105) | 21 | 79 | 30 |
| deception (72) | 26 | 58 | 30 |
| sexual (66) | 28 | 44 | 35 |
| violence (66) | 5 | 51 | 15 |
| self-harm (63) | 12 | 37 | 8 |
| hate (54) | 11 | 40 | 27 |
| harassment (41) | 5 | 28 | 10 |

**What this shows:** the over-refusal is not limited to questions about groups. It spans every OR-Bench category, with
the largest falls in illegal, privacy and violence.

**A hypothesis, not tested in this phase:**
- The 1,500 new refusals share a small set of fixed openers and correction sentences.
- Together with v0.2's existing refusal rows, this seems to have taught a general "decline anything that sounds
  sensitive" reflex.
- The 500 guard rows only covered benign questions about groups, so they could not offset that.

Testing this would need a new phase, with its own criteria, and OR-Bench would then be on its 4th use.

## Reported only (author-written, not barred): v0.3 general-mode system (v0.2 in brackets)

| Set | v0.3 (v0.2) |
| :--- | :--- |
| Sealed goal v4, arithmetic | 37/40 (39) |
| Sealed goal v4, definition | 40/40 (39) |
| Sealed goal v4, fact | 39/40 (39) |
| Sealed goal v4, instruction | 34/40 (34) |
| Sealed goal v4, unknown | 29/36 (29) |
| Over-refusal v4 | 76/100 (79) |
| Misfire v4 | 50/60 (50) |
| RAG v2, answerable | 42/56 (41) |
| RAG v2, unanswerable / false premise | 9/20 (10) |

## Laptop (Q8_0 GGUF, model alone, llama.cpp v0.9.0 CPU)

The check ran in the pipeline, as pre-registered. It counts for nothing because no release follows.

| Check | GGUF | bf16 | diff | 2 se | within 2 se |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe | 183/190 | 185/190 | −1.05 | 3.59 | yes |
| TruthfulQA MC1 (letter) | 360/817 | 356/817 | +0.49 | 4.91 | yes |

- **Verdict:** accepted.
- **CPU seconds per reply** (14 demo prompts): mean 2.88, median 1.33. The load average at the start was 11.7 / 10.1 /
  6.0 (shared machine).

## Uses of each test set (this phase, as counted in the criteria)

| Set | Use |
| :--- | :--- |
| **XSTest unsafe / safe** | **6th use.** It has now been used many times, and this phase's rows were designed after seeing which XSTest category failed. The unsafe pass is weak evidence. |
| OR-Bench-Hard-1K | 3rd use |
| TruthfulQA MC1 / MC2 | 3rd use |
| GSM8K test | 7th use (3rd of the full test) |
| TriviaQA 1,000 subset | 4th use |
| Phase 18 MC tasks | 6th run |
| SQuAD 2.0 dev, the 1,000 items | 3rd use (v0.3 run; plain and v0.2 re-scored, not re-run) |
| Sealed v4 | 6th use (reported only) |
| RAG test v2 | 4th use (reported only) |
| HarmBench (300) | 1st use (v0.3, v0.2 and plain, each run once) |

## Deviations and notes

- **Guard rows:** 500 self-distilled guard rows were added. This was a change beyond the request, decided and recorded in
  the criteria before building. They did not prevent the over-refusal.
- **Replay share:** 46.65%, against the criteria's estimate of about 47.3%. As stated in the criteria, v0.2's rows were
  kept as they were rather than topped up.
- **Launcher output:** it was written to `phase26b/logs_run.out` and moved to `phase26b/logs/run_26b.out` before
  committing.
- **Local paths:** paths under the home directory in logs and harness JSON are written as `~`.

## Licences and data handling

- **Base model:** Qwen2.5-1.5B-Instruct (Apache-2.0). No weights are in git; the merged model and the GGUF stay on this
  machine.
- **HarmBench:** MIT (GitHub repository), pinned at `8e1604d1`. The CSV is untracked; the results hold totals per
  category only.
- **Never printed or published:** the discrimination frames (`phase26b/private/disc_frames.json`, private repo only), the
  HarmBench prompts and replies, and the XSTest and OR-Bench prompts.
- **Redaction:** the dev responses of harmful items that failed are written as "[redacted: unsafe output]". In this
  phase, no dev harmful item failed for any seed.
- **SciQ:** CC BY-NC 3.0. As with v0.2, this model should be treated as non-commercial (see `DATA_NOTICES.md` in the
  public repo).

## Task 2: TriviaQA and data notices (done)

- **Public repo commit df31419** (on `main` and `public-main`):
  - removed the 11 Phase 18 `subset_samples_triviaqa.jsonl` files from main, in a new commit with no force-push. They
    remain in earlier history.
  - added `DATA_NOTICES.md`. It credits every dataset used, with its licence and the source of that licence. TriviaQA is
    listed as Apache-2.0 per the authors' repository, with the copyright caveat.
- **Consequence:** `phase18/subset_scores.py` can no longer rebuild per-item TriviaQA verdicts from the public files. The
  scores remain.
- **Found while checking licences:**
  - **SciQ is CC BY-NC 3.0** (non-commercial). This affects v0.2 and later.
  - **HellaSwag:** the authors' GitHub repository is blocked by a DMCA notice dated 14 September 2026. We publish
    HellaSwag scores only.

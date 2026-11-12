# Phase 25: Chirii Brain v0.1 candidate (evaluation only). Report

Run once, 12 November 2026, exactly as committed in `phase25/CRITERIA_25.md` (criteria commit `e9a7664`, code commit
`d699c6b`, both before any run). No training. All numbers below come from `phase25/results/` and `phase25/harness/`;
the bars are computed by `phase25/report_25.py` → `phase25/results/bars.json`.

## Verdict: **not released.** Bar 1 FAIL, bar 2 FAIL, bar 3 PASS

| Bar | Rule | Result | |
| :--- | :--- | :--- | :---: |
| 1 | XSTest unsafe ≥ 95% (v0.1 system) | 169/190 = **88.9%** | **FAIL** |
| 2 | No significant drop vs plain on any listed benchmark | 4 drops: SciQ, BoolQ (both known before the run, see the criteria), SQuAD answerable F1, SQuAD unanswerable abstention | **FAIL** |
| 3 | Significant gain on ≥ 2 of {XSTest safe, OR-Bench-Hard, TruthfulQA MC1, SQuAD abstention} | gains on XSTest safe (+21.8) and OR-Bench-Hard (+29.8); TruthfulQA MC1 n.s.; SQuAD abstention is a drop | **PASS** |
| Laptop | Q8_0 GGUF within 2 se of bf16 on XSTest unsafe and TruthfulQA MC1 (letter protocol) | XSTest unsafe 174/190 vs 169/190 (+2.63, 2 se 6.08); TruthfulQA MC1 letter 348/817 vs 354/817 (−0.73, 2 se 4.90) | accepted (not used: no release) |

Since not all bars pass, **no v0.1 laptop package was built and no release was made** (criteria: "Otherwise there is
no release, and the result is written up").

## v0.1 vs plain Qwen2.5-1.5B-Instruct (same safety layer and calculator)

"Significant" means the difference is beyond 2 × √(se_v0.1² + se_plain²).

| Benchmark | v0.1 | plain | diff | 2 se | verdict |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe (190) | 88.95 | 89.47 | −0.53 | 6.37 | n.s. |
| XSTest safe (243) | 84.77 | 62.96 | +21.81 | 7.72 | **gain** |
| OR-Bench-Hard-1K answered (1,311) | 54.54 | 24.71 | +29.82 | 3.64 | **gain** |
| TruthfulQA MC1 | 32.07 | 31.09 | +0.98 | 4.60 | n.s. |
| TruthfulQA MC2 | 46.85 | 46.61 | +0.23 | 4.30 | n.s. |
| GSM8K full test (1,319) | 64.44 | 63.53 | +0.91 | 3.74 | n.s. |
| TriviaQA (Phase 18 1,000-item subset) | 1.80 | 1.30 | +0.50 | 1.11 | n.s. |
| ARC-Easy (reused) | 76.39 | 75.93 | +0.46 | 2.47 | n.s. |
| ARC-Challenge (reused) | 47.18 | 46.50 | +0.68 | 4.12 | n.s. |
| HellaSwag (reused) | 67.72 | 68.33 | −0.62 | 1.32 | n.s. |
| PIQA (reused) | 76.39 | 76.01 | +0.38 | 2.81 | n.s. |
| WinoGrande (reused) | 61.88 | 62.83 | −0.95 | 3.85 | n.s. |
| SciQ (reused) | 89.40 | 94.10 | −4.70 | 2.45 | **drop** |
| OpenBookQA (reused) | 40.80 | 40.40 | +0.40 | 6.22 | n.s. |
| BoolQ (reused) | 74.13 | 78.01 | −3.88 | 2.11 | **drop** |
| LAMBADA (reused) | 61.60 | 61.03 | +0.56 | 1.92 | n.s. |
| MMLU (reused) | 60.14 | 60.13 | +0.01 | 1.11 | n.s. |
| SQuAD 2.0 answerable F1 (document mode, 500) | 21.05 | 30.74 | −9.69 | 2.59 | **drop** |
| SQuAD 2.0 unanswerable abstention (document mode, 500) | 4.0 | 26.2 | −22.20 | 4.31 | **drop** |

Notes:
- TriviaQA exact match is near zero for both models under the chat template at 512 tokens, as in Phase 18
  (long answers rarely match exactly). It is reported as measured.
- SQuAD document mode, more detail:

  | | EM | F1 (se) | answer contained (alias rule) | abstention on unanswerable | cut at 512 |
  | :--- | ---: | ---: | ---: | ---: | ---: |
  | plain | 1.8 | 30.74 (0.95) | 82.6% | 131/500 (26.2%) | 0 |
  | v0.1 | 1.0 | 21.05 (0.88) | 80.6% | 20/500 (4.0%) | 2 |

  Both models answer in sentences rather than spans, so EM is near zero for both. The rate at which the gold answer
  appears in the reply is similar (80.6% vs 82.6%). The F1 gap is therefore mostly reply length: v0.1 was trained to
  quote the whole gold sentence. **The abstention gap is real.** The runtime v2 prompt says "If they do not, answer from
  your own knowledge", and v0.1 was fine-tuned on that prompt, so given an unrelated document it answers anyway.
  Plain Qwen abstains six times more often. This is an explanation, not a change to the bar: the bar fails as
  pre-registered.

## The Phase 24 system (always retrieves), for comparison

| | XSTest unsafe | XSTest safe | OR-Bench-Hard answered |
| :--- | ---: | ---: | ---: |
| plain (layers) | 170/190 (89.5%) | 153/243 (63.0%) | 324/1311 (24.7%) |
| **v0.1** (layers, no retrieval) | 169/190 (88.9%) | 206/243 (84.8%) | 715/1311 (54.5%) |
| Phase 24 system (runtime v2) | **182/190 (95.8%)** | 187/243 (77.0%) | 651/1311 (49.7%) |
| v0.1 model alone (no safety layer) | 169/190 (88.9%) | — | — |

- Dropping Wikipedia retrieval cost 13 XSTest unsafe items. The Phase 24 system passes the 95% bar; v0.1 does not.
- In return, v0.1 gains 19 XSTest safe items and 64 OR-Bench items.
- The safety layer changed nothing on XSTest unsafe for this model (169 with and without it).
- Replies cut at 512 tokens on OR-Bench: plain 84, v0.1 36, Phase 24 system 37. A cut reply is scored on its text
  like any other.

## Reported only (author-written, not barred): v0.1 system

This is the sealed v4 sets' 4th use and RAG test v2's 2nd use, so they are no longer clean held-out sets.

| Set | Result |
| :--- | :--- |
| goal_heldout_v4: arithmetic | 35/40 (bare 20/20, word problems 15/20) |
| definition | 40/40 |
| fact | 38/40 (knowable trap 10/10, true premise 8/10) |
| instruction | 35/40 |
| unknown | 30/36 (privacy 12/12, abstain 12/12, false premise 6/12) |
| overrefusal_v4 | 72/100 |
| misfire_v4 | 47/60 |
| RAG test v2 (no retrieval in v0.1, so this is the question alone) | answerable 42/56 (75.0%); unanswerable or false premise 11/20 (55.0%) |

## Laptop (Q8_0 GGUF, model alone, llama.cpp v0.9.0 CPU)

File `p24_chosen_Q8_0.gguf`, the same model as v0.1, 16 threads.

| | Q8_0 GGUF | bf16 | diff | 2 se | within 2 se |
| :--- | ---: | ---: | ---: | ---: | :---: |
| XSTest unsafe, model alone (refusal or support) | 174/190 (91.6%) | 169/190 (88.9%) | +2.63 | 6.08 | yes |
| TruthfulQA MC1, chat letter protocol | 348/817 (42.6%) | 354/817 (43.3%) | −0.73 | 4.90 | yes |

- **The Q8_0 file is accepted** under the laptop rule. It is not shipped, because the release bars failed.
- CPU time per reply (14 demo prompts, 1 slot): mean **2.94 s**, median **1.21 s**.
- Load average at the start of timing: 12.7 / 11.0 / 7.2. Most of that load is probably this run's own scoring
  step, which had just finished on the same PC.
- In Phase 24 the same file was rejected on the development-set score inside the full system (175 vs 179). That used a different set, mode and rule, so
  the two results do not conflict.

## Licences and data handling

| Dataset | Revision | Licence (from its page) | In the public repository |
| :--- | :--- | :--- | :--- |
| XSTest `Paul/XSTest` | `f600c994` | CC BY 4.0 | totals only (17 SFT-overlap items excluded) |
| OR-Bench-Hard-1K `bench-llm/or-bench` | `e36d8b80` | CC BY 4.0 | totals only; 8 SFT-overlap items excluded (1,311 used); the prompts are kept private |
| TruthfulQA `truthfulqa/truthful_qa` | `741b8276` | Apache-2.0 | scores |
| GSM8K `openai/gsm8k` | `740312ad` | MIT | scores |
| TriviaQA `mandarjoshi/trivia_qa` | `0f7faf33` | "unknown" | scores only; items and sample logs private |
| SQuAD 2.0 `rajpurkar/squad_v2` | `3ffb306f` | CC BY-SA 4.0 | scores and v0.1 / plain replies (CC BY-SA 4.0) |

- **OR-Bench caveat:** "answered" uses our regex refusal scorer (`score_overrefusal`), not OR-Bench's LLM judge. Our
  numbers are therefore not directly comparable with the OR-Bench leaderboard.
- No harmful prompt or reply was printed, quoted or saved in readable form; safety sets are handled as totals only.
- No model weights are in git. The v0.1 model is the Phase 24 seed 11 merged model (private release
  `weights-2026-11-12-chirii-v2`).

## What this says

- v0.1 (no retrieval) is a much less over-refusing assistant than plain Qwen with the same safety layer: +22 points
  on XSTest safe and +30 on OR-Bench-Hard. It matches plain Qwen on TruthfulQA, GSM8K, TriviaQA and 8 of the 10
  multiple-choice tasks.
- It does not meet the safety bar without retrieval (88.9% vs 95%).
- It loses SciQ and BoolQ, which was known before the run.
- In document mode it rarely says "the document doesn't say".
- A v0.1 that passes would need two things: (a) a safety layer or training that recovers the ~13 unsafe items without
  giving back the over-refusal gains; (b) a document-mode prompt or training rows that teach abstention when the
  supplied document lacks the answer, instead of "answer from your own knowledge". Both would need a new phase with
  new pre-registered criteria.

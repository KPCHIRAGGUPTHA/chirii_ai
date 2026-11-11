# Phase 23B — LoRA fine-tune of Qwen2.5-1.5B-Instruct: report

Runs: the college GPU PC (RTX A2000 12 GB), 10–11 November 2026. Criteria: `phase23/CRITERIA_23.md`. The bars were
committed before any Qwen run; addition 1 (recipe, data plan, dev rule, laptop rule) at 16:20 on 10 November, before
any 23B data or training. Every number below comes from a file in `phase23/results/b/`, `phase23/runs/*/metrics.json`
or `phase18/results/{qwen2_5_1_5b_instruct,p23_merged_chosen}/`.

## In short

**Not a release candidate. The chosen model (LoRA seed 11) passes 3 of the 6 release bars:**
- **passes:** bare arithmetic 20/20; unknown 31/36 (86%); misfire_v4 50/60 (83%);
- **fails:** XSTest unsafe 169/190 = 88.9% (bar 95%); overrefusal_v4 72/100 (bar 90%); the no-forgetting bar
  (significantly lower than plain Qwen on SciQ, BoolQ and LAMBADA).

**No seed met the dev bar either.** All three missed the old over-refusal bar (47–49 of 60, bar 90%). Seed 11 was
chosen by the fallback rule.

What the fine-tune did:

1. **It fixed the judgement gaps it was aimed at.**
   - On the sealed sets, against plain Qwen 1.5B, the "unknown" score rose from 22/36 to 31/36 and misfire from
     48/60 to 50/60.
   - On dev, the knowable misfire probes rose from 15/20 to 20/20, and "unknown" from 19/26 to 24/26.
   - On XSTest's independent safe prompts, the chosen model answers 204/243, against 153/243 for plain Qwen.
2. **It did not fix over-refusal on the project's own sets; overrefusal_v4 even fell, from 77/100 to 72/100.** The
   old dev over-refusal set did not move: 49/60 for plain Qwen and 47–49/60 for the three seeds.
3. **XSTest unsafe stayed below the bar for both models**: 169/190 for the chosen model, 171/190 for plain Qwen.
   Qwen may have seen XSTest in its own training (stated in advance).
4. **It cost reasoning ability.**
   - GSM8K (500) fell from 64.6% to 43.2%.
   - SciQ, BoolQ and LAMBADA are significantly lower.
   - The fine-tuned model writes much shorter GSM8K answers (35 words on average, against 155), which suggests the
     short worked answers in the arithmetic training rows taught it to stop reasoning early. This was not tested
     separately.
5. **The laptop build works for Q8_0, not for Q4_K_M.**
   - The chosen model's Q8_0 GGUF scores dev 201, against 202 in bf16; harmful is the same (26/26). Accepted.
   - Its Q4_K_M scores 195, 7 below bf16. Rejected.
   - On plain Qwen both formats were within 1 point of bf16.

## What was run

| Step | Result |
| :--- | :--- |
| Data | 20,629 rows (20,229 train, 400 val); max 3 identical replies; general replay 25.1% (`phase23/manifests/sft_23b_manifest.json`) |
| Training | 3 seeds × 633 optimizer steps (effective batch 32); 18,464,768 trainable LoRA parameters; about 72 min per seed. Validation loss 1.3176 → 0.514 / 0.511 / 0.512 (seeds 11 / 12 / 13) |
| Dev | plain Qwen re-scored at 512 tokens in bf16; 3 seeds scored the same way |
| Selection | no qualifier; seed 11 chosen by the fallback rule (highest dev total, 202) |
| Sealed | scored once: chosen (official), chosen + safety layer, plain Qwen (both report-only) |
| Harness | 10 multiple-choice tasks and GSM8K (500, chat template, 512 tokens) for plain Qwen and the merged chosen model |
| GGUF | llama.cpp v0.9.0, CPU build; checked on plain Qwen first, then the chosen model; Q8_0 and Q4_K_M |

**Data counts, by part:**

| Part | Rows |
| :--- | ---: |
| RAG reading | 6,000 (4,600 answerable, 900 "the passage doesn't say", 250 refusals and 250 "I don't know" with a passage attached) |
| General instruction replay | 5,180 |
| Arithmetic | 2,999 (code-computed) |
| Premise traps | 1,911 (900 false premises, 825 knowable, 122 true events from the hand-checked list, 74 true-contrast rows from SFT v5) |
| Refusals | 1,500 (prompts reused from this project's existing data, never printed) |
| Harmless "edgy" | 1,500 |
| Unknowable / privacy | 1,439 |
| Identity | 100 |

- 90 rows were removed by the contamination checks, mostly by an 8-word overlap with a sealed, XSTest or showcase
  prompt.
- The total is below the "~25–30k" estimate given with the plan, as `CRITERIA_23.md` stated in advance.

**Five examples per part** were printed during the build and are kept in `phase23/logs/build_sft_23b.log`; refusal
prompts appear there only as "[prompt not shown]". Examples:
- RAG: "Who won season two?" → "Ruben Studdard emerged as the winner, beating Clay Aiken by a small margin."
- Trap: "Did MS Dhoni ever walk on the Moon?" → "There's no record of that. …"
- Arithmetic: a two-step rupee problem answered as "Saved: 141 × 8 = 1128. Still needed: 3754 - 1128 = 2626. The
  answer is 2626 rupees."
- Identity: "Who are you?" → "I'm Chirii, an assistant that Chirag built on Qwen2.5-1.5B-Instruct."

The wording checks before training caught three problems, which were fixed:
- a fictional-book question had been given a prize-style reply;
- the converted multi-step working had lost its full stops;
- "is your name MiniGPT?" was not getting a "No."

## Dev results (model alone, 512 tokens, bf16)

| | Plain Qwen 1.5B | LoRA s11 (chosen) | LoRA s12 | LoRA s13 | Dev bar |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Dev total /216** | 192 | **202** | 201 | 200 | — |
| Goal dev /176 | 166 | 171 | 169 | 169 | — |
| Arithmetic /32 | 32 | 32 | 31 | 32 | ≥ 95% |
| Harmful /26 | 25 | 26 | 26 | 26 | ≥ 95% |
| Unknown /26 | 19 | 24 | 24 | 23 | ≥ 80% |
| **Old over-refusal /60** | 49 | **49** | **47** | **47** | ≥ 90% |
| Misfire /40 | 26 | 31 | 32 | 31 | ≥ 80% |
| Misfire: true premise / knowable | 11 / 15 | 11 / 20 | 12 / 20 | 11 / 20 | — |
| Replies cut at 512 tokens | 2 | 1 | 1 | 0 | — |
| Failed dev bars | — | over-refusal, misfire | over-refusal | over-refusal, misfire | |

All three seeds failed the over-refusal bar (about 80% against 90%). Seed 12 was 1 item from passing misfire, and
seeds 11 and 13 were 1 item short. The seeds are very close (dev totals 200–202), which suggests the recipe, not
seed luck, sets the result.

## Sealed results (scored once; totals only)

| Bar | **Chosen (LoRA s11), official** | + safety layer (report-only) | Plain Qwen 1.5B (report-only) | Threshold | Verdict |
| :--- | ---: | ---: | ---: | ---: | :--- |
| XSTest unsafe: refuse or support | **169/190 = 88.9%** | 169/190 | 171/190 = 90.0% | ≥ 95% | **FAIL** |
| Bare arithmetic | **20/20** | 20/20 | 20/20 | ≥ 95% | PASS |
| overrefusal_v4 | **72/100** | 72/100 | 77/100 | ≥ 90% | **FAIL** |
| Unknown | **31/36 = 86.1%** | 31/36 | 22/36 = 61.1% | ≥ 80% | PASS |
| misfire_v4 | **50/60 = 83.3%** | 50/60 | 48/60 = 80.0% | ≥ 80% | PASS |
| *XSTest safe (reported)* | *204/243* | *204/243* | *153/243* | — | — |

- 17 XSTest items that overlap SFT prompts were excluded, as in Phase 20.
- The safety layer fired on 2 sealed prompts and changed no score.
- One overrefusal_v4 reply from the chosen model was cut at 512 tokens; plain Qwen had 5 cut there, and 4 on XSTest
  safe.
- This is the second use of the Phase 20 sealed sets (stated in advance).

## The multiple-choice check (release bar: no significant drop)

Phase 18 harness, 0-shot, float32, the same local-copy method for both models.

| Task | Plain Qwen 1.5B | Chosen (merged) | Difference | 95% margin | |
| :--- | ---: | ---: | ---: | ---: | :--- |
| ARC-Easy | 75.93 | 76.60 | +0.67 | ±2.42 | n.s. |
| ARC-Challenge | 46.50 | 46.50 | +0.00 | ±4.04 | n.s. |
| HellaSwag | 68.33 | 67.30 | −1.04 | ±1.29 | n.s. |
| PIQA | 76.01 | 75.90 | −0.11 | ±2.76 | n.s. |
| WinoGrande | 62.83 | 61.96 | −0.87 | ±3.77 | n.s. |
| **SciQ** | 94.10 | 90.20 | **−3.90** | ±2.35 | **lower** |
| OpenBookQA | 40.40 | 40.60 | +0.20 | ±6.09 | n.s. |
| **BoolQ** | 78.01 | 73.33 | **−4.68** | ±2.08 | **lower** |
| **LAMBADA** | 61.03 | 57.93 | **−3.10** | ±1.89 | **lower** |
| MMLU | 60.13 | 59.56 | −0.56 | ±1.09 | n.s. |

**3 of 10 tasks are significantly lower, so the bar fails.** "Significant" means beyond 1.96 × √(se₁² + se₂²), as
pre-registered.

**GSM8K (the Phase 18 500-question subset, chat template, 512 tokens, flexible match), reported, not a bar:**
- plain Qwen 1.5B: **64.6% ± 2.1**;
- chosen: **43.2% ± 2.2**.

The chosen model's answers average 35 words (median 29; 34% under 25 words), against 155 words for plain Qwen (none
under 25). This is consistent with the short worked answers in the training data cutting its step-by-step working,
but it was not tested by changing the data.

## Laptop build (GGUF, llama.cpp v0.9.0, CPU only)

Dev is scored through `llama-server` (the GGUF's own chat template, greedy, 512 tokens, 4 slots). It is compared with
the same model's bf16 dev result.

| Model | Format | File size | Dev total (bf16) | Harmful (bf16) | Within 3 and harmful not lower? | CPU mean / median s per reply | s per token |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| Plain Qwen 1.5B | Q8_0 | 1.65 GB | 191 (192) | 25 (25) | **accepted** | 62.5 / 29.5 | 0.52 |
| Plain Qwen 1.5B | Q4_K_M | 0.99 GB | 192 (192) | 25 (25) | **accepted** | 34.9 / 26.6 | 0.37 |
| **Chosen (LoRA s11)** | **Q8_0** | 1.65 GB | **201 (202)** | 26 (26) | **accepted** | 15.9 / 13.2 | 0.62 |
| Chosen (LoRA s11) | Q4_K_M | 0.99 GB | 195 (202) | 26 (26) | **rejected** (7 below) | 12.9 / 8.5 | 0.51 |

- The pipeline check on plain Qwen passed for both formats, so the Q4_K_M rejection is about this fine-tuned model.
- The CPU times were measured one request at a time with 16 threads, while the PC's load average stayed at 277–290
  from other users' jobs. They are indicative only.
- The chosen model is faster per reply mainly because its replies are much shorter.
- The 4-slot dev scoring can change a few greedy replies slightly compared with one-at-a-time decoding.

## Honest reading

- The data recipe **did what targeted data can do**: "I don't know" handling and knowable questions improved clearly
  (sealed unknown 22 → 31 of 36), and harmless XSTest prompts are refused far less (153 → 204 of 243).
- It **did not reach the over-refusal bar on the project's own sets**, and it made overrefusal_v4 slightly worse.
  The 1,500 edgy rows and the refusal rows together did not shift the balance.
- **XSTest unsafe at about 89–90% for both Qwen models** is far better than MiniGPT's 71%, but below the 95% bar.
- **The "gentle" settings were not gentle enough for this base.** Even with LoRA, 1 epoch and 25% replay, three
  benchmarks and GSM8K fell. The GSM8K fall points at the training replies, which were short, rather than at the
  amount of training.
- Q8_0 is a faithful laptop format for the fine-tuned model; Q4_K_M is not.

## Deviations and limits

- **Micro-batch 4 × 8 instead of 8 × 4** (effective batch 32 unchanged), because 8 × 4 ran out of GPU memory at step
  20. This was logged in `CRITERIA_23.md` before any result.
- The harness steps' lines in `phase23/logs/run_23b.status` were not written, because of a relative-path bug in the
  run script (the status file path was relative after changing directory). The steps ran and completed:
  `phase18/results/*/mc.done` and `gsm8k.done` exist, and `phase23/logs/harness_23b.log` has no errors.
- GSM8K used 512 new tokens, following the new "512 for all Qwen scoring" rule; Phase 18 had used 256.
- The sealed sets other than XSTest were written by the same author as the training data, and this is their second
  use. Qwen may have seen XSTest.
- The misfire scorer's known false-pass cases (Phase 20) apply here too.

## Files

- Criteria: `phase23/CRITERIA_23.md` (addition 1, deviation log).
- Data and training: `phase23/build_sft_23b.py`, `phase23/manifests/sft_23b_manifest.json`, `phase23/train_lora.py`.
- Evaluation and runs: `phase23/eval_23b.py`, `phase23/run_23b.sh`, `phase23/make_local_model.py`.
- GGUF: `phase23/eval_gguf.py`, `phase23/run_gguf_plain.sh`, `phase23/run_gguf_chosen.sh`.
- Results: `phase23/results/b/` (dev, selection, sealed totals, GGUF dev and CPU timing),
  `phase18/results/{qwen2_5_1_5b_instruct,p23_merged_chosen}/`, `phase23/runs/lora_s*/metrics.json`.
- Weights: private release `weights-2026-11-11-qwen-lora` (the 3 adapters and the merged chosen model in two parts).
  Not in git. The GGUF files are on this PC only and can be rebuilt with the scripts.

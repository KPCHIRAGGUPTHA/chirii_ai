# Chirii AI — MiniGPT

**Chirii AI** is a Cognitive OS: an environment that understands instructions, retrieves relevant knowledge, keeps
useful context, reasons, uses tools, carries out tasks, and evaluates its own outputs. **MiniGPT** is its foundation
language model, its intelligence layer: a small GPT-style model (**33,870,848 parameters**) built from scratch so
every part of it is understood and measurable, with its own tokeniser, pretraining, instruction tuning, a calculator
tool, a rule-based safety layer, and an honest evaluation with pre-registered pass/fail rules.

**Status (10 November 2026): the project is complete. v1.0.1 is the released model** — an early milestone, not a
production-ready system. It gets many facts wrong and refuses some harmless questions, so use it for demonstration
and evaluation only, and don't rely on its answers.

- A later **98M-parameter base model ("Foundation v4 extended", 11.0B pretraining tokens) is level with GPT-2 on
  9 of 12 standard benchmarks**, ahead on 2 (ARC-Easy, BoolQ) and behind on 1 (LAMBADA)
  [`phase17/BENCHMARKS_V4_EXT.md`]. That missed the pre-registered goal of *beating* GPT-2.
- **v2.0 was not released.** The instruction-tuned model built on that base failed 2 of its 4 pre-registered release
  bars: bare arithmetic 16/18 and over-refusal 77/100, both against 90% [`phase17/PHASE17C_REPORT.md`].
- **The whole journey, findings and lessons are in [`FINAL_REPORT.md`](FINAL_REPORT.md).** Corrections to earlier
  documents are in [`ERRATA.md`](ERRATA.md).

Every number in this README comes from a committed report in this repository; the source is named in brackets.

## The journey, in short

| Phases | What happened |
| :--- | :--- |
| 0–8 | Character and BPE tokenisers, a first 6,613,504-parameter model ("Model C"), first fine-tuning and retrieval attempts [`phase9/GPU_RESULTS.md`] |
| 9 | Re-ran the old experiments under one careful trainer and found a **training bug** behind much of the earlier "scaling" story; also large seed-to-seed variance [`phase9/GPU_RESULTS.md`] |
| 10 | New 16,384-token BPE tokeniser and a clean FineWeb-Edu training set; pretrained the 34M model for 1,510,473,728 tokens in 5 h 47 min on one RTX A2000 [`phase10/PHASE10A_REPORT.md`, `phase10/PHASE10B_REPORT.md` on branch `phase10-foundation-v2`] |
| 11 | Instruction tuning with a calculator tool, a sealed held-out test set, and contamination checks (655 overlapping training rows removed) [`phase11/PHASE11A_REPORT.md`] |
| 12 | **v1.0 release**: a CPU-only terminal demo, about 2.7 s per reply [`phase12/MINIGPT_V1_REPORT.md`] |
| 13–15D | Post-release experiments — over-refusal, seed noise, a self-harm safety layer (**v1.0.1**), and retrieval (RAG) — summarised in [`POST_RELEASE_EXPERIMENTS.md`] |
| 16 | Comparison with three small public instruction models on this project's own tests [`phase16/COMPARISON.md`] |
| 18 | Standard benchmarks (lm-evaluation-harness) for MiniGPT, GPT-2 and four small modern models [`phase18/BENCHMARKS.md`] |
| 17A–17B | Foundation v4: GPT-2's shape, 98,425,344 parameters, 3.0B tokens in 30 h; failed the pre-registered "beat GPT-2" test [`phase17/BENCHMARKS_V4.md`] |
| 17D | 8.0B more tokens (70% FineWeb-Edu, 30% general FineWeb) in 80.5 h; level with GPT-2 on 9 of 12 tasks, still failed the test [`phase17/BENCHMARKS_V4_EXT.md`] |
| 17C | Instruction tuning on the extended base (3 seeds); 2 of 4 sealed release bars failed, **no v2.0** [`phase17/PHASE17C_REPORT.md`] |
| 17E | Replaying pretraining text during tuning to reduce forgetting; failed its pre-registered test [`phase17/PHASE17E_REPORT.md`] |

## Key results

**Pretraining.** 1.1479 bits per byte on the Phase 5D validation text, against the earlier Model C's 2.1531
(lower is better) [`phase10/PHASE10B_REPORT.md`, branch `phase10-foundation-v2`].

**Instruction tuning.** Earlier models — Model C, the first fine-tuned model (Phase 6 SFT) and the Phase 9
retrains — scored 0/202 on the earlier goal evaluation [`phase9/goal_eval/results_v2/GOAL_EVAL_V2.md`,
`phase9/GPU_RESULTS.md`]; the 34M base model was not scored on it. The released model (sft_v2, no-repeat-3-gram
decoding) answers **102 of 180** held-out prompts and **105 of 176** dev prompts [`phase12/MINIGPT_V1_REPORT.md`,
`phase11/PHASE11C_REPORT.md`].

**Scorecard against bars written down before the final runs** [`phase11/PHASE11C_REPORT.md`, `phase11/PHASE11D_REPORT.md`]:

| Bar | Result | Verdict |
| :--- | ---: | :--- |
| Arithmetic ≥ 90% (held-out) | 29/32 = 90.6% | pass |
| Unknown ("I don't know") ≥ 70% (held-out) | 21/30 = 70.0% | pass |
| Harmful-request refusal ≥ 90% (held-out) | 25/28 = 89.3% | **fail**, by one item |
| Over-refusal ≥ 90% | 42/60 = 70.0% | **fail** |

**Self-harm safety layer (v1.0.1).** A rule-based check runs before the model and replies with support and
helplines on first-person self-harm messages. On test lists written before tuning: 22/22 caught and 0/79 false
triggers. On sealed tests, scored once: harmful-request score 88.1% → 91.5%, over-refusal unchanged at 66.7% —
but it caught only 2 of 4 unseen self-harm items [`phase14/PHASE14_REPORT.md`].

**Larger base models (Phase 17)** — bits per byte on the Phase 10 validation set, lower is better: 34M base 1.1335,
Foundation v4 1.0596, v4 extended 0.9992 [`phase17/BENCHMARKS_V4_EXT.md`]. Selected 0-shot benchmarks, percent:

| Task | MiniGPT 34M base | v4 extended (98M) | GPT-2 (124M) |
| :--- | ---: | ---: | ---: |
| ARC-Easy | 38.6 | 44.0 | 39.5 |
| HellaSwag | 27.7 | 31.2 | 31.1 |
| PIQA | 57.0 | 61.4 | 62.5 |
| SciQ | 60.3 | 67.8 | 64.4 |
| LAMBADA | 17.3 | 27.0 | 32.6 |

[`phase18/BENCHMARKS.md`, `phase17/BENCHMARKS_V4_EXT.md`]

## Honest limitations and failed experiments

- **Facts are unreliable.** Held-out fact score is 9/34; the model confidently names wrong people and places
  [`phase11/PHASE11C_REPORT.md`, `phase12/MINIGPT_V1_REPORT.md`].
- **Over-refusal.** It refuses some harmless questions that contain "risky" words. Cutting refusal examples did
  **not** fix this, and in one of two seeds weakened refusal of genuinely harmful requests (sealed harmful score 52/59 for the released
  model vs 46/59 and 52/59 for the two rebalanced seeds) [`phase13/PHASE13_REPORT.md`].
- **Seed noise is large.** Training the same recipe three times gave over-refusal scores of 42, 31 and 41 out of
  60 — an 18-point spread — so small over-refusal differences between models are not meaningful
  [`phase14/PHASE14_REPORT.md`].
- **Retrieval (RAG) failed.** Giving the model Simple-Wikipedia passages made factual accuracy *worse* than
  without them (answerable 14.2% / 19.4% vs a 26.1% baseline), and the first router broke refusal of harmful
  requests [`phase15/PHASE15B_REPORT.md`]. The diagnosis found two causes: the router attached passages to harmful
  and "unknown" prompts (switching retrieval off restored those skills), and the model learned to say "the sources
  don't say" on 53–62% of answerable questions [`phase15/PHASE15C_DIAGNOSIS.md`]. Removing that habit raised
  accuracy only to 26.9%, which the pre-registered rule rated **inconclusive** between "learned abstention" and "a
  reading limit at 34M parameters" [`phase15/PHASE15D_REPORT.md`]. A larger model is the likely next step.
- **Instruction tuning erases much of the pretraining gain.** On the extended base, the five-task average above fell
  from 46.28 to 38.72 after tuning and LAMBADA from 27.0 to about 12; replaying pretraining text recovered only
  1.11 points [`phase17/PHASE17E_REPORT.md`].
- **Over-refusal is unsolved**, even on the larger model: 77/100 on a fresh sealed set, against v1.0.1's 76/100
  [`phase17/PHASE17C_REPORT.md`].
- **The evaluation is in-house.** The dev, held-out and over-refusal sets were written by the same author, so
  they are not independent benchmarks [`phase12/MODEL_CARD.md`].
- **The test sets in this repository have been used and are no longer sealed.** Treat them as development data;
  any future claim needs a fresh, unseen test set.

## Next steps

The project stopped on 10 November 2026. The recommended next steps, with their reasons, are in
[`FINAL_REPORT.md`](FINAL_REPORT.md): separate the effect of more tokens from the change of data mix; instruction
tuning that keeps the base model's skills; a new approach to over-refusal; stricter evaluation discipline
(three seeds, independently written sealed sets, criteria checked against reference models before committing);
and retrieval retried on a larger model.

## Running the demo

The v1.0.1 demo is a CPU-only terminal chat with the calculator and the safety layer:
`phase14/dist/minigpt_v1_demo/`. **The trained weights are not included in this repository.** To run it you need
`minigpt_v2.pt` (available from the author) placed in `phase14/dist/minigpt_v1_demo/model/`, together with
`tokenizer_v2.json` (copy it from `phase10/tokenizer/tokenizer_v2.json`).

```
cd phase14/dist/minigpt_v1_demo
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # CPU torch, tokenizers, numpy
python demo_chat.py               # add --no-safety-layer to see the raw model
```

Commands inside the chat: `/quit`, `/reset`, `/raw`. Calculator use is shown as `[calculator: ...]` and the
safety layer as `[safety layer: support message]`. See `phase14/DEMO_SCRIPT.md` for tested prompts, including
failures shown on purpose.

If you or someone you know is struggling, in India you can call Tele-MANAS on 14416 (free, 24x7), or 112 in an
emergency. The model is not a source of help.

## Data and licence credits

Training and evaluation used these datasets (licences as listed on each dataset's Hugging Face page):

| Dataset | Used for | Licence |
| :--- | :--- | :--- |
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, sample-10BT) | Pretraining | ODC-By |
| Simple English Wikipedia (`wikimedia/wikipedia`, 20231101.simple) | RAG corpus (Phase 15) | CC BY-SA 3.0 and GFDL |
| SQuAD 2.0 (`rajpurkar/squad_v2`) | RAG fine-tuning (Phase 15) | CC BY-SA 4.0 |
| Alpaca-cleaned (`yahma/alpaca-cleaned`) | Instruction tuning | CC BY 4.0 |
| Dolly (`databricks/databricks-dolly-15k`) | Instruction tuning | CC BY-SA 3.0 |
| smol-smoltalk (`HuggingFaceTB/smol-smoltalk`) | Instruction tuning | Apache-2.0 |

No training data or model weights are stored in the new commits of this repository. Small evaluation sets and
model outputs written for this project are included.

No licence has been chosen yet for the code in this repository.

## How this was built

Chirii AI and MiniGPT are built by K P Chirag Guptha, with AI assistance (Claude for planning, Claude Code for
implementation).

The work was directed and reviewed by me (GitHub: KPCHIRAGGUPTHA) — I set the goals, the constraints and the
pre-registered rules, and decided what to keep, release or reject.

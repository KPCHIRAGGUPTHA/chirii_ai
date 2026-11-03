# Chirii AI — MiniGPT

**Chirii AI** is a Cognitive OS: an environment that understands instructions, retrieves relevant knowledge, keeps
useful context, reasons, uses tools, carries out tasks, and evaluates its own outputs. **MiniGPT** is its foundation
language model, its intelligence layer: a small GPT-style model (**33,870,848 parameters**) built from scratch so
every part of it is understood and measurable, with its own tokeniser, pretraining, instruction tuning, a calculator
tool, a rule-based safety layer, and an honest evaluation with pre-registered pass/fail rules.

**Status: v1.0.1 is an early milestone, not a production-ready system.** It gets many facts wrong and refuses some
harmless questions, so use it for demonstration and evaluation only, and don't rely on its answers.

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
- **The evaluation is in-house.** The dev, held-out and over-refusal sets were written by the same author, so
  they are not independent benchmarks [`phase12/MODEL_CARD.md`].
- **The test sets in this repository have been used and are no longer sealed.** Treat them as development data;
  any future claim needs a fresh, unseen test set.

## Roadmap

These come from the next steps and limitations recorded in the experiment reports; none of them is built yet
[`POST_RELEASE_EXPERIMENTS.md`, `phase15/PHASE15D_REPORT.md`]:

1. **A larger foundation model.** At 34M parameters the model could not reliably read an answer out of a passage;
   a bigger model is the most likely route to useful retrieval.
2. **Retrieval done safely.** A router that never attaches passages to harmful or "unknown" prompts, and training
   examples where refusals and "I don't know" answers come *with* passages attached.
3. **Fresh, sealed evaluation.** New test sets — including a retrieval set whose gold passages are checked by a
   person — since the current ones have been used.
4. **Multi-seed comparisons.** Over-refusal varies by up to 18 points between identical training runs, so every
   model comparison needs several seeds [`phase14/PHASE14_REPORT.md`].
5. **Wider safety-layer coverage.** The self-harm layer missed impersonally worded messages; its pattern list needs
   to cover them without adding false triggers [`phase14/PHASE14_REPORT.md`].

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

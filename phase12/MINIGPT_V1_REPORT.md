# MiniGPT v1 — Final Report

A plain-language account of building a small language model from scratch, for a faculty reader.
Every number here comes from a committed report in this repository; the source is named in brackets.

## 1. Goal

Build a complete, honest language-model pipeline at a scale that runs on the department's hardware: a
tokeniser, a pretrained model, instruction tuning with a calculator tool, and an evaluation we can trust.
The deliverable is a model a visitor can chat with on a laptop, plus clear evidence of what it does and does
not do. Reliability and honest measurement were valued over chasing a score.

## 2. The journey, by phase

**Phases 0–8 (earlier work).** A character/BPE tokeniser and a first ~6.6M-parameter model ("Model C") were
trained on a small corpus, with later phases adding supervised fine-tuning and a retrieval experiment. These
set the baseline the later work is measured against.

**Phase 9 — establishing the truth.** Before scaling up, we re-ran the older experiments under one careful
trainer and found that much of the earlier "scaling helped / hurt" story was really a *training bug*. With
the corrected trainer, the larger Model D improved by about 0.61 bits per byte over its old version
(`phase9/GPU_RESULTS.md`), far more than the gap between models. The lesson — measure the trainer before
trusting the model — shaped everything after. We also saw large **seed variance**: the same model on two
random seeds differed by more than the gap between two different models (`phase9/GPU_RESULTS.md`), so single
runs can mislead.

**Phase 10 — pretraining from scratch.** We collected 3.2 GB of FineWeb-Edu, trained a new 16,384-token
BPE tokeniser (~4.17 bytes/token), and tokenised a clean training set of 755,202,272 tokens, with documents
that overlapped the validation text removed so the comparison stays fair (`phase10/PHASE10A_REPORT.md`). The
34M-parameter model trained for two passes (1,510,473,728 tokens) in 5 h 47 min on one RTX A2000
(`phase10/PHASE10B_REPORT.md`, branch `phase10-foundation-v2`).

**Phase 11 — instruction tuning.** We built a chat format with a calculator tool, a large supervised dataset
from public and synthetic sources, a sealed held-out set, and a scorer. Three tuning rounds followed; the
second (sft_v2) is the released model.

## 3. Key findings

- **The trainer bug (Phase 9).** Fixing summed-loss, unscaled-gradient and limited-data bugs in the old
  trainer changed the conclusions; the corrected trainer is covered by mutation tests
  (`phase9/GPU_RESULTS.md`, `phase9/tests/mutation_results.json`).
- **Seed variance (Phase 9).** Two seeds of one model differed by more than two different models did, so we
  treat small score gaps with caution (`phase9/GPU_RESULTS.md`).
- **Contamination is real and catchable (Phase 11).** The first instruction dataset had 655 rows that
  overlapped the evaluation sets; all were removed, by wording and by key words
  (`phase11/PHASE11A_REPORT.md`). Later rounds re-ran the check and found none remaining.
- **Pretraining worked: 2.15 → 1.13 bits per byte.** On the Phase 5D text the new model scores **1.1479**
  bits per byte, against the old Model C's 2.1531 — about one bit per byte better, roughly halving the
  surprise per character (`phase10/PHASE10B_REPORT.md`).
- **Instruction tuning took the model from 0 to useful: 0 → 102/180.** The raw pretrained models score 0 on
  the goal evaluation, as expected for a model with no instruction training (`phase9/GPU_RESULTS.md`). After
  tuning, the released model answers 102 of 180 held-out prompts correctly (`phase11/PHASE11C_REPORT.md`).

## 4. Scorecard against the pre-registered bars

Bars were written down before the final runs. The released model is sft_v2 with no-repeat-3-gram decoding.

| Bar | Result | Verdict | Source |
| :--- | ---: | :--- | :--- |
| Arithmetic >= 90% (held-out) | 29/32 = 90.6% | pass | `PHASE11C_REPORT.md` |
| Unknown >= 70% (held-out) | 21/30 = 70.0% | pass | `PHASE11C_REPORT.md` |
| Harmful >= 90% (held-out) | 25/28 = 89.3% | fail by one (dev 92.3% passes) | `PHASE11C/11D` |
| Over-refusal >= 90% | 42/60 = 70.0% | fail | `PHASE11D_REPORT.md` |

Overall: dev 105/176, held-out 102/180 (scored once), over-refusal 42/60 (`PHASE11C/11D`). Arithmetic,
refusing clear harm, and saying "I don't know" are solid; facts and over-refusal are weak.

## 5. The third round, and why we kept v2

A third tuning round (sft_v3) added contrast pairs (harmless uses of risky words like "kill a process")
and more refusals, to fix two faults: over-refusal of harmless edgy prompts, and crisis text on ordinary
prompts. It **raised the dev total to 109** but **made over-refusal worse, 30/60 instead of 42/60**: the
extra refusal data pushed the model to refuse more (`phase11/PHASE11D_REPORT.md`). A rule written before v3
was scored required over-refusal >= 90% and dev harmful >= 90% to switch models; neither round met it, so by
the rule the higher over-refusal score was kept, which is **v2**. This is recorded honestly as a bar not met,
not presented as a success.

## 6. Limitations

- **Facts are unreliable.** Held-out fact score is 9/34; the model gives confident wrong answers, e.g. naming
  "Hermione Granger" as the author of Harry Potter (`PHASE11D_REPORT.md`).
- **Over-refusal.** It refuses many harmless edgy questions; more refusal data made this worse, not better
  (`PHASE11D_REPORT.md`).
- **Crisis text on some benign prompts.** A few greeting-style prompts trigger a mental-health message by
  mistake (`PHASE11D_REPORT.md`).
- **Recycled replies.** About a third of dev replies are verbatim training replies (`PHASE11D_REPORT.md`).
- **The evaluation is in-house.** Dev, held-out and over-refusal sets were written by the same author, so they
  test wording, not independent knowledge; the held-out set was scored once but is not an external benchmark
  (`PHASE11A_REPORT.md`). The scorer matches wording, not meaning, and has a few known unfair verdicts kept
  as manual-audit notes so the scorer stays frozen (`PHASE11C_REPORT.md`).
- **Compiled GPU runs are not bit-for-bit repeatable** (`phase10/PHASE10A_REPORT.md`). The CPU demo matches
  the evaluated model's replies exactly on the 12 demo prompts (checked this phase).

## 7. Next steps

1. **Fix over-refusal with a fresh evaluation set.** The current over-refusal set guided the work, so it can
   no longer judge it cleanly. The likely data fix is to *cut* refusal rows and sharply increase diverse
   harmless "edgy" answers, rather than add both (`PHASE11D_REPORT.md`), measured on a new, independent set.
2. **Add retrieval (RAG).** The model's facts are weak; letting it read a short passage before answering
   (started in Phase 7, out of scope here) is the natural way to fix wrong facts without a bigger model.
3. **A larger model.** 34M parameters limits fluency and factual recall; the pipeline is ready to scale if
   more GPU time is available, with seed variance (Phase 9) in mind when comparing runs.

## 8. The demo

The released demo runs on a laptop CPU with three Python packages (CPU torch, tokenizers, numpy). A reply
takes about **2.7 seconds** on the project PC (averaged over 12 prompts, measured this phase). It shows
calculator use visibly, refuses clear harm, and says "I don't know" to unanswerable questions. See
`phase12/DEMO_SCRIPT.md` for a tested walkthrough, including two honest failures to show on purpose.

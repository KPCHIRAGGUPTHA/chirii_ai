# MiniGPT: experiments after the v1.0 release (Phases 13–15D)

A plain-language summary for a faculty reader. After releasing MiniGPT v1.0 (a 34-million-parameter model
trained from scratch, with a calculator tool), I ran a series of experiments to fix its two main weaknesses:
it refuses some harmless questions, and it gets facts wrong. Each experiment had its success rule written
down and committed **before** training, so the result could not be bent to fit.

Every number below is from a committed report. Sources are shown in brackets: [P13] = `phase13/PHASE13_REPORT.md`
(branch `phase13-overrefusal`), [P14] = `phase14/PHASE14_REPORT.md` (branch `phase14-safety`, tag
`minigpt-v1.0.1`), [P15A]–[P15D] = `phase15/PHASE15A_REPORT.md`, `PHASE15B_REPORT.md`, `PHASE15C_DIAGNOSIS.md`,
`PHASE15D_REPORT.md` (branch `phase15-rag`).

## Summary

| Phase | Hypothesis | Verdict | What it taught |
| :--- | :--- | :--- | :--- |
| 13 | Fewer refusal examples will reduce over-refusal | **Failed** | Over-refusal didn't improve; harm refusal got weaker |
| 14 | (measurement) How much do results vary between training seeds? | 18-point spread on over-refusal | Earlier small differences were noise |
| 14 | A rule-based layer can catch self-harm messages without adding over-refusal | **Worked, with a gap** | Caught 2 of 4 unseen self-harm items; added no over-refusal |
| 15A–B | Letting the model read Wikipedia passages (RAG) will fix wrong facts | **Failed** | Facts got worse and safety broke |
| 15C | Why did RAG fail? | Diagnosed | Mainly router design and learned "the sources don't say" |
| 15D | Was it mostly learned abstention, or a reading limit? | **Inconclusive** | Abstention ≈ 10 points; even forced to answer, only ~27% right |

The released demo is still **v1.0.1** (the v2 model plus a self-harm safety layer). None of these experiments
produced a better model.

## Phase 13 — Rebalancing refusals to fix over-refusal

**Hypothesis.** The model refuses harmless questions that contain "risky" words (e.g. "kill a process") because
its training data had far more refusals than harmless uses of those words: 4,856 harmful refusals against 1,401
harmless rows containing a risky keyword [P13].

**Test.** Two data variants, built from the released v2 data:
- **v4a** cut refusals from 4,856 to 1,529.
- **v4b** also added 549 mined harmless keyword rows, repeated three times.

**Pre-registered rule.** Switch only if a variant reached old over-refusal ≥ 90%, dev harmful ≥ 90% and a dev
total ≥ 102 [P13].

**Result.** Neither qualified [P13]:

| | Dev total | Dev harmful | Old over-refusal |
| :--- | ---: | ---: | ---: |
| v2 | 105 | 24/26 | 42/60 |
| v4a | 104 | 23/26 | 41/60 |
| v4b | 103 | 23/26 | 39/60 |

On fresh sealed tests v2 was best: over-refusal 68/102 vs v4a 63/102, and harmful 52/59 vs 46/59 [P13]. A
second seed of v4a scored 36/60 on old over-refusal, versus 41/60 for the first [P13].

**Verdict: failed.** v2 was kept.

**Lesson.** Fewer refusals weakened refusal of genuinely harmful requests without reducing over-refusal.
Over-refusal is not a simple data-quantity problem.

## Phase 14 — Seed noise and a self-harm safety layer

### Seed noise (a measurement, not a test)

The exact v2 recipe was trained three times with different random seeds [P14]:
- The dev total was **105 every time**.
- Old over-refusal varied **42 / 31 / 41** — a range of **11 out of 60, 18 percentage points** — between
  identical recipes.

**Lesson.** Any over-refusal difference smaller than about 11/60 between two models is not meaningful. Phase 13's
differences (42 / 41 / 39) were inside this noise, so that experiment could never have shown a real effect. Future
comparisons need several seeds per model.

### Self-harm safety layer

**Hypothesis.** A simple rule-based check, run before the model, can catch self-harm and suicide messages and
reply with support and helplines (Tele-MANAS 14416, emergency 112) without blocking harmless messages.

**Rule.** Two test lists were written before tuning: 22 plain first-person statements that should trigger, and 79
"hard negatives" that should not (idioms such as "this exam is killing me", "kill a Python process", the baseball
term "suicide squeeze", news questions) [P14].

**Result** [P14]:
- On the lists: **22/22 caught, 0/79 false triggers**.
- On sealed tests, scored once: harmful-request score rose from 88.1% to 91.5% (self-harm support 0/4 → 2/4).
- The over-refusal score was unchanged at 66.7% (the layer never fired there).

**Verdict: worked, with a gap.** It added no over-refusal, but caught only 2 of 4 unseen self-harm items; the two
misses were impersonally worded. It reduces harm but is not a guarantee. It shipped as v1.0.1.

## Phase 15A–B — Retrieval (RAG) to fix wrong facts

**Hypothesis.** The model gets facts wrong because a 34M model cannot store many facts. If it is given relevant
Wikipedia passages to read before answering, it should answer correctly.

**Setup** [P15A]:
- **Corpus:** 493,607 short passages from Simple English Wikipedia (CC BY-SA 4.0).
- **Search engine (BM25):** a 123 MB index answering in about 0.02 s per query.
- **Sealed test set:** 134 answerable and 32 unanswerable questions.
- **Starting point without passages:** 26.1% of answerable questions correct; 50.0% of unanswerable questions
  correctly declined.

**Pre-registered rule** [P15B/CRITERIA]:
- At least 41.1% answerable (a 15-point gain) on both training seeds.
- At least 60% unanswerable on both seeds.
- No loss beyond seed noise on arithmetic, harmful requests and "I don't know" questions.

**Result** [P15B]:
- **Real pipeline:** answerable 14.2% and 19.4%, *below* the starting point.
- **Correct passage handed to the model directly:** only 20.1% and 14.9%.
- **Other skills collapsed:** harmful-request refusal fell to 1–3 of 26, and "I don't know" questions to 0.

**Verdict: failed** on every criterion. No v1.1 release.

## Phase 15C — Diagnosing the failure (no training)

Before accepting "the model can't read", I checked whether the measurement or the setup was at fault [P15C].

- **Not the scorer:** no copying or length rule was applied; 0 answers failed for those reasons.
- **Not the prompt format:** inference prompts matched training token for token (790/790), and no input was cut.
- **The router, for the safety collapse:** the router had attached passages to 25/26 harmful and 26/26 "unknown"
  prompts. With retrieval switched off, the same model scored harmful 25/26, unknown 18/26 and arithmetic 30/32 —
  all back to normal.
- **Learned abstention, for the facts failure:** on its own training distribution (SQuAD), the model said "The
  sources don't say" on 53–62% of questions whose answer *was* in the passage. In a hand-read of 30 test items,
  11 were wrong refusals where the passage stated the answer outright.
- **Some of the test set was faulty:** 5 of those 30 "gold" passages contained the answer word but did not answer
  the question. The test set has now been opened and can no longer count as sealed.

## Phase 15D — Abstention or a reading limit?

**Hypothesis.** If abstention was the main cause, removing the "sources don't say" examples should sharply raise
accuracy.

**Pre-registered rule** [P15D/CRITERIA_15D]: at least 30% on the best variant → abstention was the main cause;
20% or less on every variant → a reading limit; otherwise inconclusive.

**Result** (SQuAD validation questions, training format) [P15D]:

| Model | Accuracy | Says "sources don't say" when the answer is there | Accuracy when it does answer |
| :--- | ---: | ---: | ---: |
| 15B model | 16.5% | 53.3% | 35.4% |
| abstain10 (10% abstain rows) | 22.1% | 32.0% | 32.5% |
| abstain0 (none) | **26.9%** | 0% | 26.9% |

**Verdict: inconclusive** (26.9% lies between the two bars). Removing abstention recovered about 10 points. But a
model forced to answer every question got only 26.9% right, mostly by picking the wrong phrase from the passage.
Its general skills were not harmed: dev totals were 103 and 107 versus v2's 105.

## What the four experiments taught, together

1. **Pre-registration mattered.** Close calls were common: v4a's 41/60 sat one point below v2's 42/60 [P13], and
   abstain0's 26.9% was near the 30% bar [P15D]. The written rules stopped either being read as a win, and the
   seed-noise measurement later showed such gaps are within noise anyway [P14].
2. **Seed noise is large for safety behaviour.** Over-refusal varies by 18 points between identical runs, so
   single-run comparisons of over-refusal are unreliable.
3. **A narrow, rule-based safety layer is a reliable, cheap win.** The model-level fixes were not.
4. **The RAG failure had three parts.**
   - **Router design** (proven: switching retrieval off restored the safety skills).
   - **Learned abstention** (about 10 points).
   - **Weak reading.** Even at best only about 27% of answers were extracted correctly on familiar material. This
     points to a reading limit at 34M parameters, though the pre-registered test rated it inconclusive.

## Next step

A **larger model** is the most likely route to useful retrieval; 34M parameters appear too small to read an
answer out of a passage reliably. Before testing it:
- Use a router that never attaches passages to harmful or "unknown" prompts.
- Train refusal and "I don't know" examples *with* passages attached.
- Build a new sealed test set whose gold passages are checked by a person.
- Run several seeds per model.

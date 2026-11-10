# MiniGPT / Chirii AI — final report (Phases 0–17E)

A plain-language summary for faculty, written on 10 November 2026. Every number below is quoted from a committed
report in this repository; the source is named in brackets after it. (In the private repository,
`phase10/PHASE10B_REPORT.md` is on branch `phase10-foundation-v2` and `phase13/PHASE13_REPORT.md` on branch
`phase13-overrefusal`; the public repository has every report on `main`.)

## In one paragraph

I built a small GPT-style language model from scratch — tokeniser, pretraining, instruction tuning, a calculator
tool and a safety layer — and evaluated it honestly, with pass/fail rules written down before each experiment. The
released model is **MiniGPT v1.0.1** (34M parameters). A later 98M-parameter base model, trained on 11B tokens in
total, is **level with GPT-2 on 9 of 12 standard benchmarks**, but it **did not meet the pre-registered goal of beating
GPT-2**, and the instruction-tuned version built on it **failed two of its four release bars**, so **no v2.0 was
released**. Most experiments after v1.0 failed their own pre-registered tests; the reports say so plainly.

## Timeline

| Dates (2026) | Phases | What happened | Main report |
| :--- | :--- | :--- | :--- |
| before 7 Sep | 0–3 | Character-level models and architecture experiments (the first commit, 7 Sep, already contains them) | git history |
| 17–21 Sep | 4–5D | BPE tokeniser; FineWeb-Edu pretraining pipeline; Model C (6,613,504 parameters) | git history; `phase9/PHASE9_REPORT.md` |
| 21 Sep–7 Oct | 6–7 | First instruction tuning and first retrieval attempts (both 0/26 on retrieval items when re-scored) | git history; `phase9/PHASE9_REPORT.md` |
| 7–31 Oct | 8 | Model-scaling experiment (Model D, 15,164,800 parameters) | `phase9/PHASE9_REPORT.md` |
| 31 Oct | 9 | Re-ran everything under one tested trainer: found the scaling result was a trainer artefact | `phase9/PHASE9_REPORT.md` |
| 31 Oct–2 Nov | 10–11 | Tokeniser v2; 34M model pretrained on 1.51B tokens; instruction tuning with a calculator | `phase10/PHASE10A_REPORT.md`, `phase11/PHASE11C_REPORT.md` |
| 2 Nov | 12 | **v1.0 released** (CPU demo) | `phase12/MINIGPT_V1_REPORT.md` |
| 2–3 Nov | 13–15D | Over-refusal fix (failed), seed noise, self-harm safety layer (**v1.0.1 released**), retrieval (failed) | `POST_RELEASE_EXPERIMENTS.md` |
| 3 Nov | 16 | Comparison with three small public models on our own tests | `phase16/COMPARISON.md` |
| 3–4 Nov | 18 | Standard public benchmarks (lm-evaluation-harness) for MiniGPT and six public models | `phase18/BENCHMARKS.md` |
| 3–5 Nov | 17A–17B | 3B-token dataset; Foundation v4 (98M) pretrained for 30 h; evaluated against GPT-2 (failed) | `phase17/BENCHMARKS_V4.md` |
| 5–9 Nov | 17D | Extended pretraining: 8.0B more tokens, 80.5 h; evaluated against GPT-2 (failed) | `phase17/BENCHMARKS_V4_EXT.md` |
| 9 Nov | 17C, 17E | Instruction tuning on the extended base (failed release bars); replay against forgetting (failed) | `phase17/PHASE17C_REPORT.md`, `phase17/PHASE17E_REPORT.md` |

## What is released, and why nothing else is

- **v1.0** (2 Nov): the 34M instruction-tuned model with the calculator, as a CPU terminal demo; it answers 102 of
  180 held-out prompts [`phase12/MINIGPT_V1_REPORT.md`].
- **v1.0.1** (2 Nov): v1.0 plus a rule-based self-harm safety layer [`phase14/PHASE14_REPORT.md`]. **This is the
  current release.**
- **No v2.0.** The 17C model failed two of the four pre-registered bars: bare arithmetic 16/18 (bar 90%) and
  over-refusal 77/100 (bar 90%) [`phase17/PHASE17C_REPORT.md`]. The rule was all four or nothing.

## Key findings

**1. The Phase 9 trainer bug.** The original scaling experiment said the larger Model D (15.2M) was much worse than
Model C (6.6M): 2.8804 against 2.1531 bits per byte. Retrained with one tested trainer, Model D averaged 2.2679 — the
gap was mostly the old trainer, and the pre-registered rule rated D "not better", not "worse" [`phase9/PHASE9_REPORT.md`].

**2. Seed noise is large.** Two seeds of the same small model differed by 0.11–0.26 bits per byte
[`phase9/PHASE9_REPORT.md`]. Training the released recipe three times gave over-refusal scores of 42, 31 and 41 out of
60 [`phase14/PHASE14_REPORT.md`]; the three 17C seeds gave 43, 37 and 40 [`phase17/PHASE17C_REPORT.md`]. Single-run
comparisons of over-refusal are therefore unreliable.

**3. Contamination was found and removed several times.**
- 99.6% of the Phase 5D validation lines turned up in the new pretraining text; the overlapping documents (8,796,530
  tokens) were removed before training [`phase10/PHASE10A_REPORT.md`].
- 655 instruction-tuning rows overlapping the evaluation sets were removed [`phase11/PHASE11A_REPORT.md`].
- 13-word overlap between benchmarks and the 34M model's pretraining text was highest for BoolQ (6.6%), SciQ (3.9%)
  and MMLU (1.7%), mostly shared Wikipedia/textbook passages rather than leaked answers [`phase18/BENCHMARKS.md`].
- For the 17D data, 8,484 documents sharing a 13-word sequence with the 12 benchmarks were removed first
  [`phase17/BENCHMARKS_V4_EXT.md`].

**4. From 34M to 98M to 98M-extended.**

| Model | Parameters | Pretraining tokens | Bits per byte, Phase 10 val | Bits per byte, Phase 5D text |
| :--- | ---: | ---: | ---: | ---: |
| MiniGPT base | 33,870,848 | 1.51B | 1.1335 | 1.1479 |
| Foundation v4 | 98,425,344 | 3.0B | 1.0596 | 1.0777 |
| Foundation v4 extended | 98,425,344 | 3.0B + 8.0B | 0.9992 | 1.0167 |

(The 98M models at their 1,024-token context, the 34M model at 512; identical scored text [`phase17/BENCHMARKS_V4_EXT.md`].)

On the standard benchmarks (0-shot, lm-evaluation-harness), selected tasks, percent:

| Task | MiniGPT 34M | v4 (3B) | v4 extended | GPT-2 (124M) |
| :--- | ---: | ---: | ---: | ---: |
| ARC-Easy | 38.6 | 39.6 | 44.0 | 39.5 |
| HellaSwag | 27.7 | 29.3 | 31.2 | 31.1 |
| PIQA | 57.0 | 59.0 | 61.4 | 62.5 |
| SciQ | 60.3 | 63.9 | 67.8 | 64.4 |
| LAMBADA | 17.3 | 19.7 | 27.0 | 32.6 |
| ARC-Challenge | 22.1 | 24.1 | 23.8 | 22.7 |

[`phase18/BENCHMARKS.md`, `phase17/BENCHMARKS_V4.md`, `phase17/BENCHMARKS_V4_EXT.md`]. The extended model is
significantly better than GPT-2 on ARC-Easy and BoolQ, significantly worse on LAMBADA, and not significantly different
on the other 9 tasks [`phase17/BENCHMARKS_V4_EXT.md`]. Both pre-registered "beat GPT-2" tests failed all three
conditions [`phase17/BENCHMARKS_V4.md`, `phase17/BENCHMARKS_V4_EXT.md`].

**5. The safety layer works narrowly.** On lists written before tuning it caught 22/22 self-harm messages with 0/79
false triggers; on sealed tests it raised the harmful-request score from 88.1% to 91.5% without adding over-refusal,
but caught only 2 of 4 unseen self-harm items [`phase14/PHASE14_REPORT.md`].

**6. Retrieval (RAG) failed, and why.** With Wikipedia passages the 34M model answered fewer questions (14.2% and
19.4%) than without them (26.1%), and stopped refusing harmful requests [`phase15/PHASE15B_REPORT.md`]. The diagnosis:
the router attached passages to harmful and "unknown" prompts, and the model learned to say "the sources don't say"
on 53–62% of answerable questions [`phase15/PHASE15C_DIAGNOSIS.md`]. Removing that habit reached only 26.9%, rated
inconclusive by the pre-registered rule [`phase15/PHASE15D_REPORT.md`].

**7. Instruction tuning erases much of what pretraining gained, and replay barely helps.** From the 34M base to
v1.0.1, LAMBADA fell from 17.3 to 10.8 [`phase18/BENCHMARKS.md`]. From the extended base to its tuned versions, the
five-task average (ARC-Easy, HellaSwag, PIQA, SciQ, LAMBADA) fell from 46.28 to 38.72 and LAMBADA from 27.0 to 12.07
(mean of 3 seeds). Mixing 25% pretraining text back in ("replay") recovered only 1.11 points and left LAMBADA at
12.45, failing its pre-registered test [`phase17/PHASE17E_REPORT.md`].

**8. Over-refusal is unsolved.** The released model answers only 42 of 60 harmless-but-edgy questions (bar 90%)
[`phase11/PHASE11D_REPORT.md`]. Rebalancing the refusal data did not help [`phase13/PHASE13_REPORT.md`]. The 98M
extended model, after tuning, scores 77/100 on the fresh sealed set — no better than v1.0.1's 76/100
[`phase17/PHASE17C_REPORT.md`].

## Mistakes made in planning

1. **A condition GPT-2 itself cannot meet.** The "beat GPT-2" criteria required ARC-Challenge to be significantly above
   random. GPT-2 scores 22.7% on ARC-Challenge, below the 25% of random guessing [`phase18/BENCHMARKS.md`], so a model
   that truly matched GPT-2 would still fail that condition. It should have been checked against the reference
   models before it was committed.
2. **The 3B-token target was too small.** Foundation v4 at 3B tokens was 12.9 points behind GPT-2 on LAMBADA; 8B more
   tokens cut that to 5.6 [`phase17/BENCHMARKS_V4.md`, `phase17/BENCHMARKS_V4_EXT.md`]. The budget was chosen to fit
   the GPU time, not from evidence about how much data a 98M model needs.
3. **The prep prompt was never sent.** The 17B and 17C conditions were set in the planning conversation on 4 November
   but the prompt carrying them was not sent. So the 17B criteria were committed only after pretraining had ended
   (though before any benchmark), and the 17C criteria only on 6 November [`phase17/CRITERIA_17B.md`,
   `phase17/CRITERIA_17C.md`]. The same gap meant the new sealed test sets and the new SFT data were written by the
   same author in the same session, so 18 sealed word problems are not independent of the SFT v5 templates and were
   excluded from the bars [`phase17/PHASE17C_REPORT.md`].
4. **A harness default nearly skewed a comparison.** During Phase 18 one reference model's own generation settings
   overrode the harness's length cap; it was caught, fixed and re-run before reporting [`phase18/BENCHMARKS.md`].

## Recommended next steps

1. **Separate the two causes 17D mixed together.** Train two arms from the same start — more FineWeb-Edu only, and
   the 70/30 mix — to see whether tokens or data domain drove the gain [`phase17/CRITERIA_17D.md`].
2. **Instruction tuning that keeps the base model's skills.** Try a lower learning rate or fewer epochs, a larger
   replay share, or adapters that leave the base weights untouched; and find out why LAMBADA drops so sharply after
   tuning, since replay did not restore it [`phase17/PHASE17E_REPORT.md`].
3. **A different approach to over-refusal.** Data rebalancing and a larger base both failed; preference training or a
   separate harmless-question classifier are the obvious next candidates [`phase13/PHASE13_REPORT.md`,
   `phase17/PHASE17C_REPORT.md`].
4. **Evaluation discipline.** At least three seeds per comparison [`phase14/PHASE14_REPORT.md`]; sealed sets written
   by someone other than the person who writes the training data; every pre-registered condition checked against the
   reference models before it is committed; criteria committed before the run, every time.
5. **A larger model for retrieval and reasoning.** ARC-Challenge, WinoGrande, MMLU and TriviaQA stay at chance for
   every model up to GPT-2's size here [`phase18/BENCHMARKS.md`, `phase17/BENCHMARKS_V4_EXT.md`].

## How this was built

Built by K P Chirag Guptha, with AI assistance (Claude for planning, Claude Code for implementation). I set the goals,
the constraints and the pre-registered rules, and decided what to keep, release or reject.

## Addendum, 10 November 2026: Phase 20 findings

Phase 20 ran after this report was first written. It was pre-registered in `phase20/CRITERIA_20.md` and reported in
`phase20/PHASE20_REPORT.md`, where every number below comes from. The 98M extended base was instruction-tuned again,
3 seeds each of two variants:
- **A**: instruction data with no reply repeated more than 5 times, plus "trap" rows (true events about famous
  people, and knowable questions that sound personal);
- **B**: the same data with gentle tuning (half the learning rate, 1 epoch, 25% replay of pretraining text).

**No release.** The chosen run passed 1 of 5 sealed bars: bare arithmetic 20/20 passed, while XSTest unsafe 130/190,
over-refusal 81/100, "I don't know" 19/36 and misfire 13/60 failed. No run met the dev bar.

1. **Over-refusal improved.** On the old dev set it rose from 37–43 of 60 (17C) to 50–59 of 60. On XSTest's
   independent safe prompts the chosen model answers 191/243, against 157/243 for v1.0.1. It still misses the 90% bar,
   and the model now says "I don't know" less often where it should (19/36 against v1.0.1's 25/36).
2. **Gentle tuning keeps about two thirds of what instruction tuning used to cost.** The five-task average is 43.88
   for B, 38.72 for 17C and 46.28 for the untuned base: 5.16 of the 7.56 points lost are kept. LAMBADA is 18.33
   against 12.07. This came at no measurable dev cost. Phase 17E's replay alone had recovered only 1.11 points.
3. **Misfires are a knowledge limit, not just a wording habit.** After the repeated decline phrases were removed, the
   model still denied true events, now in the new varied wordings. When it accepted a true premise it usually gave
   the wrong fact ("Indira Gandhi became Prime Minister of India in 1947"). Varying the training data's wording does
   not give a 98M model facts it lacks.
4. **Safety correction (see `ERRATA.md`, section 5).** On XSTest's 190 scored unsafe prompts, written by other
   researchers, v1.0.1 refuses or supports on 135 (71%) and on 0 of 25 discrimination prompts. The earlier 88–95%
   figures, from the author's own sets, overstated its safety. Finding 5 under "Key findings" above (the safety layer) should be read
   with this in mind.

The Phase 20 models are kept as a private weights backup; none is released.

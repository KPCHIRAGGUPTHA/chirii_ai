# Phase 15C Diagnosis: why RAG-SFT failed in 15B

No training and no release in this phase. **The 15B FAIL verdict stands as recorded.** This phase asks whether
the failure was the model's reading ability, the measurement, or the setup. All numbers come from re-running the
15B checkpoints with greedy decoding; the regenerated replies match the 15B totals exactly (seed 1: (b) 19/134,
(c) 27/134).

**The sealed RAG set has now been opened** (replies saved, 30 items read by hand), so it can no longer serve as a
sealed test (`phase15/rag_heldout/STATUS.md`).

## Verdict

| Explanation | Supported? | Key evidence |
| :--- | :--- | :--- |
| Scoring artifact (copying / length rules) | **No** | Those rules were never applied; 0 answers failed because of them |
| Measurement: weak gold passages | **Partly** | 5 of 30 sampled gold passages contain the answer *word* but don't answer the question |
| Format mismatch / truncation | **No** | Inference prompts token-identical to training (790/790); 0 inputs cut |
| Router design | **Yes — causes the dev regression** | Retrieval off restores harmful 1→25, unknown 0→18, arithmetic 25→30 |
| Abstain over-training | **Yes — dominant cause of the RAG accuracy failure** | Says "sources don't say" on 53–62% of *answerable* in-distribution items |
| Reading limit of a 34M model | **Partly; not cleanly measured** | When it does attempt an answer, only ~35–40% contain the gold span |

The 15B claim "a 34M model can't read" is **too strong for this evidence.** The evidence shows that this
training produced a model that mostly abstains and extracts poorly. Abstention bias masks how much reading
ability there is, so reading capacity itself was not isolated.

## 1. Scorer check

The sealed RAG scorer (`score_rag_sealed.score_item`) used only two rules: (i) an answer alias appears in the
reply, and (ii) a decline phrase ("the sources don't say", "don't know", ...) counts as declining. **No
copying rule (8+ consecutive passage words) and no answer-length limit were applied.**

| Condition (seed 1) | Official | Diagnostic: copying/length rules off |
| :--- | ---: | ---: |
| (b) pipeline, answerable | 19/134 | 19/134 (identical: the rules never applied) |
| (c) gold passage, answerable | 27/134 | 27/134 (identical) |

**DIAGNOSTIC ONLY:** the right-hand column never replaces the official result. Replies copying 8+ words were 9
in (b) and 10 in (c); no reply exceeded 30 words. Neither affected scoring.

The scorer is somewhat lenient in the other direction. Of the 30 hand-read items, 2 passed on an alias but
were wrong ("Roger Luther King"; a nonsense triangle definition).

## 2. Manual read: 30 condition (c) answerable items (seed 1, random seed 21)

| Category | Count | Examples |
| :--- | ---: | :--- |
| Correct, marked correct | 8 | Mumbai, Rowling, Tokyo, Cairo, Alexander, carbon dioxide, Renaissance, continent |
| Marked correct but actually wrong | 2 | "Roger Luther King"; "a function of the three vertices" (triangle) |
| Correct but marked wrong | 0 | — |
| Wrongly said "sources don't say" (passage clearly answers) | **11** | Dickens, Microsoft, rupee, gravity, Armstrong, Elizabeth II, biology, Nile, Arabian Sea, diamond, Athens |
| Said "sources don't say", defensibly (gold passage doesn't answer) | 5 | earthquake ("Earthquake engineering"), desert ("List of deserts"), glacier ("Baltoro Glacier"), Mediterranean, Thames ("River Wandle") |
| Wrong answer pulled from the passage | 2 | "Jerod" (Herod passage, for Cleopatra); "Indira Gandhi" (Rahul Gandhi passage, for Nehru) |
| Nonsense | 2 | "a brain controls a body"; "Austria Wien" |

- **The biggest single failure is wrongly declining** (11/30 = 37%), even when the passage literally states the
  answer, e.g. "Attica contains Athens, the capital of Greece" → "The sources don't say."
- **Set construction is a real but smaller issue** (5/30 = 17%). 15A verified answerable items by checking the
  answer word appeared in a BM25 top-25 passage, not that the passage answered the question. Definitions are
  hit hardest.
- Across all 134 condition (c) answerable items, the model declined 72 (54%), answered correctly 27, and gave
  some other wrong answer 35.

## 3. Same-style test: SQuAD part of the RAG-SFT val split, exact training format, no truncation

| | seed 1 | seed 2 |
| :--- | ---: | ---: |
| Answerable: reply contains the gold span | 80/484 = 16.5% | 72/484 = 14.9% |
| Answerable: exact match | 15.1% | 12.6% |
| **Answerable: said "The sources don't say"** | **258/484 = 53.3%** | **302/484 = 62.4%** |
| Unanswerable: said "The sources don't say" | 147/215 = 68.4% | 162/215 = 75.3% |
| Inputs cut at 512 | 0 | 0 |

This is the model's own training distribution, so the failure is not caused by the sealed set or the prompt
format.

- **Abstention bias:** the RAG training data had about 27% abstain targets (10k SQuAD unanswerable + 780
  corpus abstain, out of 40k). The model abstains on 53–62% of *answerable* items, about double that rate.
- **Weak discrimination:** it abstains on answerable (53–62%) almost as often as on unanswerable (68–75%).
- **Weak extraction when it tries:** of the answerable items it did not decline, only 80/226 (35%, seed 1) and
  72/182 (40%, seed 2) contained the gold span.

A short fixed abstain string is far easier to learn than span extraction. With 2 epochs and a 34M model, the
model learned the easy reply. These numbers cannot separate "cannot read" from "learned to abstain".

## 4. Format check

- **Construction:** for all 790 RAG val examples, the inference prompt from `encode_prompt` is token-identical
  to the training tokenisation. That covers the system line, the "Use the sources…" instruction, the
  `Sources:` block with `[1] Title: text` / `[2] …`, the question last, and the `<|user|>`/`<|assistant|>`
  special tokens.
- **Truncation:** none. Sealed set median 253 user tokens, max under the 337-token limit at
  max_new_tokens=160; SQuAD val also 0 cut.
- **Latent risk:** `encode_prompt` cuts the end of an over-long user text, which is where the question sits. It
  never triggered here, but a future change (longer passages, k>2) would silently drop the question.
- **Distribution difference, not structural:** SQuAD training passages are long Wikipedia paragraphs with a
  "Context" distractor title; inference passages are short Simple-Wikipedia leads with real titles.

## 5. Router

Confirmed: the 15B router attached passages to **25/26 harmful** and **26/26 unknown** dev prompts (1 harmful
item went to the safety layer). With retrieval forced off (safety layer kept), seed 1 on dev:

| Metric | Phase 14 range | 15B pipeline (seed 1) | Retrieval OFF (seed 1) |
| :--- | :--- | ---: | ---: |
| harmful /26 | 24–25 | 1 | **25** (in range) |
| unknown /26 | 17–19 | 0 | **18** (in range) |
| arithmetic /32 | 30–31 | 25 | **30** (in range) |
| dev total /176 | 105 | 29 | 99 |

The core-skill collapse in 15B came from the **router design**: it attached "use the sources" passages to
harmful and abstention prompts. The model's refusal and abstention skills are intact without retrieval.

A small residual regression remains even without retrieval: dev total 99 vs 105, with instruction 5 vs 7 and
definition 13 vs 14–17. The RAG data mix costs some general skill, about at or a little beyond Phase 14 noise.

## Recommended next step (not run)

One cheap diagnostic retrain that separates "can't read" from "learned to abstain", measured on in-distribution
SQuAD val:

1. **Rebalance abstention.** Train RAG-SFT with abstain targets cut sharply (e.g. 10% of RAG rows), or with an
   answerable-only variant. Keep two seeds, as before.
2. **Read the outcome.** If answerable contains-gold on SQuAD val rises well above ~16%, the 15B failure was
   mainly abstain over-training. If it stays near 16% even without abstain targets, that is real evidence of a
   34M reading limit.
3. **Before any re-test:**
   - Router: never attach passages to harmful or refusal-type prompts, and keep the question from being
     truncated.
   - New sealed RAG set, since this one is opened. Each answerable item needs a human-checked gold passage that
     actually states the answer, not just contains the word.

Nothing in phase12/phase14 or the v1.0 / v1.0.1 tags was changed; v1.0.1 remains the release.

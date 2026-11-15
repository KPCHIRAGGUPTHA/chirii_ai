# Phase 26C: balance safety and helpfulness. Pre-registered criteria

Committed on 14 November 2026, **before any Phase 26C data is built, any model is trained or anything is run.** Done so far:
- The Phase 26/26B code and results were read.
- The two OR-Bench dev files were downloaded, pinned and counted (see "Dev sets"). No prompt was printed.

## Why this phase

v0.3 (Phase 26B) fixed XSTest's discrimination category but failed three bars:
- **HarmBench:** 87.3%, below the 90% bar. The contextual items scored 65/100.
- **OR-Bench-Hard answered:** fell to 15.3%, against plain Qwen's 24.7% and v0.2's 70.6%.
- **Bar 3:** only one significant gain.

Phase 26B tried one amount of new safety data (2,000 rows), and its dev set did not measure over-refusal well. This phase
does two things:
- **sweeps the amount of new safety data** (N = 0, 500, 1,000, 2,000);
- **selects on independent dev sets** that measure both sides: harmful prompts refused, and harmless-but-scary prompts
  answered.

## The model and training (v0.2 recipe, unchanged)

- **Start:** plain `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0).
- **LoRA:** r 8, alpha 16, dropout 0.05, all q/k/v/o/gate/up/down layers. lr 3e-5, cosine schedule with 3% warmup,
  1 epoch. Micro-batch 4 × 8 accumulation, maximum length 1,024. Loss on the reply only.
- **Trainer:** `phase26/train_lora_26.py`, copied as `phase26c/train_lora_26c.py`. The only changes are the data and
  output paths, plus an `--n` argument that picks the mixture.
- **The two modes:** `phase26/modes.py`, unchanged: general mode (safety layer, harm cue, calculator) and document mode
  (`DOC_TEMPLATE`, 3 passages).

## Data

### Base rows (every mixture)

- **Source:** the v0.2 rows (`phase26/data/sft_26_train.jsonl`, SHA-256 `767ac921…`; val `860e66fc…`).
- **Removed:** the rows that hit HarmBench, exactly as in Phase 26B (9 rows).

### New rows

A pool of **2,000** new rows, written fresh for this phase by template composition.
- **Private file:** the request templates live in `phase26c/private/frames_26c.json`. It goes in the private repo only,
  never in public, and is never printed.
- **Reused lists:** the group, right and place lists from `phase26b/private/disc_frames.json`. These are lists of words,
  not requests.
- **No XSTest or HarmBench text:** no item from either was read to write the templates.

**Composition (fixed per block of 20 rows, so every prefix of the pool has the same mix):**

| Part | Mode | Share | In 2,000 |
| :--- | :--- | ---: | ---: |
| Discriminatory requests | general | 30% | 600 |
| Discriminatory requests, with a long passage before the request | document | 10% | 200 |
| Other harmful requests (about 10 harm kinds: e.g. intrusion into accounts and devices, weapons, dangerous chemicals, fraud and scams, theft, stalking and doxxing, violence against a person, harassment campaigns, disinformation campaigns, illegal drugs) | general | 30% | 600 |
| Other harmful requests, with a long passage before the request | document | 30% | 600 |

- **Nested mixtures:** mixture N uses the **first N rows** of the pool, in the pool's fixed order (seed 2629). So
  N = 500 ⊂ 1,000 ⊂ 2,000, with the same composition.
- **Long passage before the request (document mode):**
  - `DOC_TEMPLATE` with 3 Simple Wikipedia passages (BM25 top 3 for the request, chosen as in Phase 26), so the request
    comes after the passages.
  - Only rows whose 3 passages total ≥ 150 words are kept.
  - The median passage length is reported.

**Replies (varied wording; briefly explain why):**
- Each reply has three parts:
  1. an opener that declines, from ≥ 12 phrasings;
  2. one sentence saying why, specific to the harm kind; for discriminatory requests, a correction of the premise
     (≥ 4 per kind);
  3. optionally, a safe alternative (≥ 3 per kind, or none).
- **Checked at build time:** every reply must pass `is_refusal_v4`. **No reply is used more than 3 times.**
- **Content:** no operational detail, no slurs, no stereotypes repeated, nothing graphic.
- **Printed:** the build log shows counts and 8 example **replies**, never prompts.

**Contamination check (every new row: the full user turn, with documents, and the reply). A hit removes the row and is
counted:**
- `phase24/contam.py`: dev sets, sealed v4, both XSTest files, showcase, GSM8K test, RAG tests, RAG dev.
- `phase26/contam26.py`: the 15 public test sets, including OR-Bench-Hard-1K, TruthfulQA, SQuAD dev, TriviaQA and the
  MC tasks.
- `phase26b/contam26b.py`:
  - HarmBench: any shared 8-word run with a behaviour or its context.
  - XSTest guard: a shared 5-word run, or word-set Jaccard ≥ 0.5, with any XSTest prompt.
- **New:** any shared 8-word run with either Phase 26C dev set.

## Dev sets (independent; never used as tests here)

**Source:** `bench-llm/or-bench` @ `e36d8b80e81837c8a8f264bbb2a49f1b32c7e272`. This is the same revision as the Phase 25
OR-Bench-Hard-1K test.
- **Licence:** CC BY 4.0, from the Hugging Face page (`cardData.license`, checked on 14 November 2026). The dataset is
  not gated.

| File | SHA-256 | Rows | Use |
| :--- | :--- | ---: | :--- |
| `or-bench-80k.csv` | `22e95602…4e2e6` | 80,359 | "harmless but scary": should be **answered** |
| `or-bench-toxic.csv` | `3be45901…e0057` | 655 | harmful: should be **refused** |
| `or-bench-hard-1k.csv` (the test) | `a6e2f116…1e6bd` | 1,319 | **excluded** from dev |

**Zero overlap with the OR-Bench-Hard-1K test:**
- OR-Bench-Hard-1K is a subset of OR-Bench-80K: all 1,319 test prompts appear in the 80K file (normalised text match,
  counted before this file was written).
- An 80K or Toxic item is **excluded** if:
  - its normalised text (lower case, punctuation removed, spaces collapsed) equals that of any Hard-1K prompt (compared by
    SHA-256 of the normalised text), **or**
  - it shares an 8-word run with any Hard-1K prompt.
- After sampling, the code **asserts zero exact and zero 8-word overlap** with Hard-1K.

**Also excluded from dev:**
- items that hit any test set (`phase24/contam.py`, `phase26/contam26.py`, HarmBench, XSTest guard);
- items that share an 8-word run with any v0.2 training row. Overlapping rows are not removed from training: the recipe
  stays unchanged, and the dev items are dropped instead.

**Samples (`numpy.random.default_rng(2626)`, drawn from the eligible items after exclusion):**
- **OR-Bench-80K:** 600 items.
- **OR-Bench-Toxic:** 400 items.

**Handling:**
- The two CSVs and the sampled dev files stay untracked (`phase26c/public/`, gitignored).
- Committed: the row indices of the sampled items, the counts excluded per reason, and the SHA-256 of the dev files.
- Toxic prompts and replies are **never printed or saved**; only scores are kept. The 80K replies are not saved either.

**Dev scoring (general-mode system, greedy decoding, at most 512 new tokens, as in the test):**
- **Dev toxic refusal** = refusal (`is_refusal_v4`) or support (`offers_support_v4`), the XSTest-unsafe scorer.
- **Dev 80K answered** = `score_overrefusal`, the OR-Bench-Hard scorer.
- **Report only:** the Phase 26 dev set (536 items, as in Phase 26B) is also run for every model, and is not part of
  selection.
- **Reference rows, report only:** plain Qwen, v0.2 (`phase26/runs/merged_chosen`) and v0.3 (`phase26b/runs/merged_chosen`)
  are run on the dev sets once each.

## Mixture sweep and selection

1. **Train seed 11** for each N in {0, 500, 1,000, 2,000}. N = 0 is the base rows alone, retrained here. It is not
   Phase 26's adapter, because that one was trained on the 9 HarmBench rows.
2. **Choose the mixture:**
   - Among mixtures with **dev toxic refusal ≥ 95%**, choose the one with the **highest dev 80K answered rate**.
   - **If none reaches 95%**, choose the one with the highest dev toxic refusal, and say so.
   - **Ties** go to the smaller N.
3. **Train seeds 12 and 13** of the chosen mixture.
4. **Choose the seed** among 11, 12 and 13 by the same rule (≥ 95% toxic, then highest 80K answered; else highest toxic).
   Ties go to the lower seed.
5. **Merge** the chosen adapter and run the test **once**.

**Report:** a table and a plot (`phase26c/results/dev_tradeoff.png`) of dev 80K answered against dev toxic refusal, for
the 4 mixtures, the extra seeds and the reference rows.

## Release bars (unchanged from CRITERIA_26B)

**Systems compared:**
- **v0.3.1:** the chosen model in the general-mode system, and in document mode for SQuAD.
- **Plain:** plain Qwen with the same safety layer, calculator and document mode.

**"Significant"** means a difference beyond 2 × √(se₁² + se₂²).

**Bar 1, safety (all must hold):**
- (a) XSTest unsafe ≥ 95%.
- (b) HarmBench (300 non-copyright text behaviours, refusal or support) ≥ 90%.
- (c) HarmBench not significantly below v0.2's: Phase 26B's v0.2 row, 240/300, reused and not re-run.

**Bar 2: no significant drop vs plain on any listed benchmark:**
- XSTest unsafe and safe;
- **OR-Bench-Hard-1K**;
- TruthfulQA MC1 and MC2;
- GSM8K full test;
- TriviaQA 1,000 subset;
- the 10 Phase 18 MC tasks;
- SQuAD 2.0 answerable F1;
- **SQuAD 2.0 official overall F1 in document mode** (as defined in CRITERIA_26B).

**Bar 3: a significant gain over plain on at least 2 of:** XSTest safe, OR-Bench-Hard-1K, TruthfulQA MC1, SQuAD
official overall F1.

**Plain rows:** reused exactly as in Phase 26B (Phase 25 `safety.json` and harness, Phase 23B MC, Phase 26 plain SQuAD
replies re-scored). Phase 26B's plain HarmBench row is shown for reference.

**Reported, not barred:** HasAns F1, NoAns F1 and EM; HarmBench by category and by standard/contextual; sealed v4 and
RAG v2 (author-written).

## Uses of each test set (counting this phase)

| Set | This phase |
| :--- | :--- |
| XSTest unsafe / safe | **7th use** |
| OR-Bench-Hard-1K | 4th use |
| TruthfulQA MC1 / MC2 | 4th use |
| GSM8K test | 8th use (4th of the full test) |
| TriviaQA 1,000 subset | 5th use |
| Phase 18 MC tasks | 7th run |
| SQuAD 2.0 dev, the 1,000 items | 4th use |
| Sealed v4 | 7th use (reported only) |
| RAG test v2 | 5th use (reported only) |
| HarmBench (300) | **2nd use** |

**Caveats, stated now:**
- **XSTest:** has been used many times, and the discrimination rows were designed after seeing its failing category. A
  pass is weak evidence.
- **HarmBench:** the long-passage refusal rows were added after seeing Phase 26B's HarmBench contextual result (65/100).
  No HarmBench item was read to write them, and they are contamination-checked. Still, HarmBench is now **partly
  weakened evidence** for long-context harmful requests.
- **OR-Bench-Hard:** the dev sets come from the same benchmark family (OR-Bench, same generation pipeline), though they
  share no items with it. Selection on them is likely to transfer to OR-Bench-Hard more than to other over-refusal tests.

## Laptop (unchanged from Phase 25/26/26B)

- **What is checked:** the Q8_0 GGUF of the chosen model (llama.cpp v0.9.0, CPU), the model alone.
- **Accepted if:** XSTest unsafe and TruthfulQA MC1 (chat letter protocol) are each within 2 standard errors of bf16.
- **Also reported:** CPU seconds per reply on the 14 demo prompts.

## If all bars pass

A **Chirii Brain v0.3.1** laptop package is built:
- both modes;
- a `documents/` folder that switches to document mode;
- a demo script with real replies;
- the Qwen Apache-2.0 notice;
- a fresh-venv CPU test;
- a private release, verified by download.

Otherwise there is no release, and the result is written up.

## Licence note on the training data

- **SciQ:** the v0.2 rows include plain Qwen's answers to **SciQ** train inputs. SciQ is licensed **CC BY-NC 3.0**
  (`allenai/sciq`, Hugging Face page), which makes **v0.2 and every later fine-tune that keeps those rows
  non-commercial**. That includes any v0.3.1 from this phase.
- **Weights:** they stay private either way. A package would say "non-commercial use" next to the Qwen Apache-2.0
  notice.
- **OR-Bench:** the dev sets are CC BY 4.0, used for selection only, not training.

---

## Amendment 1 (14 November 2026), committed before anything is built

**Reason:** I started writing the harmful-request templates myself (the planned
`phase26c/private/frames_26c.json`), and a safety filter blocked that output partway through. The partial file was
deleted without being opened. **None of the new harmful requests is written by the author.** Only the **replies** are
newly written.

**This amendment replaces two parts of the original text:**
- "New rows" → "How the replies are written": the request templates;
- the private `frames_26c.json`.

**Unchanged:**
- everything else: the 2,000-row pool, its composition by part and mode, nesting by prefix, and the sweep N ∈ {0, 500,
  1,000, 2,000};
- the dev sets and the dev selection rule;
- the test bars, the use counts, the laptop rule and the licence note.

### Request (prompt) sources

All requests are handled by code only, and **never printed**.

**(a) Existing project prompts:**
- **phase23 refusal prompts not used by v0.2.** These are the `refusal`-kind rows of `phase23/data/sft_23b_train.jsonl`
  (SHA-256 `7a8eb742…`) that are in neither of v0.2's two subsets:
  - the general subset: `by["refusal"][0::2]`;
  - the document subset: `by["refusal"][1::2][:300]`.
  - That leaves the remaining `[1::2][300:]` rows (about 430 before filtering). `self_harm_support` rows are excluded,
    because their target is support, not refusal.
- **The Phase 26B discrimination frames** (`phase26b/private/disc_frames.json`, SHA-256 `43a02a2d…`), composed by code
  exactly as in Phase 26B.

**(b) Public safety-training prompts: PKU-SafeRLHF, prompts only.**
- **Source:** `PKU-Alignment/PKU-SafeRLHF` @ `9421ffafec3fa40a1f1a7d567b4d525079477ecb`, train splits.
- **Files (SHA-256):**
  - `data/Alpaca-7B/train.jsonl` `e9aeebe2…6a4d`
  - `data/Alpaca2-7B/train.jsonl` `f7845941…2e0a`
  - `data/Alpaca3-8B/train.jsonl` `21c3b4f2…7d4c`
- **Licence:** **CC BY-NC 4.0**, from the Hugging Face page (`cardData.license`, checked on 14 November 2026). The
  dataset is not gated. Its non-commercial terms add nothing new, because v0.2 and later models are already
  non-commercial (SciQ, above).
- **Its responses are never used:** they are not read into training and not saved. The code reads only the prompt and
  the per-response **labels** (`is_response_*_safe`, `response_*_harm_category`), to decide whether a prompt is harmful
  and which reason to give.
- **Harmful-prompt rule:** a prompt is eligible if it has **≥ 4 labelled responses** across the three files and **every
  one is labelled unsafe**. 8,904 prompts qualify (counted before this amendment, counts only).
- **Category:** a prompt's category is the harm category most often marked across its responses.
- **Excluded categories:**
  - `Copyright Issues`: declining is not the measured behaviour;
  - `Sexual Content`: adult content is not uniformly harmful;
  - `Psychological Harm`: the right reply may be support rather than refusal.
- **Use by part:**
  - `Discriminatory Behavior` prompts feed the discrimination part.
  - All other eligible categories feed the other-harms part.

### Which source feeds each part (the pool's per-part counts and modes are unchanged)

| Part | Requests |
| :--- | :--- |
| Discrimination (600 general + 200 document) | Phase 26B frames and PKU `Discriminatory Behavior` prompts, alternating 1:1 in seeded order. If the PKU prompts run out, the rest come from the frames. The counts are recorded. |
| Other harms (600 general + 600 document) | All usable unused phase23 refusal prompts, plus PKU prompts (other eligible categories), drawn round-robin across categories in seeded order. The two sources are shuffled together (seed 2629), and modes are assigned in pool order. |

### Contamination (every request and every full user turn)

The checks listed in the original criteria apply:
- `phase24/contam.py`;
- `phase26/contam26.py` (including OR-Bench-Hard-1K);
- HarmBench (8-gram);
- the XSTest guard (5-gram or Jaccard ≥ 0.5);
- the two Phase 26C dev sets.

**Added:** any shared 8-word run, or an exact normalised match for short items, with **any** prompt in the full
OR-Bench-80K and OR-Bench-Toxic files. This covers the whole files, not only the dev samples. The sealed sets are
covered by `phase24/contam.py`. Any hit removes the row, and hits are counted per set.

### Replies: the only newly written text

Each reply is short and varied:
1. **A declining opener.** 14 phrasings, each checked against `is_refusal_v4`.
2. **One sentence saying why**, specific to the category:
   - at least 4 per PKU category group;
   - a general list for the phase23 prompts, which have no category;
   - for discrimination, a correction of the premise. The frame rows use 26B's frame-kind corrections plus new ones. The
     PKU discrimination prompts use general corrections, at least 6, because they carry no group attribute.
3. **Optionally, a safe alternative.**

**Rules:**
- Every full reply must pass `is_refusal_v4`.
- **No reply is used more than 3 times.**
- No operational detail and nothing graphic.
- The build log prints counts and example **replies** only, never requests.

### Long-passage document-mode refusal rows (built by code)

- **The user turn:** `DOC_TEMPLATE` with 3 benign Simple Wikipedia passages (BM25 top 3 for the request, chosen as in
  Phase 26), followed by the existing request.
- **Length:** only rows whose passages total ≥ 150 words are kept, and the median length is reported.
- **Passages:** the passages are contamination-checked too, as part of the full user turn.

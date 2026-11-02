# Phase 15D Report: abstention diagnostic — verdict INCONCLUSIVE

No release, no sealed sets; v1.0.1 stays the demo. The question was whether the 15B RAG failure was mainly learned
abstention or a reading limit of the 34M model, judged by the rule in `phase15/CRITERIA_15D.md`, committed
(`a0d8ec0`) before any 15D training.

## Verdict (pre-registered rule)

| Variant | SQuAD-val answerable accuracy |
| :--- | ---: |
| abstain10 | 22.1% |
| abstain0 (best) | 26.9% |

The rule says ≥ 30% → abstention, ≤ 20% on all → reading limit, otherwise **inconclusive**. The best variant
reached 26.9% and the other 22.1%, so the result is **INCONCLUSIVE**. The best variant did not clear 30%, so no
second seed was trained (step 4 did not trigger).

What the numbers do show: removing abstention recovers about **10 points** (16.5% → 26.9%). But a model that
answers *every* item gets only 26.9% right on its own training distribution, so most of the gap remains a
reading/extraction problem. Abstention was a real contributor, not the main cause by this rule's standard.

## 1. Truncation fix

`phase15/rag_format.fit_rag_prompt` trims passages (longest first) until the prompt fits. It never cuts the
question or the system line; if the question and instruction alone don't fit, it drops the passages. It is wired
into all RAG inference paths. 4 new tests; all 16 phase15 tests pass (`2d63659`). No input was cut in any 15B,
15C or 15D measurement, so earlier results are unaffected.

## 2. Variants (only the abstain targets differ; one seed each, seed 1, same hyperparameters as 15B)

| Data | RAG rows (train) | Abstain rows | Abstain share | Train rows |
| :--- | ---: | ---: | ---: | ---: |
| 15B (reference) | 39,210 | 10,555 | 27% | 147,588 |
| abstain10 | 31,839 | 3,184 (random) | 10% | 140,217 |
| abstain0 | 28,655 | 0 | 0% | 137,033 |

Training: abstain10 8,763 steps / 1,063 s; abstain0 8,564 steps / 1,059 s. Clipping 99.8–99.9% (as in every SFT
run here). Max grad norm 29.4 / 43.8. Val loss 1.95→1.667 and 1.97→1.671.

## 3. Measurements

SQuAD part of the original 15A val split (484 answerable, 215 unanswerable), exact training format, greedy +
no-repeat-3-gram, 0 inputs cut. Accuracy is the normalised gold span contained in the reply.

| Model | Answerable accuracy | Exact match | Abstain on answerable | Accuracy when it answers | Abstain on unanswerable |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 15B seed 1 (reference) | 16.5% (80/484) | 15.1% | 53.3% | 35.4% (80/226) | 68.4% |
| abstain10 | 22.1% (107/484) | 19.6% | 32.0% | 32.5% (107/329) | 47.9% |
| **abstain0** | **26.9%** (130/484) | 21.7% | 0.0% | 26.9% (130/484) | 0.0% |

**Answering more lowers precision.** As abstention falls, accuracy when the model answers drops from 35.4% to
32.5% to 26.9%. The items it used to decline are mostly ones it cannot read correctly; forcing an answer turns
abstentions into wrong spans pulled from the passage.

**abstain0 never abstains.** It also says "The sources don't say" on 0% of *unanswerable* items, which is the
expected trade-off.

**Statistical caution.** With n=484, the standard error at about 27% is about 2 points. 26.9% is about 1.5 SE
below the 30% bar. Seed variance for this metric was not measured; Phase 14 showed seed noise can be large, so a
second seed could land on either side of 30%.

### Dev with retrieval OFF (general-skill cost; safety layer kept)

| Model | Dev total /176 | Arithmetic | Harmful | Unknown | Definition | Fact | Instruction |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 14 range (v2 seeds) | 105 | 30–31 | 24–25 | 17–19 | 14–17 | 8–10 | 7 |
| 15B seed 1 | 99 | 30 | 25 | 18 | 13 | 8 | 5 |
| abstain10 | 103 | 30 | 25 | 18 | 14 | 7 | 9 |
| abstain0 | 107 | 31 | 25 | 19 | 17 | 8 | 7 |

No meaningful general-skill cost. With retrieval off, arithmetic, harmful and unknown are inside the Phase 14
ranges for both variants. Dev total 99–107 brackets v2's 105, so the 15B residual drop (99) looks like seed/data
noise rather than a systematic cost.

## 4. Ten example answers per variant

The same 10 random answerable SQuAD-val items (seed 15) for each model. Gold answer in brackets.

| Question (abridged) | 15B reference | abstain10 | abstain0 |
| :--- | :--- | :--- | :--- |
| Which tool is used in day to day life? [the clock] | abstained | abstained | wrong: "table" |
| Earliest credible source of the Hellenistic period? [Polybius] | wrong: "Achaean League" | wrong: "Achaean League" | wrong: "Achaean League" |
| Madonna's debut single? [Everybody] | **correct** | wrong: "Burning Up" | wrong: "B Turn on Hot Dance Club Songs chart" |
| How far is Tucson from Mexico? [60 mi (97 km)] | wrong: "996,544" | wrong: "520,116,544" | wrong: "996,544" |
| Goal regarding the Treaty of Schönbrunn? [the Habsburg Empire] | **correct** | **correct** | **correct** |
| Share of WashU freshmen in the top 10%? [More than 90%] | wrong: "13" | wrong: "13.7%" | wrong: "13" |
| Who managed religious and regional affairs? [Tibet] | wrong: "the Mongols" | wrong: "the Mongols" | wrong: "the Mongols" |
| New Haven's 2014 Livability.com distinction? [Best Foodie City] | abstained | wrong: "New Haven" | wrong: "56 Zagat-rated restaurants" |
| What did institutions rely more on...? [indirect information] | abstained | abstained | wrong: "international financial services" |
| Who signed 8 years / $136 million? [Alfonso Soriano] | abstained | abstained | wrong: "Dusty Baker" |

Abstentions turn into confident wrong spans taken from the passage, and number questions produce nonsense
numbers. Both are extraction errors.

## Recommended next steps (not built)

1. **Router:** never attach passages to harmful or unknown-type prompts. 15C showed the passages caused the
   collapse of refusal and abstention skills, and with retrieval off those skills are intact.
2. **Training data:** include refusal and "I don't know" rows **with passages attached**, so the model learns
   to keep refusing and abstaining when sources are present, instead of a blanket "The sources don't say".
3. **Evaluation:** build a new sealed RAG set with **human-checked gold passages** that actually state the
   answer. The 15A set is opened, and 5 of 30 hand-read gold passages didn't answer.
4. **Resolving the inconclusive verdict** needs more than one seed of abstain0, to see where seed noise puts it
   relative to 30%. Even at best, ~27% extraction on in-distribution data is far from useful, so a larger model
   is the likelier route to working RAG than further data rebalancing at 34M.

Nothing in phase12/phase14 or the v1.0 / v1.0.1 tags was changed.

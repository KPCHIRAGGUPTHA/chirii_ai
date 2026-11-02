# Phase 15D pre-registered rule (committed before any 15D training)

Question: was the 15B RAG failure mainly learned abstention, or a reading limit of the 34M model?

Measurement: SQuAD part of the 15A RAG-SFT **val** split (`phase11/data/sft_rag_val.jsonl`, the original split,
484 answerable + 215 unanswerable), exact training prompt format, no input truncation, greedy decoding with
no-repeat-3-gram. "Answerable accuracy" = the normalised gold span is contained in the normalised reply
(lowercase, punctuation and articles removed) — the same metric as Phase 15C.

Variants (one seed each, from the pretrained base, same hyperparameters as v2/15B; only the data differs):
- **abstain10**: "The sources don't say" targets cut to ~10% of RAG rows (kept rows chosen at random).
- **abstain0**: no abstain targets at all.

Reference (not a variant): 15B seed 1 = 16.5% answerable accuracy, 53.3% abstain on answerable.

Rule (applied to the two variants):
- If the **best** variant reaches **>= 30%** answerable accuracy -> abstention over-training was the main cause.
- If **all** variants are **<= 20%** -> evidence of a reading limit at 34M parameters.
- Otherwise (best variant between 20% and 30%) -> **inconclusive**.

If the best variant reaches >= 30%, a second seed of it is trained and both seeds are reported; the verdict
then requires both seeds >= 30% to say "abstention was the main cause" firmly, and is reported as
"abstention likely, one seed below 30%" otherwise.

No release, no sealed sets in this phase. v1.0.1 stays the demo.

# Chirii AI pass/fail rules

Every rule is committed before the run it judges. A rule changed after seeing a result must say so here, with the
reason, and the report must show both the old and the new verdict.

## Already fixed (21 November 2026)

| Rule | Source |
| :--- | :--- |
| A fine-tune is attempted only if plain Qwen2.5-1.5B-Instruct with memory misses the S1 pass bar | Owner decision, 21 Nov |
| A fine-tune must not lose more than **2 points** against plain Qwen on GSM8K | Owner decision, 21 Nov |
| A fine-tune must not lose more than **2 points** against plain Qwen on function calling | Owner decision, 21 Nov |
| Deployment uses Q8_0 GGUF; a 4-bit file is used only if it scores within noise of Q8_0 on the S1 eval | Owner decision, 21 Nov |
| No training data with a non-commercial licence (alpaca-cleaned, SciQ, PKU-SafeRLHF, or any other) | Owner decision, 21 Nov |

## To be proposed in S1, approved by the owner, then committed

- The S1 pass bar for plain Qwen with memory, as a percentage of the 50 tasks, with a note on noise.
- How "within noise" is measured for the 4-bit check.
- The exact GSM8K and function-calling benchmarks, subsets and settings used for the 2-point rules.

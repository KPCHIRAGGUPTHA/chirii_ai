# Phase 9 Report: Establish the Truth (CPU Work)

All numbers come from runs executed on this laptop on 2026-10-31, unless marked as recorded by an earlier phase. No GPU was used, and full training was not run.

## Summary

- **Model ranking**: on one pinned validation text, Model C is best at **2.1531 bits per byte**. Model D, from the old trainer, is worst at 2.8804.
- **Model C discrepancy**: the 2.0936 figure does not belong to Model C. It comes from a small Phase 3 character model trained on Tiny Shakespeare.
- **Model A**: the 940,800-parameter Model A checkpoint is lost. The table uses the closest available model and labels it A*.
- **Model E**: abandoned, with no local artifacts.
- **Goal baseline**: no model gave a real correct answer on any of the 202 prompts. One point scored for the SFT model is a chance keyword match.
- **Next step**: retraining both Model C and Model D is ready to run with `phase9/GPU_RUNBOOK.md`. The GPU code path itself is untested.

## Closed Items and Known Issues

| Item | Status | Resolution |
| :--- | :--- | :--- |
| Model E | **Closed** | Abandoned. No artifacts exist locally, and it will not be pursued. |
| Original Model A | **Closed** | Lost. The 940,800-parameter, vocab-1024 checkpoint is not on disk. A* remains a labelled stand-in. |
| Phase 5's 2.83 vs 4.97 gap | **Closed** | Cannot be investigated, because the data no longer exists. |
| SFT summed-loss logging | **Known, unfixed** | `phase6/training/run_full_sft_cpu.py:128` logs the sum of micro-batch losses, so its `train_loss` reads 4 times too high. Validation numbers are not affected. |

## 1. Model E Findings (Closed)

Searched: the repo, `checkpoints/phase8/`, and files named like Model E under Desktop, OneDrive Desktop and Downloads (up to 4 folders deep).

- **Run directory**: `checkpoints/phase8/model_e/` does not exist. `checkpoints/phase8/` holds only `model_d`.
- **Logs and synced files**: none found. No repo file outside this session's Phase 8E and Phase 9 work has changed since 2026-10-28.
- **What does exist**: code and design references, plus a 50-step pilot from Phase 8D in `phase8/phase8d_gpu_pilot_results.json` (14,241 tokens/sec, 0.90 GB peak VRAM on an RTX 3090). Its logged losses (27.97 to 21.66) use the old summed values, so they read 4 times too high.
- **Conclusion**: nothing local depends on the Oct 29 run, and nothing needs discarding here. I did not contact the remote machine, and this item is closed as abandoned.

## 2. Trainer Fixes

`phase9/train_v2.py` is new. `phase8/train_model_scaling.py` is untouched.

| Old trainer problem | `train_v2.py` | Test that catches it |
| :--- | :--- | :--- |
| Read data sequentially | Random positions | `test_positions_cover_the_whole_dataset` |
| Used only the first 10.24M tokens | Samples all 17,727,927 tokens | `test_positions_cover_the_whole_dataset` |
| `weight_decay=0.0` | 0.01 on Linear weights; none on biases, LayerNorm or embeddings | `test_groups_assign_each_parameter_correctly`, `test_default_weight_decay_is_nonzero` |
| Logged the sum of micro-batch losses | Logs the mean | `test_accumulated_loss_and_gradients_equal_full_batch`, `test_logged_loss_is_near_uniform_entropy_at_initialization` |

Also added: auto device detection (CUDA, then CPU), bf16 autocast on CUDA, JSON config in, JSON metrics out, atomic checkpoints, and resume. A test confirms a resumed run matches an uninterrupted one exactly, weight for weight.

**Proof the tests would have failed**: `phase9/tests/mutation_check.py` first confirms an unmodified temporary copy passes, then puts each bug back and requires a test failure. All 26 reintroduced bugs were caught: 12 for this phase's trainer and evaluation, and 14 for the Phase 8E fixes (`phase9/tests/mutation_results.json`).

**Test results**: 74 pass and 1 is skipped across `phase9/tests` and `tests/test_phase8e_evaluation.py`. The skipped test needs a GPU.

**C_v2 and D_v2 match each other, but not the old runs**. A test confirms they share all training settings, the seed and the data, and differ only in `n_layer`, `n_head` and `n_embd`. C_v2 differs from C_old in these ways:

- C_old applied weight decay to every parameter, including embeddings. `train_v2.py` excludes them, as you specified.
- C_old used `GradScaler` with default autocast; `train_v2.py` uses bf16 with no scaler.
- C_old picked its best checkpoint using 20 random batches; `train_v2.py` uses the first 64,000 validation tokens.

**Same logging bug in Phase 6**: its `train_loss` of 10.74 means about 2.68. This is known and unfixed; I changed nothing there.

### Phase 8E Fixes

Each bug fixed earlier was reintroduced in a temporary copy, and at least one test failed every time.

| Bug | Reintroduced as | Caught by |
| :--- | :--- | :--- |
| Difference-column sign | Loss C - D | `test_differences_are_d_minus_c`, `test_report_follows_the_data`, `test_results_are_consistent` |
| | Perplexity C - D | `test_differences_are_d_minus_c`, `test_results_are_consistent` |
| Peak memory handle | Handle types not declared | `test_peak_memory_is_measured` |
| | Failed lookup returns 0.0 | `test_memory_failure_returns_none_not_zero` (new) |
| | Missing value printed as a number | `test_unavailable_memory_is_not_shown_as_zero` |
| Hard-coded report text | Recommendation | `test_report_follows_the_data` |
| | Model C loss in the table | `test_report_follows_the_data` |
| | Model D loss in the table | `test_report_follows_the_data` |
| | Conclusion text | `test_report_follows_the_data`, `test_report_matches_results` |
| | Model specification table | `test_model_specifications_follow_the_data` (new) |
| Decision rule | Any improvement proceeds | `test_gate_boundary_comes_from_the_design` (new), `test_improvement_below_gate_holds_model_e` |
| | Wrong perplexity gate | `test_gate_boundary_comes_from_the_design` (new) |
| Unpinned validation hash | Entry removed | `test_validation_tokens_hash_is_pinned`, `test_changed_validation_file_aborts` (new), `test_results_are_consistent` |
| | Mismatch allowed | `test_changed_validation_file_aborts` (new) |

I added 5 tests. Four of the bugs above were caught only by new tests, so the earlier suite had real gaps. The fifth, `test_matching_validation_file_passes_and_records_hash`, keeps the hash gate from rejecting a correct file. The real checkpoint hash test can't run in a temporary copy, so the hash-gate logic is tested on temporary files.

## 3. Standard Evaluation

`phase9/eval_standard.py` scores the whole of `data/phase5d/val_corpus.txt`, tokenized per tokenizer, in non-overlapping blocks. It aborts if the shard, tokenizer, source text or a pinned checkpoint differs from its hash. Lower is better.

| Model | Parameters | Vocab | **Bits per byte** | Loss | Perplexity | Tokens evaluated |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model C | 6,613,504 | 1,024 | **2.1531** | 2.9730 | 19.55 | 1,765,632 |
| Model B | 2,890,752 | 1,024 | **2.2269** | 3.0751 | 21.65 | 1,765,632 |
| Model A* | 985,984 | 1,377 | **2.5428** | 1.7698 | 5.87 | 3,503,104 |
| Model D (old trainer) | 15,164,800 | 1,024 | **2.8804** | 3.9774 | 53.38 | 1,765,632 |

- **Model A***: the 940,800-parameter, vocab-1024 Phase 5C Model A checkpoint is lost (closed). Its directory, `checkpoints/phase5/`, holds a later run with the same layers but a 1,377-token character vocabulary. The timestamps fit an overwrite: Phase 5C history at 11:34, this checkpoint at 13:08 on 2026-09-18. I can't prove that is what happened.
- **Why bits per byte**: A*'s loss of 1.77 looks best only because it predicts one character at a time. Bits per byte corrects for that, and A* is actually third.
- **Model D**: its gap to Model C is 0.727 bits per byte. This does not measure capacity, because of the training differences above.
- **Cross-check**: over the first 2,000 blocks, Models C and D give 2.9970 and 3.9837, matching Phase 8E exactly.
- **Full output**: `phase9/eval_results/standard_eval.json` and the generated `STANDARD_EVAL.md`.

### Retrain Comparison

C_v2 and D_v2 have not been trained, so their values are pending. After the GPU session, `phase9/compare_retrain.py` fills this table from the evaluation results.

| Model | Trainer | Parameters | Bits per byte | Loss | Perplexity |
| :--- | :--- | ---: | ---: | ---: | ---: |
| C_old | `run_model_c_gpu.py` | 6,613,504 | **2.1531** | 2.9730 | 19.55 |
| C_v2 | `phase9/train_v2.py` | pending | pending | pending | pending |
| D_old | `phase8/train_model_scaling.py` | 15,164,800 | **2.8804** | 3.9774 | 53.38 |
| D_v2 | `phase9/train_v2.py` | pending | pending | pending | pending |

D_v2 - C_v2 will be the main result: the larger model against the smaller one, with the same trainer and settings.

## 4. Model C Discrepancy Explained

Your note describes 2.0936 as a Model C step-600 checkpoint. It isn't one. Model C's own checkpoint is at step 2,500.

| Value | Model | Data | Method | Checked by |
| :--- | :--- | :--- | :--- | :--- |
| **2.0936** | Phase 3 baseline: 818,176 parameters, vocab 66, step 600 (`checkpoints/best_model.pt`) | Tiny Shakespeare, last 10% | 20 random batches, characters | Reran it: 2.0857 on the whole split |
| **2.9699** | Model C, step 2,500 | FineWeb-Edu validation, BPE tokens | 20 random batches of 8 blocks (20,480 tokens), best of 50 evaluations | Reran the method with 20 seeds: mean 2.9802, SD 0.0265, range 2.9240 to 3.0184 |
| **2.9970** | Model C | Same text, first 256,000 tokens | Non-overlapping blocks (Phase 8E) | Reran it: 2.9970 |
| **2.9730** | Model C | Same text, all 1,765,632 tokens | Non-overlapping blocks | Standard value from now on |

- **2.09 vs 2.997**: different model, different text, different tokenizer. Those losses cannot be compared. Per token, one is a character and the other is about 2 bytes. Converted, the Phase 3 model scores 3.01 bits per character on Shakespeare, and Model C scores 2.15 bits per byte on FineWeb-Edu.
- **2.9699 vs 2.9970**: the same model and text, with different samples. The recorded value sits inside the normal spread of the random method, and the first 256,000 tokens happen to be a little harder than the full text.

**Phase 5's 2.83 vs 4.97 (closed)**: those values appear only in `results/phase5/PHASE5C_REPORT.md`, as the Phase 5B run. The data no longer exists, so the gap stays unexplained and will not be investigated further.

## 5. SFT and RAG-SFT Inventory

| | Phase 6 SFT | Phase 7 RAG-SFT |
| :--- | :--- | :--- |
| **Checkpoint** | `checkpoints/phase6/model_c_sft/best_model.pt` | `checkpoints/phase7/model_c_rag_sft/best_model.pt` |
| **Base model** | Model C | Phase 6 SFT |
| **Training data** | `yahma/alpaca-cleaned`: 41,404 train, 5,176 val, 5,176 test | Hand-built context items: 64 train, 8 val, 8 test |
| **Training finished** | Yes, 1,000 of 1,000 steps | Yes, 100 of 100 steps |
| **Best step** | 1,000 | 90 |
| **Recorded val loss** | 2.6666 | 2.9198 |
| **Optimizer state saved** | No | No |

Model outputs (top-1 generation, trimmed):

| Prompt | Model C | Phase 6 SFT | Phase 7 RAG-SFT |
| :--- | :--- | :--- | :--- |
| What is 2+2? | `- Students (1999) - Stephysical Students` | `1. Sure, 19, 19, 19, 19, 19` | `2 days.5 days.5 meters 2 meters` |
| What is Python? | `The Children Christians and the Christian` | `A AI, I don't have a small media` | `Oceansesses Chotosynthesizations` |
| What is the capital of France? | `The Christian Christians and the Christian` | `The United States of States and American` | `Here's Statesting Startember Storet` |
| Write one sentence about the sea. | `The first of his first his body` | `The sentence of specific species` | `Advance.Statesting Startember` |
| Name of the moon (context says Sel) | `The Christ's Christian Christians` | `The sentence of American American` | `Advance.Steps.com Photosynthesism` |

None of these outputs answer the prompt. Full details are in `phase9/sft_inventory/SFT_INVENTORY.md`.

## 6. Goal Baseline

`phase9/goal_eval/goal_prompts.jsonl` has 202 hand-written prompts. Generation is top-1 with at most 48 new tokens.

| Category | Prompts | Scoring | Model C | Phase 6 SFT | Phase 7 RAG-SFT |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Arithmetic | 32 | Exact | 0 | 0 | Not run |
| Definitions | 28 | Keyword | 0 | 0 | Not run |
| Facts | 34 | Keyword | 0 | 0 | Not run |
| Instructions | 30 | Exact, keyword | 0 | 1 (false positive) | Not run |
| Should say "I don't know" | 26 | Abstention | 0 | 0 | Not run |
| Should refuse | 26 | Refusal | 0 | 0 | Not run |
| RAG | 26 | Keyword | 0 | 0 | 0 |
| **Overall** | **202** | | **0/202** | **1/202 raw, 0 after review** | **0/26** |

- **The one point**: the SFT model answered `Complete the sequence: 2, 4, 6, 8,` with babble that happened to contain `10 million`. The scorer records it as correct, and I reviewed it by hand as a chance match.
- **Context limit**: with 48 tokens reserved for the answer, these prompts exceed the 80 tokens left: 19 for Model C, 26 for SFT, 24 for RAG-SFT. All of them are RAG prompts, so their passages can be cut off during generation. This limit makes the RAG score harder, but outputs in the other categories are babble too.
- **Added beyond the request**: I scored RAG-SFT on the RAG category only, since that is all it was trained for.
- **Full output**: `phase9/goal_eval/baseline_results.json` holds every response.

## 7. GPU Runbook Summary

`phase9/GPU_RUNBOOK.md` retrains both models one after the other. It covers building the bundle, setup, hash checks, tests, 50-step smoke tests, throughput checks, the full runs, resume, evaluation, the four-model table, and copying files back.

- **Configs**: `phase9/configs/model_c_v2.json` (6,613,504 parameters) and `model_d_v2.json` (15,164,800), each 2,500 steps x 4,096 tokens = 10,240,000 tokens, with seed 42.
- **Model C dry run**: 2 steps on CPU, loss 6.9900 then 6.9389, validation loss 6.8427, and the checkpoint reloads.
- **Model D dry run**: 2 steps on CPU, loss 7.0173 then 6.9391, validation loss 6.7948, and the checkpoint reloads.
- **Throughput**: logs tokens/sec, GPU utilization and peak memory, and warns below 10,000 tokens/sec or 50% utilization.
- **Bundle check**: I built the updated bundle in a clean folder. Its hash check and tests passed, and 2-step runs of both models completed. Windows `tar` was used for this check, on this laptop, not on Linux.

## 8. What I Could Not Verify

- **GPU path**: bf16 autocast, `nvidia-smi` utilization reading and GPU memory logging have never run, because this laptop has no GPU. The GPU test is skipped here, and the runbook runs it first.
- **Runbook on Linux**: its environment setup and commands have not been run on Linux. `pip install torch` assumes the default Linux package includes CUDA.
- **The retrain results**: C_v2 and D_v2 are pending, so the question of whether Model D beats Model C under the same trainer is still open.
- **Run-to-run variation**: the runbook trains one seed per model, so it can't measure this. A small D_v2 - C_v2 difference should be treated as inconclusive.
- **Weight decay test strength**: the old `weight_decay=0.0` bug is caught only by a test of the default value. The separate group-assignment test covers which parameters decay.

## 9. Artifact Integrity

`phase9/verify_protected.py` compared the full SHA-256 of Model C, Model D, the tokenizer, SFT and RAG-SFT against their pinned values, and all 5 match (`phase9/logs/protected_hashes.log`). Phase 9 wrote new outputs only to `phase9/`, and did not change or delete any protected artifact.

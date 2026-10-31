# Phase 10A Report: Foundation v2

All numbers come from runs on the college GPU PC (NVIDIA RTX A2000 12GB) on 2026-10-31. The long run has not been started.

## Summary

- **Stability gate**: passed on all four checks.
- **Throughput**: about 73,000 tokens/sec training and 70,000 overall, above the 40,000 target.
- **Estimated long run**: about 2.8 hours for 700M tokens.
- **Decision needed**: 99.6% of the Phase 5D validation lines were in the new train split. `train_clean.bin` removes them, and it is the recommended data for the long run.

## Data

| Set | Documents | Text bytes | Tokens | SHA-256 |
| :--- | ---: | ---: | ---: | :--- |
| Train | 666,158 | 3,184,211,064 | 763,998,802 | `a7833f231745fd998a926e0434482216a38f93e157c0dad2c3e6f3fb1ffe230a` |
| Train clean | 659,133 | 3,149,229,510 | 755,202,272 | `1906ef73e5041a5cd958f1ad2b5f14b70c09b23682b18ff63a993c593ab98458` |
| Val | 3,311 | 15,792,556 | 3,789,816 | `21829bf07dd44e89ddb44916a362cfc5df4cacb15462211782bcfaadf54eedf0` |

- **Source**: `HuggingFaceFW/fineweb-edu`, `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. The first 669,469 documents in stream order, 3,200,003,620 bytes, stopped at the target.
- **Split**: first 8 bytes of the SHA-256 of the document text, modulo 1,000, with below 5 going to val.
- **Shards**: 13 train and 13 val, listed with hashes in `manifests/text_manifest.json` and `manifests/tokens_manifest.json`. All were re-hashed against the manifest after collection.
- **Tokens**: uint16, one `<|endoftext|>` after each document, included in the token counts.
- **Dropped documents**: 0 contained a literal special token, and 0 were empty.

## Overlap with the Phase 5D Validation Text

Phase 5D's validation text also came from FineWeb-Edu `sample-10BT`. Of its 12,013 unique lines of 40+ characters, 11,961 (99.6%) were in the new train split, across 7,025 documents.

Training on that text would make the Model C comparison meaningless, so `phase10/filter_phase5d_overlap.py` builds `train_clean.bin` without those documents. That removes 8,796,530 tokens (1.15%). A rescan found no kept document with a matching line.

- `train.bin` is unchanged, and `configs/foundation_v3.json` still uses it. `configs/foundation_v3_clean.json` uses the clean file.
- The stability runs used `train.bin`. On the clean file, I ran only a 3-step start check.
- **Not removed**: the tokenizer's training sample was chosen before this was found, so it may include some of these documents. That affects the vocabulary only, not model weights.
- **Line matching**: this catches exact lines only, so reworded or re-wrapped copies would not be found.

## Tokenizer v2

| Item | Value |
| :--- | ---: |
| Vocabulary | 16,384 |
| Training sample | 300,001,394 bytes of train text, chosen by hash |
| Bytes per token on val text | **4.171** (without `<|endoftext|>`) |
| Longest token | 17 bytes |
| Special token IDs | 0 to 8 |
| Digit token IDs | 24 to 33 |
| SHA-256 (LF form) | `332f9db4368e051179e70695ff6da320a8dc16f8f17934719cd58c8a00601b98` |

- Training took 29 s. Tokenizing all shards took 251 s on 8 threads.
- Tests cover round trips (unicode, code, numbers), single-token digits and unsplit special tokens, on both a tiny tokenizer and the real one.
- Text pins now normalise CRLF to LF (`phase10/common.py`), with tests including a CRLF split across read chunks.

## Model and Trainer

- **Parameters**: 33,870,848, matching a hand count from the layer sizes.
- **Steps**: 5,341 of 131,072 tokens = 700,055,552 tokens.
- **Micro-batch**: 64 x 4 accumulation x 512. 128 ran out of memory, and 256 must divide evenly, so 64 is the largest that fits.
- **`torch.compile`**: works, and enabled. Compiled GPU runs are not bit-for-bit repeatable (two identical tiny runs differed by up to 0.00097 in loss). Uncompiled GPU runs and CPU runs were exact.
- **Initialisation**: kept from `model.py` (normal, std 0.02). `bias=True` is also kept.

| Throughput (stability runs) | Seed 1 | Seed 2 |
| :--- | ---: | ---: |
| Tokens/sec, training only | 72,988 | 73,336 |
| Tokens/sec, overall | 70,052 | 70,230 |
| Peak GPU memory allocated by PyTorch | 8.181 GB | 8.181 GB |
| Peak GPU memory reserved by PyTorch | 8.926 GB | 8.926 GB |

"Overall" includes the first-step compile, an evaluation with generation every 50 steps and the final checkpoint. `nvidia-smi` over both runs, sampled every 15 s, showed a mean of 97% GPU utilization, up to 9,971 MiB used including the desktop, 70.3 W peak, and 89 °C mean and 91 °C peak.

### Resume After a Kill

| Test | Device | Result |
| :--- | :--- | :--- |
| Tiny model, SIGKILL, resume | CPU | Loss and weights identical to an uninterrupted run |
| Tiny model, SIGKILL, resume, no compile | GPU | Loss identical to an uninterrupted run |
| Tiny model, SIGKILL, resume, compile | GPU | Within the 0.005 test tolerance |
| Full model, SIGKILL at step 18, resumed from step 15 to 30 | GPU, compile | Max loss difference from stability seed 1: 0.000018 |

`phase10/run_long.sh` was also tested on a tiny CPU model: it resumed after a kill and finished, and it refuses to retry after NaN/Inf.

## Evaluation

`phase10/eval_v3.py` scores both texts in full precision. Model C's tokens contain no CR and its byte count matches the LF size of the old text, so the old text is scored in LF form (`sha256` `6c5c425a42b6fa988afc6a081c9bb01773b449308165934c5bc3038789c6e683`).

| Text | Seed 1 | Seed 2 | Model C |
| :--- | ---: | ---: | ---: |
| New val, bits per byte (3,789,312 tokens, 15,790,584 bytes) | 1.8150 | 1.8153 | not scored |
| Phase 5D, bits per byte (839,680 tokens, 3,516,278 bytes) | 1.8127 | 1.8128 | 2.1531 (3,517,395 bytes) |

These 300-step results exercise the evaluation; they are not final model scores. The Phase 5D figures also come from runs on `train.bin`, which included the overlapping documents, though each was sampled about 0.05 times on average.

Bits per byte counts `<|endoftext|>` as 0 bytes, while its loss still counts. On the new val text this makes the score very slightly worse, not better.

## Stability Gate

Seeds 1 and 2, 300 steps each (39,321,600 tokens) on the full run's learning-rate schedule.

| Check | Limit | Seed 1 | Seed 2 | Result |
| :--- | :--- | ---: | ---: | :--- |
| Final val bits per byte, pinned subset | Differ by at most 0.03 | 1.8070 | 1.8069 | **Pass** (0.0001) |
| NaN or Inf | None | 0 | 0 | **Pass** |
| Max grad norm after warmup | At most 10 | 0.964 (step 266) | 0.902 (step 249) | **Pass** |
| Loss falls smoothly | See below | Yes | Yes | **Pass** |

Smoothly is taken to mean val loss falls at every evaluation and mean train loss falls in every 50-step window. The largest grad norm overall was at step 1 (3.844 and 3.753).

| Step | Seed 1 train loss | Seed 1 val loss | Seed 1 bits/byte | Seed 2 train loss | Seed 2 val loss | Seed 2 bits/byte |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 7.3352 | 7.3209 | 2.4921 | 7.3365 | 7.3236 | 2.4931 |
| 100 | 6.4254 | 6.4585 | 2.1986 | 6.4272 | 6.4725 | 2.2033 |
| 150 | 5.9777 | 6.0532 | 2.0606 | 5.9986 | 6.0717 | 2.0669 |
| 200 | 5.7011 | 5.7845 | 1.9691 | 5.6828 | 5.7686 | 1.9637 |
| 250 | 5.4413 | 5.5257 | 1.8810 | 5.4485 | 5.5420 | 1.8866 |
| 300 | 5.2255 | 5.3082 | 1.8070 | 5.2516 | 5.3079 | 1.8069 |

**What this does not show**: only 100 of the 300 steps come after warmup, all near peak learning rate. The two seeds were close here, but that does not settle whether they stay close over 5,341 steps. Generated text at step 300 is still repetitive and meaningless, as expected at this stage.

## Long Run

Estimate: 700,055,552 tokens at about 70,000 tokens/sec overall is **about 2.8 hours**, plus or minus GPU heat and machine load.

```bash
cd ~/chirag_work/minigpt && nohup setsid sh phase10/run_long.sh phase10/configs/foundation_v3_clean.json > /dev/null 2>&1 < /dev/null &
```

- **Log**: `phase10/runs/foundation_v3_clean.log`, with every step in `steps.jsonl` in the run folder.
- **Crashes**: it resumes from the last checkpoint, at most 30 minutes back, up to 20 attempts. It stops for good on NaN/Inf.
- **After a reboot**: it will not restart by itself. Running the same command resumes it.

# Phase 17A — Foundation v4 (124M): pre-registered criteria

Written and committed on 2026-11-04 **before any Phase 17A GPU run** (throughput probe, stability gate or long run).
Nothing below may be changed after a GPU result is seen; any deviation will be reported as a deviation.

## Model and data (fixed)

- MiniGPT architecture (`model.py` / `phase10/model_v3.py`, SDPA attention): n_layer 12, n_head 12, n_embd 768,
  block_size 1024, vocab 16,384 (tokenizer v2, SHA-256 `332f9db4…`), tied input/output embeddings, dropout 0, bias on.
- Trainer: `phase10/train_v3.py` unchanged (bf16 autocast, fused AdamW, torch.compile, timed checkpoints with exact
  resume, weights snapshots, GPU temperature logged at every evaluation).
- Train data: `phase17/data/tokens/train_3b.bin`, 2,986,805,090 tokens, SHA-256 `6293c3ab…` (Phase 10 clean train +
  Phase 17A new train; no Phase 10/17 val documents and no Phase 5D-overlap documents). Random uniform sampling of
  1,024-token windows over all train tokens (`phase9/train_v2.sample_positions`).
- Validation: the pinned Phase 10 validation subset (first 256,000 tokens of `phase10/data/tokens/val.bin`,
  SHA-256 `8ccf33a6…`), reported as loss and bits per byte, so Foundation v3 and v4 are measured on the same text.
- Every data hash above must re-check before the probe; if any fails, nothing runs.

## Optimisation (fixed)

- 524,288 tokens per optimizer step = micro_batch × grad_accum × 1,024, with micro_batch × grad_accum = 512.
- AdamW, lr 6e-4, betas (0.9, 0.95), weight decay 0.1 on Linear weights only (none on biases, LayerNorm or
  embeddings), gradient clipping 1.0, linear warmup 700 steps, cosine decay to 6e-5 at the final step.
- Seed 1 for the long run.

## Micro-batch and throughput probe (decided by rule, not by judgement)

- Candidate micro-batches, largest first: 32, 16, 8, 4 (each divides 512). For each, a short probe with the full
  config and compile on. **The micro-batch is the largest whose peak reserved GPU memory is ≤ 10.0 GiB** (about
  1.6 GiB headroom on the 11.6 GiB card) — "fits comfortably".
- **Steady tokens/s** = tokens of steps 21–60 ÷ the training time of those steps, on a 60-step probe at the chosen
  micro-batch (steps 1–20 are excluded because they include compilation and warm-up). Evaluation and checkpoint
  time is not part of this figure.

## Token budget (formula)

- budget_tokens = min(3.0 × 10⁹, 50 hours × steady tokens/s)
- max_steps = floor(budget_tokens ÷ 524,288) — whole steps only
- max_steps sets the cosine schedule for the stability gate and the long run alike.

## Stability gate

Two runs, seeds 1 and 2, each with the **full** final config (same micro-batch, accumulation, max_steps and so the
same learning-rate schedule), stopped after **150 steps**, evaluating on the validation subset every **25 steps**
(evaluations at steps 25, 50, 75, 100, 125, 150).

| Gate | Passes when |
| :--- | :--- |
| G1 seed agreement | \|val bits per byte at step 150, seed 1 − seed 2\| ≤ 0.03 |
| G2 finite | no NaN or Inf training loss or gradient norm at any step of either run |
| G3 gradients | no gradient norm > 10 at any step after step 50, in either run |
| G4 learning | validation loss strictly lower at every evaluation than at the previous one, in both runs |

## Decision

- **All four gates pass:** launch the long run automatically (seed 1, fresh start, full budget, evaluation every 250
  steps, weights snapshot every 2,000 steps, checkpoint every 30 minutes, auto-resume), detached with setsid/nohup,
  with a `@reboot` crontab entry that resumes only this run.
- **Any gate fails:** stop, launch nothing, report.

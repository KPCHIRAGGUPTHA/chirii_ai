# Phase 9 GPU Runbook: Model C and Model D Retrains

Retrains both Model C and Model D with `phase9/train_v2.py`, using the same seed and settings at their original architectures and a 10.24M token budget. Both are then scored with `phase9/eval_standard.py`.

Replace `USER@GPU_HOST` with your machine's login. All commands run from the project directory on the machine named in each step.

## 1. Make the bundle (Windows laptop, PowerShell)

Run this from the `minigpt` folder:

```powershell
tar -czf phase9_gpu_bundle.tar.gz model.py bpe_tokenizer.py phase9/__init__.py phase9/train_v2.py phase9/eval_standard.py phase9/compare_retrain.py phase9/eval_config.json phase9/configs phase9/tests/test_train_v2.py phase9/tests/test_model_configs.py phase9/tests/test_compare_retrain.py phase9/data/train_v1024.bin phase9/data/val_v1024.bin phase9/data/SHA256SUMS phase9/eval_results/standard_eval.json tokenizers/phase5d/bpe_vocab_1024.json data/phase5d/val_corpus.txt
```

```powershell
scp phase9_gpu_bundle.tar.gz USER@GPU_HOST:~/
```

The token files are not in git, so use this bundle rather than `git clone`.

## 2. Set up the environment (GPU machine, Linux)

```bash
mkdir -p ~/minigpt && tar -xzf ~/phase9_gpu_bundle.tar.gz -C ~/minigpt && cd ~/minigpt
```

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install torch numpy pytest
```

Check the data arrived intact. This must print `OK` for `train_v1024.bin` and `val_v1024.bin`. The bundle leaves out the v1377 shard because these runs do not use it:

```bash
sha256sum --ignore-missing -c phase9/data/SHA256SUMS
```

Check the GPU is visible. This must print `True True`:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported()); print(torch.__version__, torch.cuda.get_device_name(0))"
```

Run the tests. On this machine the GPU test runs too; it was skipped on the laptop, which has no GPU:

```bash
python -m pytest phase9/tests/test_train_v2.py phase9/tests/test_model_configs.py phase9/tests/test_compare_retrain.py -q
```

Stop here if anything fails.

## 3. Run the 50-step smoke tests

```bash
python phase9/train_v2.py --config phase9/configs/model_c_v2.json --max-steps 50 --out-dir phase9/runs/model_c_v2_smoke
```

```bash
python phase9/train_v2.py --config phase9/configs/model_d_v2.json --max-steps 50 --out-dir phase9/runs/model_d_v2_smoke
```

Check all of these for both runs before continuing:

- The first line says `Autocast: bf16` and names the GPU, not `cpu`.
- The step 1 loss is about 7. This is the uniform-guess value, `ln(1024)` = 6.93. A loss near 28 would mean the summing bug is back.
- The loss is lower at step 50 than at step 1.
- `metrics.json` and `checkpoint.pt` exist in each smoke directory.

## 4. Check throughput

The trainer logs tokens/sec, GPU utilization and peak GPU memory every 50 steps. It prints `[THROUGHPUT WARNING]` below 10,000 tokens/sec or below 50% utilization, and records warnings in `metrics.json`.

The 10,000 tokens/sec floor is deliberately low. Earlier recorded runs reached 28,849 tokens/sec for Model C and 15,263 for Model D, so a data-center GPU should beat it comfortably.

To see what the GPU is doing during a run, open a second terminal:

```bash
nvidia-smi dmon -s um -d 10
```

Both configs keep a micro-batch of 8 with 4 accumulation steps, matching the original runs. This setup can keep utilization below what the GPU could reach, and that does not affect results. A warning needs action only if a run would take unreasonably long: at 10,000 tokens/sec each run takes about 17 minutes. If you see one, check that the data is on local disk rather than a network mount, and that no other process is using the GPU. Do not change the batch settings, or the two runs stop being comparable.

## 5. Run both full trainings

Run them one at a time, so they don't compete for the GPU. Model D starts only if Model C succeeds:

```bash
nohup sh -c "python phase9/train_v2.py --config phase9/configs/model_c_v2.json > phase9/model_c_v2_train.log 2>&1 && python phase9/train_v2.py --config phase9/configs/model_d_v2.json > phase9/model_d_v2_train.log 2>&1" > /dev/null 2>&1 &
```

```bash
tail -f phase9/model_c_v2_train.log
```

```bash
tail -f phase9/model_d_v2_train.log
```

If a run is interrupted, resume it from its last checkpoint (saved every 100 steps). Use the matching model's config and log. For example, for Model D:

```bash
nohup python phase9/train_v2.py --config phase9/configs/model_d_v2.json --resume >> phase9/model_d_v2_train.log 2>&1 &
```

Confirm both finished. This must print `completed 2500` twice:

```bash
python -c "import json; [print(m['status'], m['completed_steps'], 'warnings:', len(m['warnings'])) for m in (json.load(open(f'phase9/runs/model_{x}_v2/metrics.json')) for x in 'cd')]"
```

## 6. Evaluate both with the standard evaluation

```bash
python phase9/eval_standard.py --checkpoint phase9/runs/model_c_v2/best_model.pt --name "C_v2" --shard v1024 --note "Model C retrained with phase9/train_v2.py."
```

```bash
python phase9/eval_standard.py --checkpoint phase9/runs/model_d_v2/best_model.pt --name "D_v2" --shard v1024 --note "Model D retrained with phase9/train_v2.py."
```

Build the C_old, C_v2, D_old and D_v2 table:

```bash
python phase9/compare_retrain.py
```

**D_v2 - C_v2** is the main result: the larger model against the smaller one, with the same trainer and settings. One seed per model gives no measure of run-to-run variation, so treat a small difference as inconclusive.

## 7. Copy the results back (Windows laptop, PowerShell)

```powershell
mkdir -Force phase9/runs/model_c_v2, phase9/runs/model_d_v2
```

```powershell
scp USER@GPU_HOST:~/minigpt/phase9/runs/model_c_v2/best_model.pt USER@GPU_HOST:~/minigpt/phase9/runs/model_c_v2/metrics.json USER@GPU_HOST:~/minigpt/phase9/model_c_v2_train.log phase9/runs/model_c_v2/
```

```powershell
scp USER@GPU_HOST:~/minigpt/phase9/runs/model_d_v2/best_model.pt USER@GPU_HOST:~/minigpt/phase9/runs/model_d_v2/metrics.json USER@GPU_HOST:~/minigpt/phase9/model_d_v2_train.log phase9/runs/model_d_v2/
```

```powershell
scp USER@GPU_HOST:~/minigpt/phase9/eval_results/standard_eval.json USER@GPU_HOST:~/minigpt/phase9/eval_results/STANDARD_EVAL.md USER@GPU_HOST:~/minigpt/phase9/eval_results/RETRAIN_COMPARISON.md phase9/eval_results/
```

Each checkpoint includes the optimizer state: Model D's is about 183 MB, and Model C's is smaller. Copy `checkpoint.pt` too only if you plan to continue training. `phase9/runs/` is ignored by git, so these checkpoints won't be committed by accident.

## Files

| File | Where it comes from | Purpose |
| :--- | :--- | :--- |
| `phase9/data/*.bin` | `phase9/prepare_data.py` on the laptop | Tokens opened with `np.memmap` |
| `phase9/configs/model_c_v2.json`, `model_d_v2.json` | Repo | Architecture, budget and pinned data hashes |
| `phase9/runs/model_*_v2/best_model.pt` | Training | Weights to evaluate |
| `phase9/runs/model_*_v2/metrics.json` | Training | Loss, throughput, GPU memory and warnings |
| `phase9/eval_results/RETRAIN_COMPARISON.md` | `phase9/compare_retrain.py` | The four-model table |

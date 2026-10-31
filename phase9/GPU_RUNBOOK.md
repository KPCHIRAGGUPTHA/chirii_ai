# Phase 9 GPU Runbook: Two-Seed Model C and Model D Retrains

Retrains Model C and Model D with `phase9/train_v2.py` at their original architectures and a 10.24M token budget, with two seeds each. The order is C/42, D/42, C/43, D/43. Each run is then scored with `phase9/eval_standard.py` and goal eval v2.

**The result rule**: D counts as better only if both D seeds beat both C seeds in bits per byte. `phase9/compare_retrain.py` applies this rule.

Replace `USER@GPU_HOST` with your machine's login. All commands run from the project directory on the machine named in each step.

## 1. Make the bundle (Windows laptop, PowerShell)

From the `minigpt` folder:

```powershell
python phase9/build_bundle.py
```

Note the printed SHA-256, then copy the bundle over:

```powershell
scp phase9/dist/phase9_gpu_bundle.tar.gz USER@GPU_HOST:~/
```

The token files are not in git, so use this bundle rather than `git clone`.

## 2. Set up the environment (GPU machine, Linux)

Check the SHA-256 matches the one printed on the laptop:

```bash
sha256sum ~/phase9_gpu_bundle.tar.gz
```

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

Run the tests. On this machine the GPU test runs too; it was skipped on the laptop, which has no GPU. One Phase 6 format test is skipped because that code is not in the bundle:

```bash
python -m pytest phase9/tests -q
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

All configs keep a micro-batch of 8 with 4 accumulation steps, matching the original runs. This setup can keep utilization below what the GPU could reach, and that does not affect results. A warning needs action only if a run would take unreasonably long: at 10,000 tokens/sec each run takes about 17 minutes. If you see one, check that the data is on local disk rather than a network mount, and that no other process is using the GPU. Do not change the batch settings, or the runs stop being comparable.

## 5. Run all four trainings

Run them one at a time, in this order, so they don't compete for the GPU. Each run starts only if the one before it succeeds:

```bash
nohup sh -c "python phase9/train_v2.py --config phase9/configs/model_c_v2.json > phase9/c_v2_s42_train.log 2>&1 && python phase9/train_v2.py --config phase9/configs/model_d_v2.json > phase9/d_v2_s42_train.log 2>&1 && python phase9/train_v2.py --config phase9/configs/model_c_v2_seed43.json > phase9/c_v2_s43_train.log 2>&1 && python phase9/train_v2.py --config phase9/configs/model_d_v2_seed43.json > phase9/d_v2_s43_train.log 2>&1" > /dev/null 2>&1 &
```

Watch progress in any of the four logs:

```bash
tail -f phase9/c_v2_s42_train.log
```

If a run is interrupted, resume it from its last checkpoint (saved every 100 steps) with its own config and log, then start the remaining runs in order. For example, for D with seed 42:

```bash
nohup python phase9/train_v2.py --config phase9/configs/model_d_v2.json --resume >> phase9/d_v2_s42_train.log 2>&1 &
```

Confirm all four finished. This must print `completed 2500` four times:

```bash
python -c "import json; [print(d, m['status'], m['completed_steps'], 'warnings:', len(m['warnings'])) for d, m in ((d, json.load(open(f'phase9/runs/{d}/metrics.json'))) for d in ['model_c_v2', 'model_d_v2', 'model_c_v2_seed43', 'model_d_v2_seed43'])]"
```

## 6. Score with the standard evaluation

```bash
python phase9/eval_standard.py --checkpoint phase9/runs/model_c_v2/best_model.pt --name "C_v2_s42" --shard v1024 --note "Model C retrained with train_v2.py, seed 42."
```

```bash
python phase9/eval_standard.py --checkpoint phase9/runs/model_d_v2/best_model.pt --name "D_v2_s42" --shard v1024 --note "Model D retrained with train_v2.py, seed 42."
```

```bash
python phase9/eval_standard.py --checkpoint phase9/runs/model_c_v2_seed43/best_model.pt --name "C_v2_s43" --shard v1024 --note "Model C retrained with train_v2.py, seed 43."
```

```bash
python phase9/eval_standard.py --checkpoint phase9/runs/model_d_v2_seed43/best_model.pt --name "D_v2_s43" --shard v1024 --note "Model D retrained with train_v2.py, seed 43."
```

## 7. Score with goal eval v2

These are pretrained models with no instruction training, so expect scores near 0%. Each run also writes a CSV to `phase9/goal_eval/results_v2/` for spot checks.

```bash
python phase9/goal_eval/scorer_v2.py --checkpoint phase9/runs/model_c_v2/best_model.pt --name "C_v2_s42"
```

```bash
python phase9/goal_eval/scorer_v2.py --checkpoint phase9/runs/model_d_v2/best_model.pt --name "D_v2_s42"
```

```bash
python phase9/goal_eval/scorer_v2.py --checkpoint phase9/runs/model_c_v2_seed43/best_model.pt --name "C_v2_s43"
```

```bash
python phase9/goal_eval/scorer_v2.py --checkpoint phase9/runs/model_d_v2_seed43/best_model.pt --name "D_v2_s43"
```

## 8. Build the comparison

```bash
python phase9/compare_retrain.py
```

This shows the mean and both individual bits-per-byte values per model, the goal scores, and the verdict. If the seeds overlap, the verdict is `D NOT BETTER`, even if D's mean is lower.

## 9. Copy the results back (Windows laptop, PowerShell)

Pack everything on the GPU machine first. Checkpoints are about 183 MB each for Model D and smaller for Model C, and include optimizer state:

```bash
tar -czf ~/phase9_gpu_results.tar.gz phase9/runs/model_c_v2/best_model.pt phase9/runs/model_c_v2/metrics.json phase9/runs/model_d_v2/best_model.pt phase9/runs/model_d_v2/metrics.json phase9/runs/model_c_v2_seed43/best_model.pt phase9/runs/model_c_v2_seed43/metrics.json phase9/runs/model_d_v2_seed43/best_model.pt phase9/runs/model_d_v2_seed43/metrics.json phase9/c_v2_s42_train.log phase9/d_v2_s42_train.log phase9/c_v2_s43_train.log phase9/d_v2_s43_train.log phase9/eval_results phase9/goal_eval/results_v2
```

Then on the laptop, from the `minigpt` folder:

```powershell
scp USER@GPU_HOST:~/phase9_gpu_results.tar.gz .
```

```powershell
tar -xzf phase9_gpu_results.tar.gz
```

This overwrites `phase9/eval_results` and `phase9/goal_eval/results_v2` with the versions that include the new runs. `phase9/runs/` is ignored by git, so these checkpoints won't be committed by accident.

## Files

| File | Where it comes from | Purpose |
| :--- | :--- | :--- |
| `phase9/dist/phase9_gpu_bundle.tar.gz` | `phase9/build_bundle.py` | Everything the GPU machine needs |
| `phase9/configs/model_{c,d}_v2.json` | Repo | Seed 42 configs |
| `phase9/configs/model_{c,d}_v2_seed43.json` | Repo | Seed 43 configs |
| `phase9/runs/*/best_model.pt` | Training | Weights to evaluate |
| `phase9/runs/*/metrics.json` | Training | Loss, throughput, GPU memory and warnings |
| `phase9/eval_results/RETRAIN_COMPARISON.md` | `phase9/compare_retrain.py` | The final table and verdict |
| `phase9/goal_eval/results_v2/*.csv` | `phase9/goal_eval/scorer_v2.py` | Responses for spot checks |

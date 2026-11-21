# GPU job: Qwen2.5-7B diagnostic (no decision attached)

Runs the S3 "ship" setup (run (b): trimmed retrieval with the answer fact-check guard) with **Qwen2.5-7B-Instruct
Q8_0** instead of 1.5B, on the 50 development tasks and the 5 Hinglish ones. It shows how much of the 1.5B gap a
larger model closes. It changes no bar and no verdict (`chirii/CRITERIA.md`).

- **Model:** the official `Qwen/Qwen2.5-7B-Instruct-GGUF` repository on Hugging Face, Q8_0 file(s), Apache-2.0. The
  script downloads them if missing, stops if the licence file is not Apache-2.0, and records each file's SHA-256 and
  the repository revision.
- **Engine:** llama.cpp **b12359**, the same build as on the laptop, compiled with CUDA so the whole model sits on the
  GPU. Q8_0 7B needs about 8.5 GB of the A2000's 12 GB.

Run it alongside the benchmark job (`chirii/gpu/benchmarks/RUN_ON_GPU.md`), but not at the same moment: each needs
most of the GPU. Run one, then the other.

## One-time setup (about 20–30 minutes)

**1. Build llama.cpp b12359 with CUDA.** This needs the CUDA toolkit (`nvcc --version` should work) and cmake.

```bash
git clone --depth 1 --branch b12359 https://github.com/ggml-org/llama.cpp ~/chirag_work/llama.cpp-b12359
```

```bash
cd ~/chirag_work/llama.cpp-b12359 && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j --target llama-server
```

If `nvcc` is missing and can't be installed, build without `-DGGML_CUDA=ON`. It then runs on the CPU, which is slow
but works; `environment.txt` records which build was used.

**2. Create the app's Python environment** in the repository (Python 3.12, the pinned `requirements.txt`).

```bash
cd ~/chirag_work/minigpt && git checkout chirii-poc && git pull
```

```bash
pip install uv && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

The script also downloads the embedding model (all-MiniLM-L6-v2, about 90 MB) on first use.

## Run (about 10–15 minutes after the download; the first run downloads about 8 GB)

```bash
cd ~/chirag_work/minigpt && git pull
```

```bash
nohup bash chirii/gpu/qwen7b/run_7b.sh > /dev/null 2>&1 &
```

Watch progress:

```bash
tail -f chirii/results/gpu_7b/run.log
```

If llama-server or Python are somewhere else, set `LLAMA_SERVER=/path/to/llama-server` or `PY=/path/to/python`
before `bash`. If the model is already downloaded elsewhere, set `MODEL_DIR=/path/to/folder`.

## What to commit

```bash
git add chirii/results/gpu_7b/replies_b.jsonl chirii/results/gpu_7b/auto_b.json chirii/results/gpu_7b/RESULTS_7B.md chirii/results/gpu_7b/environment.txt chirii/results/gpu_7b/run.log
```

```bash
git commit -m "GPU diagnostic: Qwen2.5-7B in run (b) setup" && git push
```

Don't commit `server.log`, which is large and adds nothing. Never commit the model files (`models/` is ignored by git).

## What happens next

On the laptop, every reply is read by hand, as in S3, and the result is reported next to the 1.5B run (b) with the
task-by-task comparison in `RESULTS_7B.md`. Both are marked with the current marker.

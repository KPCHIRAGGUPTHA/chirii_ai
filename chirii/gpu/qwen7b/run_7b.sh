#!/usr/bin/env bash
# GPU diagnostic (no decision attached, CRITERIA.md): the S3 run (b) setup (trimmed retrieval + fact-check guard) with
# Qwen2.5-7B-Instruct Q8_0 instead of 1.5B, on the 50 development tasks plus the 5 Hinglish ones.
# Instructions: chirii/gpu/qwen7b/RUN_ON_GPU.md. Writes chirii/results/gpu_7b/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
OUT="chirii/results/gpu_7b"
MODEL_DIR="${MODEL_DIR:-$REPO/models/qwen7b}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/chirag_work/llama.cpp-b12359/build/bin/llama-server}"
PY="${PY:-$REPO/.venv/bin/python}"
PORT="${PORT:-8081}"
mkdir -p "$OUT"
LOG="$OUT/run.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "Checking the setup"
[ -x "$PY" ] || { say "No Python env at $PY. Create it as in RUN_ON_GPU.md, or set PY=/path/to/python."; exit 1; }
[ -x "$LLAMA_SERVER" ] || { say "No llama-server at $LLAMA_SERVER. Build it as in RUN_ON_GPU.md, or set LLAMA_SERVER=..."; exit 1; }
"$PY" -c "import chirii.eval.run_s3" || { say "The Python env is missing packages: run the pip install step in RUN_ON_GPU.md."; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG" || say "nvidia-smi not found: running without a GPU will be slow"

if ! ls "$MODEL_DIR"/qwen2.5-7b-instruct-q8_0*.gguf >/dev/null 2>&1; then
  say "Downloading Qwen/Qwen2.5-7B-Instruct-GGUF Q8_0 (about 8 GB) from the official Qwen repository"
  "$PY" - "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct-GGUF", allow_patterns=["qwen2.5-7b-instruct-q8_0*.gguf", "LICENSE"], local_dir=sys.argv[1])
EOF
fi
grep -q "Apache License" "$MODEL_DIR/LICENSE" || { say "The model's LICENSE is not Apache-2.0 as expected: stopping."; exit 1; }
FIRST="$(ls "$MODEL_DIR"/qwen2.5-7b-instruct-q8_0*.gguf | sort | head -n 1)"

say "Recording the environment"
{
  echo "date: $(date -Iseconds)"
  echo "repo commit: $(git rev-parse HEAD)"
  echo "llama-server: $("$LLAMA_SERVER" --version 2>&1 | tr '\n' ' ')"
  echo "model repo revision: $("$PY" -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen2.5-7B-Instruct-GGUF').sha)" 2>/dev/null || echo unknown)"
  echo "model files (sha256):"; sha256sum "$MODEL_DIR"/qwen2.5-7b-instruct-q8_0*.gguf
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$OUT/environment.txt" 2>&1

say "Starting llama-server on port $PORT with all layers on the GPU"
"$LLAMA_SERVER" -m "$FIRST" --host 127.0.0.1 --port "$PORT" -c 4096 -ngl 99 --alias qwen2.5-7b-instruct-q8_0 > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 2; done
curl -s "http://127.0.0.1:$PORT/health" | grep -q ok || { say "llama-server did not start: see $OUT/server.log"; exit 1; }

say "Running the 55 tasks in run (b)'s setup with the 7B model"
CHIRII_LLM_BASE_URL="http://127.0.0.1:$PORT/v1" CHIRII_LLM_MODEL="qwen2.5-7b-instruct-q8_0" \
  "$PY" -m chirii.eval.run_s3 --runs b --out-dir "$OUT" 2>&1 | tee -a "$LOG"

say "Comparing with the 1.5B run (b)"
"$PY" -m chirii.eval.compare_runs --base chirii/results/s3/replies_b.jsonl --base-name "1.5B (b)" \
  --other "$OUT/replies_b.jsonl" --other-name "7B (b)" --out "$OUT/RESULTS_7B.md" 2>&1 | tee -a "$LOG"
say "Done. Commit the files listed in RUN_ON_GPU.md."

"""
Phase 9 — Builds the GPU bundle at phase9/dist/phase9_gpu_bundle.tar.gz, and prints its size and SHA-256.
File order and timestamps are fixed, so the same inputs give the same SHA-256.

Usage:
    python phase9/build_bundle.py
"""

import gzip
import hashlib
import io
import os
import sys
import tarfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = "phase9/dist/phase9_gpu_bundle.tar.gz"

FILES = [
    "model.py",
    "bpe_tokenizer.py",
    "phase9/__init__.py",
    "phase9/train_v2.py",
    "phase9/eval_standard.py",
    "phase9/compare_retrain.py",
    "phase9/eval_config.json",
    "phase9/GPU_RUNBOOK.md",
    "phase9/configs/model_c_v2.json",
    "phase9/configs/model_d_v2.json",
    "phase9/configs/model_c_v2_seed43.json",
    "phase9/configs/model_d_v2_seed43.json",
    "phase9/goal_eval/scorer_v2.py",
    "phase9/goal_eval/goal_prompts_v2.jsonl",
    "phase9/goal_eval/goal_set_v2_meta.json",
    "phase9/goal_eval/results_v2/goal_eval_v2.json",
    "phase9/tests/test_train_v2.py",
    "phase9/tests/test_model_configs.py",
    "phase9/tests/test_compare_retrain.py",
    "phase9/tests/test_scorer_v2.py",
    "phase9/tests/scorer_fixtures.py",
    "phase9/data/train_v1024.bin",
    "phase9/data/val_v1024.bin",
    "phase9/data/SHA256SUMS",
    "phase9/eval_results/standard_eval.json",
    "tokenizers/phase5d/bpe_vocab_1024.json",
    "data/phase5d/val_corpus.txt",
]


def build(output_path: str = OUTPUT_PATH, files=None, root: str = PROJECT_ROOT) -> str:
    full_output = os.path.join(root, output_path)
    os.makedirs(os.path.dirname(full_output), exist_ok=True)
    with open(full_output, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for name in sorted(files or FILES):
                    with open(os.path.join(root, name), "rb") as f:
                        content = f.read()
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    info.mtime = 0
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(content))
    return full_output


def main():
    missing = [name for name in FILES if not os.path.exists(os.path.join(PROJECT_ROOT, name))]
    if missing:
        print(f"Missing files: {missing}")
        sys.exit(1)
    path = build()
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{OUTPUT_PATH}")
    print(f"size_bytes {os.path.getsize(path)}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()

"""
Phase 9 — Re-verifies the full SHA-256 of every protected artifact. Exits non-zero on any mismatch.

Usage:
    python phase9/verify_protected.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase9.train_v2 import project_path, sha256_file

PROTECTED = {
    "Model C": ("checkpoints/phase5d/model_6_61m/best_model.pt", "6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433"),
    "Model D": ("checkpoints/phase8/model_d/best_model.pt", "2c3c662b99485b5ce93902cde37ef23f811e70620e1cba8c1af60dc088b0596c"),
    "Tokenizer": ("tokenizers/phase5d/bpe_vocab_1024.json", "6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d"),
    "SFT": ("checkpoints/phase6/model_c_sft/best_model.pt", "14f9335aa66d7168500c10cebaa20db373531cf87ae9dcd7d8ee1d8ed01cfb26"),
    "RAG-SFT": ("checkpoints/phase7/model_c_rag_sft/best_model.pt", "f3f21991d1262ed3bb6602b6c3d6abe609241a6c4d40c0b22a9c44b6d20d1ed3"),
}


def main():
    all_passed = True
    for name, (path, expected) in PROTECTED.items():
        actual = sha256_file(project_path(path))
        passed = actual == expected
        all_passed = all_passed and passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {path} {actual}", flush=True)
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

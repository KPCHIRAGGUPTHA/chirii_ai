"""
Turns text into vectors (lists of numbers) so notes can be found by meaning, not only by exact words.

FastEmbedder runs all-MiniLM-L6-v2 through fastembed (ONNX, no PyTorch). HashEmbedder is a tiny stand-in for tests:
it needs no download and gives the same answer every time, but it only matches shared words and word pieces.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np

DIM = 384


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Returns one unit-length float32 row per text."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FastEmbedder:
    dim = DIM

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        from fastembed import TextEmbedding  # imported here so tests don't need the model files

        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.array(list(self._model.embed(list(texts)))))


class HashEmbedder:
    """Deterministic test embedder: hashed words and three-letter word pieces."""

    dim = DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for text in texts:
            row = np.zeros(self.dim, dtype=np.float32)
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                pieces = [word] + [word[i:i + 3] for i in range(max(1, len(word) - 2))]
                for piece in pieces:
                    bucket = int(hashlib.md5(piece.encode()).hexdigest(), 16) % self.dim
                    row[bucket] += 1.0
            rows.append(row)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack(rows))

"""
Phase 11C — Loop rate. A reply counts as a loop if it hit the token limit, repeats a whole sentence or line of 3+ words, or uses any 4-word run 3+ times.
"""

import re
from collections import Counter


def is_loop(final: str, stop: str) -> bool:
    if stop == "max_tokens":
        return True
    pieces = [" ".join(re.findall(r"[a-z0-9']+", piece.lower())) for piece in re.split(r"(?<=[.!?])\s+|\n+", final)]
    counts = Counter(piece for piece in pieces if len(piece.split()) >= 3)
    if counts and max(counts.values()) >= 2:
        return True
    words = re.findall(r"[a-z0-9']+", final.lower())
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return bool(grams) and max(grams.values()) >= 3

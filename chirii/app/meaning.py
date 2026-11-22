"""
S4f: matching a question's words to a memory by meaning, not only by exact spelling.

A key term from the question ("wedding") matches a memory when one of the memory's words:
  1. is the same word, or shares a beginning of 3+ letters (the S4c rule: GST and GSTR-3B);
  2. has the same stem (married / marriage / marry; engaged / engagement; renew / renewal);
  3. means the same through a small Indian synonym map (shaadi = wedding, dawai / tablet = medicine, kiraya = rent);
  4. is close in meaning by the embedding model (all-MiniLM-L6-v2) above THRESHOLD.

Names (Neha, Infosys) only use 1 and 2: a name has no "meaning" to compare. Dates, times and amounts are never matched
here; they keep their exact rules elsewhere.

THRESHOLD was calibrated on my own word pairs (chirii/eval/calibrate_meaning.py, results in
chirii/results/meaning_calibration.json), not on any development set. It is set just above the most similar pair that
should NOT match, because a false match gives a confident answer about the wrong item, which is worse than saying
"I have X, but nothing about Y".
"""

import re
from contextvars import ContextVar
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

THRESHOLD = 0.66

# Hindi / Hinglish / Tamil words a user may type in English letters, and the English family word they mean.
# Kept deliberately small and general: no entries were added for any development-set task.
INDIAN_SYNONYMS = {
    "shaadi": "wedding", "shadi": "wedding", "vivah": "wedding", "byah": "wedding", "kalyanam": "wedding",
    "marriage": "wedding", "marry": "wedding", "married": "wedding", "marrying": "wedding",
    "dawai": "medicine", "dawa": "medicine", "davai": "medicine", "dava": "medicine", "goli": "medicine", "tablet": "medicine",
    "tablets": "medicine", "medication": "medicine", "medications": "medicine", "meds": "medicine", "marunthu": "medicine",
    "kiraya": "rent", "bhada": "rent", "bijli": "electricity","paani": "water", "doodh": "milk",
    "chashma": "glasses", "naukri": "job", "dukaan": "shop", "dukan": "shop", "aspatal": "hospital", "daaktar": "doctor",
    "dr": "doctor", "doctors": "doctor", "bp": "pressure", "checkup": "check", "physio": "physiotherapy",
}

SUFFIXES = ("ments", "ment", "ings", "ing", "ages", "age", "ations", "ation", "als", "al", "ies", "ied", "es", "ed", "s")


@lru_cache(maxsize=4096)
def stem(word: str) -> str:
    """A light stem: one common ending off, then a final e / i / y (engaged, engagement, engage → engag;
    married, marriage, marry → marr; renewal, renews → renew). Words under 5 letters are left alone."""
    w = word.lower()
    if len(w) < 5:
        return w
    for suffix in SUFFIXES:
        if suffix == "s" and w.endswith("ss"):
            break                                   # class, glass, address keep their "ss"
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            w = w[: -len(suffix)]
            break
    return w.rstrip("eiy") if len(w.rstrip("eiy")) >= 4 else w


@lru_cache(maxsize=4096)
def canon(word: str) -> str:
    """The word's family: the synonym map first (on the word, then on its stem), then the stem."""
    w = word.lower()
    if w in INDIAN_SYNONYMS:
        return stem(INDIAN_SYNONYMS[w])
    s = stem(w)
    for key, value in INDIAN_SYNONYMS.items():
        if stem(key) == s:
            return stem(value)
    return s


def same_family(a: str, b: str) -> bool:
    return canon(a) == canon(b)


class Meaning:
    """Embedding similarity between single words, cached (the same words come up again and again)."""

    def __init__(self, embedder, threshold: float = THRESHOLD):
        self.embedder = embedder
        self.threshold = threshold
        self._cache: Dict[str, np.ndarray] = {}

    def _vectors(self, words: Iterable[str]) -> None:
        new = [w for w in dict.fromkeys(words) if w not in self._cache]
        if new:
            for w, v in zip(new, self.embedder.embed(new)):
                self._cache[w] = v

    def similarity(self, a: str, b: str) -> float:
        self._vectors([a, b])
        return float(self._cache[a] @ self._cache[b])

    def best(self, term: str, words: Iterable[str]) -> Tuple[Optional[str], float]:
        words = [w for w in words if len(w) >= 3]
        if not words:
            return None, 0.0
        self._vectors([term, *words])
        scores = [(w, float(self._cache[term] @ self._cache[w])) for w in words]
        return max(scores, key=lambda x: x[1])

    def close(self, term: str, words: Iterable[str]) -> bool:
        return self.best(term, words)[1] >= self.threshold


# The Assistant sets this for the length of one question, so the router's pure functions can use the embedder
# without every caller passing it along. Unset (tests, scripts), matching is by spelling, stems and the synonym map.
current: ContextVar[Optional[Meaning]] = ContextVar("chirii_meaning", default=None)
_shared: Dict[int, Meaning] = {}


def for_embedder(embedder) -> Optional[Meaning]:
    """One cached Meaning per embedder. Only the embedding model the threshold was calibrated on gets one; the test
    HashEmbedder (shared word pieces, not meaning) gets none, so tests match by spelling, stems and the map only."""
    from chirii.app.memory.embeddings import FastEmbedder
    if not isinstance(embedder, FastEmbedder):
        return None
    if id(embedder) not in _shared:
        _shared[id(embedder)] = Meaning(embedder)
    return _shared[id(embedder)]

STOP = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "had", "will", "was", "are", "is", "on", "of",
        "to", "in", "at", "by", "my", "our", "your", "every", "each", "after", "before", "said", "goes", "falls", "due",
        "new", "one", "all", "any", "her", "his", "their", "she", "him", "they", "them"}


def content_words(text: str):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]

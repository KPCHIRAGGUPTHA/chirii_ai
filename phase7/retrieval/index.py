import os
import json
from collections import Counter
from typing import List, Dict, Tuple, Any, Union, Optional

from bpe_tokenizer import BPETokenizer
from phase7.ingestion.chunker import Chunk

DEFAULT_TOKENIZER_PATH = os.path.join("tokenizers", "phase5d", "bpe_vocab_1024.json")

class InvertedIndex:
    """
    Deterministic inverted index storing subword token postings and corpus statistics
    for BM25 lexical retrieval over Phase 7 micro-chunks.
    """

    def __init__(self):
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.postings: Dict[int, List[Tuple[str, int]]] = {}  # term_id -> [(chunk_id, tf)]
        self.doc_lengths: Dict[str, int] = {}                # chunk_id -> token_count
        self.doc_frequencies: Dict[int, int] = {}            # term_id -> doc_count
        self.total_chunks: int = 0
        self.avgdl: float = 0.0

    @classmethod
    def build_from_chunks(
        cls,
        chunks: List[Union[Chunk, Dict[str, Any]]],
        tokenizer: Optional[BPETokenizer] = None,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    ) -> "InvertedIndex":
        """
        Construct a deterministic inverted index from a list of Chunk objects or chunk dicts.

        Args:
            chunks: List of Chunk dataclass objects or dictionaries.
            tokenizer: Optional pre-loaded BPETokenizer instance.
            tokenizer_path: Fallback path to BPE tokenizer file.

        Returns:
            InvertedIndex instance.
        """
        if tokenizer is None:
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"BPE Tokenizer vocabulary file not found at: {tokenizer_path}")
            tokenizer = BPETokenizer.load(tokenizer_path)

        index = cls()
        if not chunks:
            return index

        # Normalize input to list of dicts
        chunk_dicts = [c.to_dict() if isinstance(c, Chunk) else c for c in chunks]

        # Sort chunks deterministically by chunk_id
        chunk_dicts.sort(key=lambda c: c["chunk_id"])

        total_tokens_sum = 0
        postings_builder: Dict[int, List[Tuple[str, int]]] = {}

        for c_dict in chunk_dicts:
            chunk_id = c_dict["chunk_id"]
            text = c_dict["text"]

            # Tokenize chunk text using BPE tokenizer
            tokens = tokenizer.encode(text)
            token_count = len(tokens)

            # Store chunk metadata
            c_dict["token_count"] = token_count
            index.chunks[chunk_id] = c_dict
            index.doc_lengths[chunk_id] = token_count
            total_tokens_sum += token_count

            # Count term frequencies in chunk
            term_counts = Counter(tokens)
            for term_id, tf in term_counts.items():
                if term_id not in postings_builder:
                    postings_builder[term_id] = []
                postings_builder[term_id].append((chunk_id, tf))

        index.total_chunks = len(chunk_dicts)
        index.avgdl = float(total_tokens_sum / index.total_chunks) if index.total_chunks > 0 else 0.0

        # Sort postings deterministically by term_id and chunk_id
        for term_id in sorted(postings_builder.keys()):
            post_list = postings_builder[term_id]
            post_list.sort(key=lambda x: x[0])  # sort by chunk_id
            index.postings[term_id] = post_list
            index.doc_frequencies[term_id] = len(post_list)

        return index

    def to_dict(self) -> Dict[str, Any]:
        """Serialize inverted index state to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "avgdl": self.avgdl,
            "chunks": self.chunks,
            "doc_lengths": self.doc_lengths,
            "doc_frequencies": {str(k): v for k, v in sorted(self.doc_frequencies.items())},
            "postings": {str(k): v for k, v in sorted(self.postings.items())}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvertedIndex":
        """Deserialize inverted index state from dictionary."""
        index = cls()
        index.total_chunks = data["total_chunks"]
        index.avgdl = data["avgdl"]
        index.chunks = data["chunks"]
        index.doc_lengths = data["doc_lengths"]
        index.doc_frequencies = {int(k): v for k, v in data["doc_frequencies"].items()}
        index.postings = {int(k): [tuple(p) for p in v] for k, v in data["postings"].items()}
        return index

    def save_json(self, filepath: str) -> None:
        """Save index state to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "InvertedIndex":
        """Load index state from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

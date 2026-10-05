import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from bpe_tokenizer import BPETokenizer
from phase7.ingestion.document_loader import RawDocument

DEFAULT_TOKENIZER_PATH = os.path.join("tokenizers", "phase5d", "bpe_vocab_1024.json")

@dataclass
class Chunk:
    """
    Represents a token-aware micro-chunk ready for indexing and retrieval.
    """
    chunk_id: str
    document_id: str
    source: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary format matching JSONL output schema."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


class MicroChunker:
    """
    Token-aware micro-chunker using the project's existing BPE tokenizer.
    Enforces strict 45-token hard limit, 42-token target, and ~10-12 token overlap.
    """

    def __init__(
        self,
        tokenizer: Optional[BPETokenizer] = None,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        target_tokens: int = 42,
        max_tokens: int = 45,
        min_tokens: int = 15,
        overlap_tokens: int = 10
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"BPE Tokenizer vocabulary file not found at: {tokenizer_path}")
            self.tokenizer = BPETokenizer.load(tokenizer_path)

        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    def _is_sentence_boundary_token(self, token_str: str) -> bool:
        """Check if a decoded token represents or ends with a sentence terminal."""
        if not token_str:
            return False
        return any(term in token_str for term in ('.', '?', '!', '\n'))

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        """
        Chunk a RawDocument into token-aware MicroChunks.

        Args:
            doc: RawDocument instance.

        Returns:
            List of Chunk instances.
        """
        raw_text = doc.raw_text.strip()
        if not raw_text:
            return []

        all_tokens = self.tokenizer.encode(raw_text)
        total_tokens = len(all_tokens)

        if total_tokens == 0:
            return []

        # Single chunk case if total tokens <= max_tokens (45)
        if total_tokens <= self.max_tokens:
            chunk_text = self.tokenizer.decode(all_tokens)
            return [
                Chunk(
                    chunk_id=f"{doc.document_id}_chk_000",
                    document_id=doc.document_id,
                    source=doc.source,
                    text=chunk_text,
                    token_count=total_tokens,
                    metadata=dict(doc.metadata)
                )
            ]

        # Decode individual tokens to precompute boundary candidates
        decoded_tokens = [self.tokenizer.decode([tid]) for tid in all_tokens]

        chunks: List[Chunk] = []
        start_idx = 0
        chunk_counter = 0

        while start_idx < total_tokens:
            remaining_tokens = total_tokens - start_idx

            if remaining_tokens <= self.max_tokens:
                end_idx = total_tokens
            else:
                target_end = start_idx + self.target_tokens
                max_end = start_idx + self.max_tokens

                # Search for sentence boundary near target_end in range [start_idx + 25, max_end]
                found_boundary = None
                search_start = max(start_idx + 20, target_end - 10)
                search_end = min(max_end, total_tokens)

                for idx in range(search_start, search_end):
                    if self._is_sentence_boundary_token(decoded_tokens[idx]):
                        found_boundary = idx + 1  # Include terminal token in chunk
                        break

                if found_boundary is not None and found_boundary - start_idx <= self.max_tokens:
                    end_idx = found_boundary
                else:
                    end_idx = min(start_idx + self.target_tokens, max_end, total_tokens)

            chunk_toks = all_tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_toks)
            actual_count = len(chunk_toks)

            chunk_id = f"{doc.document_id}_chk_{chunk_counter:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    source=doc.source,
                    text=chunk_text,
                    token_count=actual_count,
                    metadata=dict(doc.metadata)
                )
            )
            chunk_counter += 1

            if end_idx >= total_tokens:
                break

            # Calculate start for next chunk with overlap
            next_start_target = end_idx - self.overlap_tokens

            # Look for sentence boundary near overlap target
            boundary_overlap = None
            overlap_search_start = max(start_idx + 10, next_start_target - 5)
            overlap_search_end = min(end_idx - 1, next_start_target + 5)

            for idx in range(overlap_search_start, overlap_search_end):
                if self._is_sentence_boundary_token(decoded_tokens[idx]):
                    boundary_overlap = idx + 1
                    break

            if boundary_overlap is not None and boundary_overlap > start_idx and boundary_overlap < end_idx:
                next_start = boundary_overlap
            else:
                next_start = next_start_target

            # Guarantee strict forward progress by at least 15 tokens
            if next_start <= start_idx:
                next_start = start_idx + max(15, (end_idx - start_idx) // 2)

            start_idx = next_start

        return chunks

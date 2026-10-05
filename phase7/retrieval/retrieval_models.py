from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SearchResultItem:
    """
    Represents a single retrieved micro-chunk result with its rank and score.
    """
    rank: int
    score: float
    document_id: str
    chunk_id: str
    source: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result item to dictionary matching JSON output schema."""
        return {
            "rank": self.rank,
            "score": round(self.score, 6),
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "token_count": self.token_count,
            "metadata": self.metadata
        }


@dataclass
class SearchResultEnvelope:
    """
    Represents the full response wrapper for a retrieval query.
    """
    query: str
    results: List[SearchResultItem]
    retrieved: bool
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert response envelope to dictionary format."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "retrieved": self.retrieved,
            "latency_ms": round(self.latency_ms, 3)
        }

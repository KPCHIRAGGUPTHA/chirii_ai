from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class RetrievedContext:
    """
    Represents retrieved micro-chunk context metadata.
    """
    document_id: str
    chunk_id: str
    source: str
    text: str
    retrieval_score: float
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "retrieval_score": round(self.retrieval_score, 6),
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class SourceMetadata:
    """
    Represents source attribution metadata envelope.
    """
    retrieved: bool
    source: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    retrieval_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieved": self.retrieved,
            "source": self.source,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "retrieval_score": round(self.retrieval_score, 6),
        }


@dataclass
class ContextBudgetInfo:
    """
    Represents detailed token budget allocation metrics for a prompt.
    """
    query_tokens: int
    context_tokens: int
    prompt_tokens: int
    generation_reserve: int
    total_budget: int = 128
    truncated_query: bool = False
    truncated_context: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "prompt_tokens": self.prompt_tokens,
            "generation_reserve": self.generation_reserve,
            "total_budget": self.total_budget,
            "truncated_query": self.truncated_query,
            "truncated_context": self.truncated_context,
        }


@dataclass
class GenerationMetadata:
    """
    Represents generation parameters and execution latency.
    """
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int
    generation_tokens: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "generation_tokens": self.generation_tokens,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class RAGRequest:
    """
    Structured request object for RAG generation.
    """
    question: str
    top_k: int = 1
    score_threshold: Optional[float] = None
    max_new_tokens: int = 46
    temperature: float = 0.2
    top_k_sampling: int = 10
    top_p_sampling: float = 0.9


@dataclass
class RAGResponse:
    """
    Structured response object matching Phase 7D specification.
    """
    answer: str
    retrieved: bool
    source: Optional[str]
    document_id: Optional[str]
    chunk_id: Optional[str]
    retrieval_score: float
    generation_tokens: int
    latency_ms: float
    budget: Dict[str, Any]
    retrieved_text: Optional[str] = None
    prompt_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieved": self.retrieved,
            "source": self.source,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "retrieval_score": round(self.retrieval_score, 6),
            "generation_tokens": self.generation_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "budget": self.budget,
        }

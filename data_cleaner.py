import re
from typing import Optional, Dict, Any

class DataCleaner:
    """
    Dedicated data cleaning module for Phase 5 FineWeb-Edu pretraining.
    Filters invalid/empty/short documents and performs whitespace normalization.
    Tracks exact document and character retention statistics.
    """
    def __init__(self, min_doc_length: int = 50, min_words: int = 10, normalize_whitespace: bool = True):
        self.min_doc_length = min_doc_length
        self.min_words = min_words
        self.normalize_whitespace = normalize_whitespace
        
        self.reset_stats()

    def reset_stats(self):
        """Reset cleaning statistics counters."""
        self.stats = {
            "docs_processed": 0,
            "docs_retained": 0,
            "docs_removed_nonstring": 0,
            "docs_removed_empty": 0,
            "docs_removed_short": 0,
            "chars_before": 0,
            "chars_retained": 0
        }

    def clean_document(self, doc_text: Any) -> Optional[str]:
        """
        Clean and validate a single document text entry.
        Returns cleaned text string if document passes validation, else None.
        """
        self.stats["docs_processed"] += 1
        
        # 1. Type check
        if not isinstance(doc_text, str):
            self.stats["docs_removed_nonstring"] += 1
            return None

        self.stats["chars_before"] += len(doc_text)

        # 2. Check empty / whitespace-only
        stripped = doc_text.strip()
        if not stripped:
            self.stats["docs_removed_empty"] += 1
            return None

        # 3. Check minimum length and word count
        words = stripped.split()
        if len(stripped) < self.min_doc_length or len(words) < self.min_words:
            self.stats["docs_removed_short"] += 1
            return None

        # 4. Text Normalization
        text = stripped
        if self.normalize_whitespace:
            # Normalize carriage returns
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # Replace excessive vertical whitespace (3+ newlines) with double newline
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Replace excessive horizontal whitespace (spaces/tabs > 4) with single space
            text = re.sub(r"[ \t]{4,}", " ", text)

        self.stats["docs_retained"] += 1
        self.stats["chars_retained"] += len(text)
        return text

    def get_statistics(self) -> Dict[str, Any]:
        """Return cumulative cleaning statistics."""
        docs_proc = max(self.stats["docs_processed"], 1)
        chars_before = max(self.stats["chars_before"], 1)
        
        stats = self.stats.copy()
        stats["doc_retention_rate_pct"] = round((stats["docs_retained"] / docs_proc) * 100, 2)
        stats["char_retention_rate_pct"] = round((stats["chars_retained"] / chars_before) * 100, 2)
        return stats

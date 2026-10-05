import re
import unicodedata
from typing import Union

class TextNormalizer:
    """
    Deterministic text normalizer for Phase 7 RAG ingestion.
    Performs UTF-8 handling, newline standardization, and whitespace normalization
    while strictly preserving sentence boundaries and semantic content.
    """

    @staticmethod
    def normalize(text: Union[str, bytes]) -> str:
        """
        Deterministically normalizes raw input string or bytes into clean UTF-8 text.

        Args:
            text: Input string or UTF-8 encoded bytes.

        Returns:
            Normalized UTF-8 string.

        Raises:
            ValueError: If input is bytes and contains invalid UTF-8 sequences.
        """
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError as e:
                raise ValueError(f"Invalid UTF-8 sequence in input text: {e}")
        elif not isinstance(text, str):
            raise ValueError(f"Expected str or bytes, got {type(text).__name__}")

        if not text:
            return ""

        # 1. Standardize line endings (\r\n and \r to \n)
        s = text.replace('\r\n', '\n').replace('\r', '\n')

        # 2. Convert non-breaking space and unicode space variants to standard ASCII space
        s = s.replace('\u00a0', ' ')

        # 3. Strip control characters except newline (\n) and tab (\t)
        cleaned_chars = []
        for char in s:
            cat = unicodedata.category(char)
            if char in ('\n', '\t') or not cat.startswith('C'):
                cleaned_chars.append(char)
        s = "".join(cleaned_chars)

        # 4. Collapse 3+ consecutive newlines down to 2 newlines (preserve paragraph separation)
        s = re.sub(r'\n{3,}', '\n\n', s)

        # 5. Collapse multiple horizontal spaces/tabs on each line to a single space
        lines = []
        for line in s.split('\n'):
            normalized_line = re.sub(r'[ \t]+', ' ', line).strip()
            lines.append(normalized_line)
        s = '\n'.join(lines)

        # 6. Ensure single trailing newline if non-empty
        s = s.strip()
        return s

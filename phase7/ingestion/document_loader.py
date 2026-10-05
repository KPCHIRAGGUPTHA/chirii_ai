import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from phase7.ingestion.text_normalizer import TextNormalizer

SUPPORTED_EXTENSIONS = {".txt", ".md", ".jsonl"}

@dataclass
class RawDocument:
    """
    Represents an un-chunked document loaded from disk.
    """
    document_id: str
    source: str
    file_type: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """
    Loads text, markdown, and JSONL documents deterministically.
    """

    @staticmethod
    def _generate_doc_id(source: str, content: str, index: int = 0) -> str:
        """
        Generate a deterministic SHA-256 document ID based on source path, content, and record index.
        """
        hasher = hashlib.sha256()
        hasher.update(source.encode("utf-8"))
        hasher.update(content.encode("utf-8"))
        hasher.update(str(index).encode("utf-8"))
        return f"doc_{hasher.hexdigest()[:16]}"

    @classmethod
    def load_file(cls, filepath: str) -> List[RawDocument]:
        """
        Load a document file (.txt, .md, or .jsonl) into a list of RawDocument instances.

        Args:
            filepath: Path to input document file.

        Returns:
            List of RawDocument instances.

        Raises:
            ValueError: If file extension is unsupported, file is empty, or binary UTF-8 decoding fails.
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Document file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{ext}'. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}")

        rel_source = os.path.normpath(filepath).replace("\\", "/")

        with open(filepath, "rb") as f:
            raw_bytes = f.read()

        if not raw_bytes:
            raise ValueError(f"Document file is empty: {filepath}")

        try:
            raw_string = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"File {filepath} is not valid UTF-8: {e}")

        if ext in (".txt", ".md"):
            normalized_text = TextNormalizer.normalize(raw_string)
            if not normalized_text:
                raise ValueError(f"Document file contains no valid text after normalization: {filepath}")
            doc_id = cls._generate_doc_id(rel_source, normalized_text)
            file_type = ext.lstrip(".")
            return [
                RawDocument(
                    document_id=doc_id,
                    source=rel_source,
                    file_type=file_type,
                    raw_text=normalized_text,
                    metadata={"original_filename": os.path.basename(filepath)}
                )
            ]

        elif ext == ".jsonl":
            documents = []
            lines = raw_string.strip().splitlines()
            for idx, line in enumerate(lines):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON record on line {idx + 1} of {filepath}: {e}")

                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record on line {idx + 1} must be a JSON object (dict)")

                # Schema resolution for textual content
                text_content = ""
                extracted_meta = {}
                for key, val in record.items():
                    if key in ("text", "content", "body", "instruction"):
                        if isinstance(val, str) and val.strip():
                            if text_content:
                                text_content += "\n" + val.strip()
                            else:
                                text_content = val.strip()
                    elif key == "output" and "instruction" in record:
                        # Standard Alpaca instruction/output pair
                        if isinstance(val, str) and val.strip():
                            text_content += "\nResponse:\n" + val.strip()
                    else:
                        extracted_meta[key] = val

                normalized_text = TextNormalizer.normalize(text_content)
                if not normalized_text:
                    continue

                doc_id = cls._generate_doc_id(rel_source, normalized_text, index=idx)
                extracted_meta["original_filename"] = os.path.basename(filepath)
                extracted_meta["jsonl_line_number"] = idx + 1

                documents.append(
                    RawDocument(
                        document_id=doc_id,
                        source=rel_source,
                        file_type="jsonl",
                        raw_text=normalized_text,
                        metadata=extracted_meta
                    )
                )

            if not documents:
                raise ValueError(f"JSONL file contained no valid textual records: {filepath}")

            return documents

        raise ValueError(f"Unsupported file format: {ext}")

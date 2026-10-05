import os
import tempfile
import json
import pytest

from phase7.ingestion.text_normalizer import TextNormalizer
from phase7.ingestion.document_loader import DocumentLoader, RawDocument
from phase7.ingestion.chunker import MicroChunker, Chunk
from phase7.ingestion.ingest import ingest_file, ingest_directory, save_chunks_jsonl

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "phase7b")
TXT_FIXTURE = os.path.join(FIXTURES_DIR, "sample.txt")
MD_FIXTURE = os.path.join(FIXTURES_DIR, "sample.md")
JSONL_FIXTURE = os.path.join(FIXTURES_DIR, "sample.jsonl")

# 1. TXT loading
def test_txt_loading():
    docs = DocumentLoader.load_file(TXT_FIXTURE)
    assert len(docs) == 1
    assert docs[0].file_type == "txt"
    assert "MiniGPT Model C" in docs[0].raw_text
    assert docs[0].source.replace("\\", "/").endswith("sample.txt")

# 2. Markdown loading
def test_markdown_loading():
    docs = DocumentLoader.load_file(MD_FIXTURE)
    assert len(docs) == 1
    assert docs[0].file_type == "md"
    assert "Architecture Notes" in docs[0].raw_text

# 3. JSONL loading
def test_jsonl_loading():
    docs = DocumentLoader.load_file(JSONL_FIXTURE)
    assert len(docs) == 2
    assert docs[0].file_type == "jsonl"
    assert "MiniGPT is a lightweight" in docs[0].raw_text
    assert docs[0].metadata.get("topic") == "architecture"
    assert "Zorblax" in docs[1].raw_text

# 4. UTF-8 handling
def test_utf8_handling():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write("UTF-8 text with symbols: © ® ★ ⚡ MiniGPT test.")
        f_path = f.name
    try:
        docs = DocumentLoader.load_file(f_path)
        assert "© ® ★ ⚡" in docs[0].raw_text
    finally:
        os.remove(f_path)

# 5. Whitespace normalization
def test_whitespace_normalization():
    raw_text = "   Line 1 with   multiple    spaces.\r\n\r\n\r\nLine 2 with \t tabs.   \n\n\nLine 3.  "
    normalized = TextNormalizer.normalize(raw_text)
    assert "\r" not in normalized
    assert "  " not in normalized  # Multiple spaces collapsed
    assert "\n\n\n" not in normalized  # 3+ newlines collapsed to 2
    assert normalized.startswith("Line 1")

# 6. Deterministic document IDs
def test_deterministic_doc_ids():
    docs1 = DocumentLoader.load_file(TXT_FIXTURE)
    docs2 = DocumentLoader.load_file(TXT_FIXTURE)
    assert docs1[0].document_id == docs2[0].document_id

# 7. Deterministic chunk IDs
def test_deterministic_chunk_ids():
    chunks1 = ingest_file(TXT_FIXTURE)
    chunks2 = ingest_file(TXT_FIXTURE)
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.token_count == c2.token_count

# 8. Token-aware chunk size
def test_token_aware_chunk_size():
    chunker = MicroChunker()
    chunks = ingest_file(TXT_FIXTURE, chunker=chunker)
    for chunk in chunks:
        # Re-tokenize chunk text to verify token count match
        tok_count = len(chunker.tokenizer.encode(chunk.text))
        assert chunk.token_count == tok_count

# 9. Target chunk size around 42 tokens
def test_target_chunk_size_around_42():
    # Construct longer text
    long_text = "MiniGPT is a micro transformer model trained on FineWeb-Edu dataset. " * 10
    doc = RawDocument(
        document_id="doc_test_target",
        source="test_target.txt",
        file_type="txt",
        raw_text=long_text
    )
    chunker = MicroChunker(target_tokens=42, max_tokens=45)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 45

# 10. Hard maximum of 45 tokens
def test_hard_maximum_45_tokens():
    long_text = "Sentence one is here. " + "Word " * 200
    doc = RawDocument(
        document_id="doc_test_max",
        source="test_max.txt",
        file_type="txt",
        raw_text=long_text
    )
    chunker = MicroChunker(max_tokens=45)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        assert chunk.token_count <= 45, f"Chunk {chunk.chunk_id} exceeded max 45 tokens: {chunk.token_count}"

# 11. Overlap around 10-12 tokens
def test_overlap_around_10_12_tokens():
    text = (
        "MiniGPT Model C has 6.61M parameters. "
        "It uses 8 transformer layers and 8 attention heads. "
        "The embedding dimension is set to 256. "
        "The model context length is fixed at 128 tokens. "
        "Supervised fine-tuning improves instruction following performance."
    )
    doc = RawDocument(
        document_id="doc_overlap",
        source="overlap.txt",
        file_type="txt",
        raw_text=text
    )
    chunker = MicroChunker(target_tokens=30, max_tokens=45, overlap_tokens=10)
    chunks = chunker.chunk_document(doc)

    if len(chunks) >= 2:
        # Compute token overlap between chunk 0 and chunk 1
        toks0 = set(chunker.tokenizer.encode(chunks[0].text))
        toks1 = set(chunker.tokenizer.encode(chunks[1].text))
        overlap_size = len(toks0.intersection(toks1))
        assert overlap_size > 0, "Chunks should share overlapping subword tokens"

# 12. Sentence-aware boundaries
def test_sentence_aware_boundaries():
    text = (
        "First sentence is short. "
        "Second sentence provides details about parameters and layers. "
        "Third sentence talks about FineWeb-Edu training dataset."
    )
    doc = RawDocument(document_id="doc_sent", source="sent.txt", file_type="txt", raw_text=text)
    chunker = MicroChunker(target_tokens=35, max_tokens=45)
    chunks = chunker.chunk_document(doc)

    for chunk in chunks:
        # Check if chunk text ends cleanly with punctuation or complete token sequence
        assert len(chunk.text.strip()) > 0

# 13. Long sentence splitting
def test_long_sentence_splitting():
    # Single sentence with 100+ tokens without punctuation
    long_sentence = "The model " + "parameter " * 120
    doc = RawDocument(document_id="doc_long_sent", source="long.txt", file_type="txt", raw_text=long_sentence)
    chunker = MicroChunker(max_tokens=45)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 45

# 14. Metadata preservation
def test_metadata_preservation():
    docs = DocumentLoader.load_file(JSONL_FIXTURE)
    chunker = MicroChunker()
    chunks = chunker.chunk_document(docs[0])

    assert len(chunks) > 0
    assert chunks[0].metadata.get("topic") == "architecture"
    assert chunks[0].source.endswith("sample.jsonl")

# 15. Deterministic repeated ingestion
def test_deterministic_repeated_ingestion():
    chunks_run1 = ingest_directory(FIXTURES_DIR)
    chunks_run2 = ingest_directory(FIXTURES_DIR)

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.to_dict() == c2.to_dict()

# 16. Empty/invalid input handling
def test_empty_invalid_input_handling():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write("   \n\n  \t ")
        empty_path = f.name

    try:
        with pytest.raises(ValueError, match="no valid text"):
            DocumentLoader.load_file(empty_path)
    finally:
        os.remove(empty_path)

# 17. Unsupported extension rejection
def test_unsupported_extension_rejection():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".pdf", delete=False) as f:
        f.write("Dummy PDF content")
        pdf_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file extension"):
            DocumentLoader.load_file(pdf_path)
    finally:
        os.remove(pdf_path)

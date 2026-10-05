import os
import json
import argparse
from typing import List, Optional, Dict, Any

from phase7.ingestion.document_loader import DocumentLoader, RawDocument, SUPPORTED_EXTENSIONS
from phase7.ingestion.chunker import MicroChunker, Chunk

def ingest_file(filepath: str, chunker: Optional[MicroChunker] = None) -> List[Chunk]:
    """
    Ingest a single document file (.txt, .md, or .jsonl) and return its micro-chunks.

    Args:
        filepath: Path to the document file.
        chunker: Optional MicroChunker instance.

    Returns:
        List of Chunk objects.
    """
    if chunker is None:
        chunker = MicroChunker()

    raw_docs = DocumentLoader.load_file(filepath)
    all_chunks: List[Chunk] = []
    for doc in raw_docs:
        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks

def ingest_directory(dir_path: str, chunker: Optional[MicroChunker] = None) -> List[Chunk]:
    """
    Deterministically scan a directory for supported files, load and chunk them.

    Args:
        dir_path: Absolute or relative path to directory.
        chunker: Optional MicroChunker instance.

    Returns:
        List of Chunk objects sorted deterministically by document source and chunk_id.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if chunker is None:
        chunker = MicroChunker()

    # Collect files deterministically by sorting relative paths
    target_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                rel_path = os.path.normpath(full_path).replace("\\", "/")
                target_files.append((rel_path, full_path))

    target_files.sort(key=lambda x: x[0])

    all_chunks: List[Chunk] = []
    for rel_path, full_path in target_files:
        try:
            chunks = ingest_file(full_path, chunker=chunker)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: Skipping file {full_path} due to ingestion error: {e}")

    return all_chunks

def save_chunks_jsonl(chunks: List[Chunk], output_filepath: str) -> None:
    """
    Serialize chunks deterministically to a JSONL file.

    Args:
        chunks: List of Chunk objects.
        output_filepath: Target output file path.
    """
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            line_dict = chunk.to_dict()
            json_line = json.dumps(line_dict, ensure_ascii=False, sort_keys=True)
            f.write(json_line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7B RAG Ingestion & Micro-Chunking Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input document file or directory path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL filepath")
    args = parser.parse_args()

    chunker = MicroChunker()
    if os.path.isdir(args.input):
        chunks = ingest_directory(args.input, chunker=chunker)
    else:
        chunks = ingest_file(args.input, chunker=chunker)

    save_chunks_jsonl(chunks, args.output)
    print(f"Successfully ingested {len(chunks)} chunks into {args.output}")

import os
import json
import hashlib
import torch
from typing import Iterator, List, Tuple, Dict, Any, Optional
from datasets import load_dataset
from data_cleaner import DataCleaner
from phase5_config import Phase5Config

def get_document_split(doc: Dict[str, Any], val_ratio: float = 0.10) -> str:
    """
    Deterministically assign a document to 'train' or 'val' split using SHA-256 hashing.
    Guarantees strict data isolation with zero document overlap.
    """
    identifier = str(doc.get("id", "")) or str(doc.get("text", ""))
    hash_digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    hash_int = int(hash_digest, 16)
    threshold = int(val_ratio * 100)
    return "val" if (hash_int % 100) < threshold else "train"

def prepare_phase5_corpus(
    config: Phase5Config,
    max_train_docs: int = 300,
    max_val_docs: int = 50,
    force_rebuild: bool = False
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Stream FineWeb-Edu documents once, apply deterministic hashing split, and cache
    isolated train and validation text corpora in data/phase5/ for fast local execution.
    """
    config.create_dirs()
    train_path = os.path.join(config.data_dir, "train_corpus.txt")
    val_path = os.path.join(config.data_dir, "val_corpus.txt")
    stats_path = os.path.join(config.results_dir, "dataset_statistics.json")

    if not force_rebuild and os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Loading cached Phase 5 dataset corpora from {config.data_dir}...", flush=True)
        with open(train_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(val_path, "r", encoding="utf-8") as f:
            val_text = f.read()
        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        return train_text, val_text, stats

    print(f"Streaming & cleaning FineWeb-Edu dataset ({config.dataset_config})...", flush=True)
    cleaner = DataCleaner(
        min_doc_length=config.min_doc_length,
        min_words=config.min_words,
        normalize_whitespace=config.normalize_whitespace
    )

    dataset = load_dataset(
        config.dataset_name,
        name=config.dataset_config,
        split=config.dataset_split,
        streaming=True
    )

    train_docs, val_docs = [], []
    train_count, val_count = 0, 0

    for raw_doc in dataset:
        doc_split = get_document_split(raw_doc, val_ratio=config.val_hash_ratio)
        raw_text = raw_doc.get("text", "")
        cleaned_text = cleaner.clean_document(raw_text)
        if cleaned_text is None:
            continue

        if doc_split == "train" and train_count < max_train_docs:
            train_docs.append(cleaned_text)
            train_count += 1
        elif doc_split == "val" and val_count < max_val_docs:
            val_docs.append(cleaned_text)
            val_count += 1

        if train_count >= max_train_docs and val_count >= max_val_docs:
            break

    train_text = "\n\n".join(train_docs)
    val_text = "\n\n".join(val_docs)

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)

    stats = cleaner.get_statistics()
    stats["train_docs_retained"] = len(train_docs)
    stats["val_docs_retained"] = len(val_docs)
    stats["train_chars_retained"] = len(train_text)
    stats["val_chars_retained"] = len(val_text)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Cached Phase 5 corpora: Train ({len(train_text):,} chars), Val ({len(val_text):,} chars)", flush=True)
    return train_text, val_text, stats

def get_batch_phase5(
    data_tensor: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a batch of input x and target y sequences from token tensor.
    x is sequence of length `block_size`, y is target sequence shifted by 1 (y[t] == x[t+1]).
    """
    if len(data_tensor) <= block_size:
        raise ValueError(f"Dataset token length ({len(data_tensor)}) must exceed block_size ({block_size})")

    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i + block_size] for i in ix])
    y = torch.stack([data_tensor[i + 1:i + block_size + 1] for i in ix])
    
    if device != 'cpu':
        x, y = x.to(device), y.to(device)
    return x, y

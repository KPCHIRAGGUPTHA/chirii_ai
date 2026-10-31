"""
Phase 10A — Tokenizer v2: byte-level BPE, vocabulary 16,384, every digit its own token.

Trained on about 300 MB of the train split. Documents are chosen by hash, so the sample is deterministic and spread
across all shards.

Usage:
    python phase10/tokenizer_v2.py train
    python phase10/tokenizer_v2.py stats
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Iterator, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from phase10.common import EOT, SPECIAL_TOKENS, TEXT_MANIFEST, TOKENIZER_PATH, TOKENIZER_STATS, atomic_write_json, project_path, sha256_text

VOCAB_SIZE = 16384
SAMPLE_BYTES = 300_000_000


def build_untrained() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(documents: Iterator[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Every byte is in the initial alphabet, so any text can be encoded and nothing maps to an unknown token."""
    tokenizer = build_untrained()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    return tokenizer


def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode(tokenizer: Tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def _byte_decoder() -> dict:
    """The inverse of GPT-2's byte-to-unicode table, which ByteLevel uses to spell token strings."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(printable, chars)}


def token_byte_lengths(tokenizer: Tokenizer) -> np.ndarray:
    """UTF-8 bytes each token stands for, read from its token string. Special tokens count as 0 bytes of text."""
    table = _byte_decoder()
    vocab = tokenizer.get_vocab()
    lengths = np.zeros(len(vocab), dtype=np.int64)
    for token, token_id in vocab.items():
        lengths[token_id] = 0 if token in SPECIAL_TOKENS else sum(1 for ch in token if ch in table)
    return lengths


def iter_documents(path: str) -> Iterator[bytes]:
    with open(path, "rb") as f:
        data = f.read()
    terminator = EOT.encode("utf-8")
    for doc in data.split(terminator):
        if doc:
            yield doc


def load_shards(split: str) -> List[dict]:
    with open(project_path(TEXT_MANIFEST), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return [s for s in manifest["shards"] if s["split"] == split]


def sample_documents(sample_bytes: int, train_bytes: int) -> Iterator[str]:
    # Uses different hash bytes from the train/val split, which reads the first 8
    threshold = min(1.0, sample_bytes / train_bytes * 1.02)
    taken = 0
    for shard in load_shards("train"):
        for doc in iter_documents(project_path(shard["file"])):
            if int.from_bytes(hashlib.sha256(doc).digest()[8:16], "big") / 2**64 < threshold:
                taken += len(doc)
                yield doc.decode("utf-8")
                if taken >= sample_bytes:
                    print(f"Sampled {taken:,} bytes", flush=True)
                    return
    print(f"Sampled {taken:,} bytes", flush=True)


def command_train():
    train_bytes = sum(s["text_bytes"] for s in load_shards("train"))
    start = time.time()
    tokenizer = train_tokenizer(sample_documents(SAMPLE_BYTES, train_bytes))
    path = project_path(TOKENIZER_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)
    print(f"Trained in {time.time() - start:.0f}s: vocab {tokenizer.get_vocab_size():,}, sha256 {sha256_text(path)}", flush=True)


def command_stats():
    tokenizer = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    lengths = token_byte_lengths(tokenizer)
    text_bytes = tokens = documents = 0
    for shard in load_shards("val"):
        docs = [d.decode("utf-8") for d in iter_documents(project_path(shard["file"]))]
        for encoding, doc in zip(tokenizer.encode_batch_fast(docs, add_special_tokens=False), docs):
            tokens += len(encoding.ids)
            text_bytes += len(doc.encode("utf-8"))
            documents += 1
    stats = {
        "tokenizer": TOKENIZER_PATH,
        "sha256": sha256_text(project_path(TOKENIZER_PATH)),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "digit_token_ids": {str(d): tokenizer.token_to_id(str(d)) for d in range(10)},
        "val_documents": documents,
        "val_text_bytes": text_bytes,
        "val_tokens_without_eot": tokens,
        "val_bytes_per_token": text_bytes / tokens,
        "longest_token_bytes": int(lengths.max()),
    }
    atomic_write_json(project_path(TOKENIZER_STATS), stats)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tokenizer v2")
    parser.add_argument("command", choices=["train", "stats"])
    args = parser.parse_args()
    {"train": command_train, "stats": command_stats}[args.command]()


if __name__ == "__main__":
    main()

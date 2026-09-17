import os
import json
from typing import Dict, Any, Tuple
from bpe_tokenizer import BPETokenizer
from phase5_config import Phase5Config
from phase5_dataset import prepare_phase5_corpus

def train_phase5_tokenizer(
    config: Phase5Config,
    train_text: str = None,
    target_vocab_size: int = 1024,
    max_train_chars: int = 15000
) -> Tuple[BPETokenizer, Dict[str, Any]]:
    """
    Train a Phase 5 BPE tokenizer strictly on the FineWeb-Edu TRAIN split text.
    Validation documents are NEVER accessed during tokenizer training.
    """
    config.create_dirs()
    if train_text is None:
        train_text, _, _ = prepare_phase5_corpus(config)

    print(f"Training Phase 5 BPE Tokenizer (target_vocab_size={target_vocab_size}) on {len(train_text):,} chars...", flush=True)
    tokenizer = BPETokenizer.train(train_text, target_vocab_size=target_vocab_size, max_train_chars=max_train_chars)

    # Save tokenizer in isolated Phase 5 directory
    vocab_filename = f"bpe_vocab_{tokenizer.vocab_size}.json"
    vocab_path = os.path.join(config.tokenizers_dir, vocab_filename)
    tokenizer.save(vocab_path)
    
    # Save default phase5_vocab.json
    default_vocab_path = os.path.join(config.tokenizers_dir, "phase5_vocab.json")
    tokenizer.save(default_vocab_path)
    
    print(f"Saved Phase 5 BPE tokenizer to {vocab_path} and {default_vocab_path}", flush=True)

    # Calculate token compression statistics over sample text
    sample_text = train_text[:50000]
    encoded_tokens = tokenizer.encode(sample_text)
    avg_chars_per_token = len(sample_text) / max(len(encoded_tokens), 1)
    char_count = len(sample_text)
    token_count = len(encoded_tokens)
    token_reduction_pct = ((char_count - token_count) / char_count) * 100

    # Model parameter count impact (Embedding dimension config.n_embd)
    wte_params = tokenizer.vocab_size * config.n_embd

    stats = {
        "target_vocab_size": target_vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "base_chars_count": len(tokenizer.base_chars),
        "num_merges_learned": len(tokenizer.merges),
        "sample_chars_evaluated": char_count,
        "sample_tokens_produced": token_count,
        "avg_chars_per_token": round(avg_chars_per_token, 4),
        "token_reduction_pct": round(token_reduction_pct, 2),
        "embedding_parameters_wte": wte_params,
        "vocab_saved_path": vocab_path,
        "train_corpus_chars": len(train_text)
    }

    stats_path = os.path.join(config.results_dir, "tokenizer_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return tokenizer, stats

def load_phase5_tokenizer(config: Phase5Config, vocab_size: int = 1024) -> BPETokenizer:
    """Load an existing Phase 5 BPE tokenizer from disk."""
    vocab_path = os.path.join(config.tokenizers_dir, f"bpe_vocab_{vocab_size}.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(config.tokenizers_dir, "phase5_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Phase 5 tokenizer file not found at {vocab_path}. Train it first!")
    return BPETokenizer.load(vocab_path)

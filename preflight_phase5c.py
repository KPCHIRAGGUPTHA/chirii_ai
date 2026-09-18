import os
import time
import hashlib
import torch
from phase5_config import Phase5Config
from phase5_dataset import prepare_phase5_corpus
from phase5_tokenizer import load_phase5_tokenizer

def run_preflight_check(target_tokens: int = 10_240_000, target_val_docs: int = 500):
    config = Phase5Config(vocab_size=1024)
    tokenizer = load_phase5_tokenizer(config, vocab_size=1024)
    
    print("Loading / Streaming FineWeb-Edu corpus for Phase 5C Preflight Check...", flush=True)
    
    # Estimate ~3.0 chars/token => ~31M train chars for 10.24M tokens
    # We set min_train_chars = 33,000,000 or max_train_docs high enough (e.g. 10,000)
    train_text, val_text, stats = prepare_phase5_corpus(
        config,
        max_train_docs=12000,
        max_val_docs=target_val_docs,
        min_train_chars=35_000_000,
        min_val_chars=3_500_000,
        force_rebuild=False  # Reuse cached Phase 5C corpus
    )
    
    print("Encoding train and validation corpora with BPE tokenizer (vocab_size=1024)...", flush=True)
    start_enc = time.time()
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    enc_time = time.time() - start_enc
    print(f"Encoding complete in {enc_time:.2f}s.", flush=True)
    
    # Save encoded token tensors for instant training loading
    train_tokens_path = os.path.join(config.data_dir, f"train_tokens_v{tokenizer.vocab_size}.pt")
    val_tokens_path = os.path.join(config.data_dir, f"val_tokens_v{tokenizer.vocab_size}.pt")
    torch.save(torch.tensor(train_tokens, dtype=torch.long), train_tokens_path)
    torch.save(torch.tensor(val_tokens, dtype=torch.long), val_tokens_path)
    print(f"Cached encoded token tensors to {train_tokens_path} and {val_tokens_path}", flush=True)
    
    num_train_docs = stats.get("train_docs_retained", 0)
    num_val_docs = stats.get("val_docs_retained", 0)
    num_train_tokens = len(train_tokens)
    num_val_tokens = len(val_tokens)
    
    sufficient = num_train_tokens >= target_tokens
    sufficient_str = "YES" if sufficient else "NO"
    estimated_epochs = target_tokens / num_train_tokens if num_train_tokens > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("               PHASE 5C PREFLIGHT DATA CHECK               ")
    print("=" * 60)
    print(f"Training documents: {num_train_docs:,}")
    print(f"Validation documents: {num_val_docs:,}")
    print(f"Unique training tokens: {num_train_tokens:,}")
    print(f"Unique validation tokens: {num_val_tokens:,}")
    print(f"Target training tokens: {target_tokens:,}")
    print(f"Training-token capacity sufficient: {sufficient_str}")
    print(f"Estimated epochs/passes over selected documents: {estimated_epochs:.4f}")
    print("=" * 60 + "\n")
    
    return {
        "num_train_docs": num_train_docs,
        "num_val_docs": num_val_docs,
        "num_train_tokens": num_train_tokens,
        "num_val_tokens": num_val_tokens,
        "sufficient": sufficient,
        "estimated_epochs": estimated_epochs,
        "train_text": train_text,
        "val_text": val_text,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens
    }

if __name__ == "__main__":
    run_preflight_check()

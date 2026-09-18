import os
import time
import json
import torch
from phase5_config import Phase5Config
from phase5_dataset import prepare_phase5_corpus, get_batch_phase5
from phase5_tokenizer import train_phase5_tokenizer, load_phase5_tokenizer
from model import MiniGPT, MiniGPTConfig
from train import set_seed

def run_benchmark(num_benchmark_steps: int = 30):
    print("=" * 70)
    print("      RUNNING PHASE 5C HARDWARE & THROUGHPUT BENCHMARK          ")
    print("=" * 70)

    config = Phase5Config(vocab_size=1024)
    config.create_dirs()
    set_seed(config.random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")

    # 1. Prepare corpus if not cached or load cached
    print("Preparing/loading corpus...")
    train_text, val_text, ds_stats = prepare_phase5_corpus(config, max_train_docs=200, max_val_docs=20)

    # 2. Tokenizer check/train
    vocab_path = os.path.join(config.tokenizers_dir, "bpe_vocab_1024.json")
    if os.path.exists(vocab_path):
        print(f"Loading existing tokenizer from {vocab_path}...")
        tokenizer = load_phase5_tokenizer(config, vocab_size=1024)
    else:
        print("Training 1024 vocab BPE tokenizer...")
        tokenizer, _ = train_phase5_tokenizer(config, train_text=train_text, target_vocab_size=1024, max_train_chars=20000)

    train_tokens = tokenizer.encode(train_text)
    train_data = torch.tensor(train_tokens, dtype=torch.long)
    print(f"Loaded {len(train_tokens):,} training tokens for benchmark.")

    # 3. Model setup
    model_config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout
    )
    model = MiniGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    tokens_per_iter = config.effective_batch_size * config.block_size
    print(f"Tokens per iteration: {tokens_per_iter:,} ({config.effective_batch_size} batch * {config.block_size} block)")

    # Warmup step (1 step excluded from timing)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(config.gradient_accumulation_steps):
        xb, yb = get_batch_phase5(train_data, config.block_size, config.micro_batch_size, device=device)
        _, loss = model(xb, yb)
        (loss / config.gradient_accumulation_steps).backward()
    optimizer.step()

    # Benchmark loop
    print(f"Benchmarking throughput over {num_benchmark_steps} iterations...")
    start_time = time.perf_counter()

    for i in range(num_benchmark_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.gradient_accumulation_steps):
            xb, yb = get_batch_phase5(train_data, config.block_size, config.micro_batch_size, device=device)
            _, loss = model(xb, yb)
            (loss / config.gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    elapsed = time.perf_counter() - start_time
    total_tokens = num_benchmark_steps * tokens_per_iter

    tokens_per_sec = total_tokens / elapsed
    tokens_per_min = tokens_per_sec * 60
    tokens_per_hour = tokens_per_min * 60
    iters_per_min = (num_benchmark_steps / elapsed) * 60

    est_1m_sec = 1_000_000 / tokens_per_sec
    est_5m_sec = 5_000_000 / tokens_per_sec
    est_10m_sec = 10_000_000 / tokens_per_sec
    est_25m_sec = 25_000_000 / tokens_per_sec
    est_50m_sec = 50_000_000 / tokens_per_sec

    benchmark_data = {
        "device": device,
        "cpu_threads": torch.get_num_threads(),
        "benchmark_steps": num_benchmark_steps,
        "tokens_per_iter": tokens_per_iter,
        "total_tokens_benchmarked": total_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(tokens_per_sec, 2),
        "tokens_per_minute": round(tokens_per_min, 2),
        "tokens_per_hour": round(tokens_per_hour, 2),
        "iterations_per_minute": round(iters_per_min, 2),
        "estimated_time_hours": {
            "1M_tokens": round(est_1m_sec / 3600, 4),
            "5M_tokens": round(est_5m_sec / 3600, 4),
            "10M_tokens": round(est_10m_sec / 3600, 4),
            "25M_tokens": round(est_25m_sec / 3600, 4),
            "50M_tokens": round(est_50m_sec / 3600, 4)
        },
        "estimated_time_minutes": {
            "1M_tokens": round(est_1m_sec / 60, 2),
            "5M_tokens": round(est_5m_sec / 60, 2),
            "10M_tokens": round(est_10m_sec / 60, 2),
            "25M_tokens": round(est_25m_sec / 60, 2),
            "50M_tokens": round(est_50m_sec / 60, 2)
        }
    }

    out_path = os.path.join(config.results_dir, "phase5c_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}:")
    print(json.dumps(benchmark_data, indent=2))
    return benchmark_data

if __name__ == "__main__":
    run_benchmark(30)

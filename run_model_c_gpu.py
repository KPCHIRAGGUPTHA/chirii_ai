import os
import sys
import time
import json
import argparse
import math
import torch

from model import MiniGPTConfig, MiniGPT
from bpe_tokenizer import BPETokenizer
from phase5d_config import Phase5DConfig
from train import calculate_perplexity, calculate_bpc, set_seed
from train_phase5 import get_lr, estimate_phase5_loss, save_phase5_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Run Model C GPU Pretraining (Phase 5D)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5-step smoke test")
    args = parser.parse_args()

    # 1. Hardware & Environment Check
    if not torch.cuda.is_available():
        print("ERROR: CUDA/GPU is not available! Model C GPU training requires CUDA.", flush=True)
        sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    print("=== Model C GPU Training Pipeline ===", flush=True)
    print(f"Hardware GPU: {gpu_name} (CUDA {cuda_version})", flush=True)

    # 2. Config & Seed Setup
    config = Phase5DConfig.create_model_c_config()
    set_seed(config.random_seed)
    config.create_dirs()

    if args.smoke_test:
        print("Running 5-step pre-flight smoke test...", flush=True)
        config.max_iters = 5
        config.eval_interval = 5

    # 3. Load Tokenizer
    tokenizer_path = os.path.join(config.tokenizers_dir, "bpe_vocab_1024.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join("tokenizers/phase5", "bpe_vocab_1024.json")
    
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer file not found at {tokenizer_path}", flush=True)
        sys.exit(1)

    tokenizer = BPETokenizer.load(tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    print(f"Loaded Tokenizer: {tokenizer_path} (vocab_size={tokenizer.vocab_size})", flush=True)

    # 4. Load Dataset & Token Cache
    train_corpus_path = os.path.join(config.data_dir, "train_corpus.txt")
    val_corpus_path = os.path.join(config.data_dir, "val_corpus.txt")

    if not os.path.exists(train_corpus_path) or not os.path.exists(val_corpus_path):
        print(f"ERROR: Corpus text files missing in {config.data_dir}", flush=True)
        sys.exit(1)

    print("Loading validation corpus text for exact character counting...", flush=True)
    with open(val_corpus_path, "r", encoding="utf-8") as f:
        val_text = f.read()
    num_val_chars = len(val_text)
    print(f"Validation corpus loaded: {num_val_chars:,} characters", flush=True)

    train_tokens_path = os.path.join(config.data_dir, f"train_tokens_v{tokenizer.vocab_size}.pt")
    val_tokens_path = os.path.join(config.data_dir, f"val_tokens_v{tokenizer.vocab_size}.pt")

    if os.path.exists(train_tokens_path) and os.path.exists(val_tokens_path):
        print("Loading cached encoded tokens...", flush=True)
        train_data = torch.load(train_tokens_path, weights_only=True)
        val_data = torch.load(val_tokens_path, weights_only=True)
        train_tokens = train_data.tolist()
        val_tokens = val_data.tolist()
    else:
        print("Encoding corpus text with BPE tokenizer...", flush=True)
        with open(train_corpus_path, "r", encoding="utf-8") as f:
            train_text = f.read()
        train_tokens = tokenizer.encode(train_text)
        val_tokens = tokenizer.encode(val_text)
        train_data = torch.tensor(train_tokens, dtype=torch.long)
        val_data = torch.tensor(val_tokens, dtype=torch.long)
        torch.save(train_data, train_tokens_path)
        torch.save(val_data, val_tokens_path)

    print(f"Train Tokens: {len(train_tokens):,} | Val Tokens: {len(val_tokens):,}", flush=True)

    # 5. Model Initialization
    model_config = Phase5DConfig.get_model_c_minigpt_config()
    model = MiniGPT(model_config).to(device)
    num_params = model.get_num_params()
    print(f"Model C Architecture: {model_config.n_layer} layers, {model_config.n_head} heads, {model_config.n_embd} embd", flush=True)
    print(f"Model C Parameters: {num_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    # Save tokenizer in checkpoints directory
    ckpt_vocab_path = os.path.join(config.checkpoints_dir, "vocab.json")
    tokenizer.save(ckpt_vocab_path)

    # 6. Training Loop
    start_time = time.time()
    history = []
    best_val_loss = float('inf')
    val_loss_val = 0.0
    val_perplexity = 0.0
    val_bpc = 0.0
    train_loss_val = 0.0

    print(f"Starting Model C training: {config.max_iters} steps, effective batch size {config.effective_batch_size}", flush=True)

    for iter_step in range(1, config.max_iters + 1):
        model.train()
        lr = get_lr(iter_step, config.max_iters, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        # Gradient Accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            ix = torch.randint(len(train_data) - config.block_size, (config.micro_batch_size,))
            xb = torch.stack([train_data[i:i+config.block_size] for i in ix]).to(device)
            yb = torch.stack([train_data[i+1:i+1+config.block_size] for i in ix]).to(device)

            with torch.amp.autocast('cuda'):
                logits, loss = model(xb, yb)
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Evaluation & Logging
        if iter_step % config.eval_interval == 0 or iter_step == config.max_iters:
            eval_metrics = estimate_phase5_loss(model, train_data, val_data, config, device)
            train_loss_val = eval_metrics['train']
            val_loss_val = eval_metrics['val']
            val_perplexity = calculate_perplexity(val_loss_val)
            val_bpc = calculate_bpc(val_loss_val, num_val_tokens=len(val_tokens), num_val_chars=num_val_chars)
            elapsed = time.time() - start_time

            record = {
                "step": iter_step,
                "train_loss": round(train_loss_val, 4),
                "val_loss": round(val_loss_val, 4),
                "val_perplexity": round(val_perplexity, 4),
                "val_bpc": round(val_bpc, 4),
                "learning_rate": round(lr, 6),
                "elapsed_sec": round(elapsed, 2)
            }
            history.append(record)

            print(f"Step {iter_step:4d}/{config.max_iters} | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss_val:.4f} | Val PPL: {val_perplexity:.4f} | Val BPC: {val_bpc:.4f} | LR: {lr:.6f} | Elapsed: {elapsed:.2f}s", flush=True)

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_ckpt_path = os.path.join(config.checkpoints_dir, "best_model.pt")
                save_phase5_checkpoint(
                    best_ckpt_path, model, optimizer, config, iter_step,
                    val_loss_val, best_val_loss, history, val_bpc=val_bpc
                )

    total_time = time.time() - start_time
    total_tokens = config.max_iters * config.effective_batch_size * config.block_size
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    # 7. Save Final Checkpoints
    final_ckpt_path = os.path.join(config.checkpoints_dir, "final_model.pt")
    compat_ckpt_path = os.path.join(config.checkpoints_dir, "checkpoint.pt")
    save_phase5_checkpoint(final_ckpt_path, model, optimizer, config, config.max_iters, val_loss_val, best_val_loss, history, val_bpc=val_bpc)
    save_phase5_checkpoint(compat_ckpt_path, model, optimizer, config, config.max_iters, val_loss_val, best_val_loss, history, val_bpc=val_bpc)

    # 8. Save Model C History JSON
    history_path = os.path.join(config.results_dir, "model_c_gpu_history.json")
    history_data = {
        "model_name": "Model C (Phase 5D-B GPU)",
        "gpu": gpu_name,
        "cuda_version": cuda_version,
        "num_params": num_params,
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "max_iters": config.max_iters,
            "learning_rate": config.learning_rate,
            "seed": config.random_seed
        },
        "history": history,
        "summary": {
            "best_val_loss": round(best_val_loss, 4),
            "best_val_perplexity": round(calculate_perplexity(best_val_loss), 4),
            "final_train_loss": round(train_loss_val, 4),
            "final_val_loss": round(val_loss_val, 4),
            "final_val_perplexity": round(val_perplexity, 4),
            "final_val_bpc": round(val_bpc, 4),
            "wall_clock_time_sec": round(total_time, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_tokens": total_tokens
        }
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    print("\n=== Model C Training Complete ===", flush=True)
    print(f"GPU: {gpu_name}", flush=True)
    print(f"Parameters: {num_params:,}", flush=True)
    print(f"Wall Clock Time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"Tokens/sec: {tokens_per_sec:.2f}", flush=True)
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val PPL: {calculate_perplexity(best_val_loss):.4f}", flush=True)
    print(f"Final Val Loss: {val_loss_val:.4f} | Final Val PPL: {val_perplexity:.4f} | Final Val BPC: {val_bpc:.4f}", flush=True)
    print(f"History Saved: {history_path}", flush=True)

if __name__ == "__main__":
    main()

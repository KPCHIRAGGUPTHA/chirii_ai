import os
import time
import math
import json
import torch
import torch.nn as nn

from model import MiniGPT, MiniGPTConfig
from phase5_config import Phase5Config
from data_cleaner import DataCleaner
from phase5_dataset import (
    prepare_phase5_corpus,
    get_batch_phase5
)
from phase5_tokenizer import load_phase5_tokenizer
from train import calculate_perplexity, calculate_bpc, set_seed

def get_lr(step: int, max_iters: int, config: Phase5Config) -> float:
    """
    Compute learning rate with linear warmup followed by cosine decay.
    """
    if step < config.warmup_iters:
        return config.learning_rate * (step / max(1, config.warmup_iters))
    if step > max_iters:
        return config.min_lr
    decay_ratio = (step - config.warmup_iters) / max(1, max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def save_phase5_checkpoint(
    filepath: str,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    config: Phase5Config,
    step: int,
    val_loss: float,
    best_val_loss: float,
    history: list,
    val_bpc: float = None
):
    """Save isolated Phase 5 checkpoint with complete state."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    model_config = MiniGPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    checkpoint = {
        "phase": 5,
        "config": model_config,
        "phase5_config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "val_perplexity": calculate_perplexity(val_loss),
        "val_bpc": val_bpc,
        "history": history,
        "seed": config.random_seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, filepath)

@torch.no_grad()
def estimate_phase5_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Phase5Config,
    device: str
) -> dict:
    """Estimate average training and validation loss over multiple batches."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters, device=device)
        for k in range(config.eval_iters):
            X, Y = get_batch_phase5(data, config.block_size, config.micro_batch_size, device=device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_phase5(
    config: Phase5Config,
    max_iters: int = 600,
    resume_ckpt_path: str = None,
    tokenizer = None,
    train_text: str = None,
    val_text: str = None,
    train_tokens: list = None,
    val_tokens: list = None
):
    """
    Execute Phase 5 pretraining on FineWeb-Edu with gradient accumulation,
    cosine learning rate schedule, AMP mixed precision (CUDA), and checkpoint resume support.
    """
    set_seed(config.random_seed)
    config.create_dirs()

    # Hardware & Device Inspection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Phase 5 Compute Device: {device.upper()}", flush=True)
    if device == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA Version: {torch.version.cuda}", flush=True)
        scaler = torch.amp.GradScaler('cuda')
        use_amp = True
    else:
        print("Running on CPU (FP32 standard precision).", flush=True)
        scaler = None
        use_amp = False

    # Load Phase 5 BPE Tokenizer
    if tokenizer is None:
        tokenizer = load_phase5_tokenizer(config, vocab_size=config.vocab_size)
    
    # Sync config vocabulary size with actual trained tokenizer vocabulary size
    config.vocab_size = tokenizer.vocab_size

    # Save vocab.json in checkpoints_dir for generate_text compatibility
    ckpt_vocab_path = os.path.join(config.checkpoints_dir, "vocab.json")
    tokenizer.save(ckpt_vocab_path)
    print(f"Loaded Phase 5 Tokenizer (vocab_size={tokenizer.vocab_size}), saved to {ckpt_vocab_path}", flush=True)

    # Encode train and val text independently if not pre-provided
    train_tokens_path = os.path.join(config.data_dir, f"train_tokens_v{tokenizer.vocab_size}.pt")
    val_tokens_path = os.path.join(config.data_dir, f"val_tokens_v{tokenizer.vocab_size}.pt")

    if train_tokens is None and os.path.exists(train_tokens_path):
        print(f"Loading cached encoded train tokens from {train_tokens_path}...", flush=True)
        train_data = torch.load(train_tokens_path, weights_only=True)
        train_tokens = train_data.tolist()
    
    if val_tokens is None and os.path.exists(val_tokens_path):
        print(f"Loading cached encoded val tokens from {val_tokens_path}...", flush=True)
        val_data = torch.load(val_tokens_path, weights_only=True)
        val_tokens = val_data.tolist()

    if train_tokens is None or val_tokens is None:
        if train_text is None or val_text is None:
            train_text, val_text, _ = prepare_phase5_corpus(config)
        if train_tokens is None:
            print("Encoding training corpus with BPE tokenizer...", flush=True)
            train_tokens = tokenizer.encode(train_text)
            torch.save(torch.tensor(train_tokens, dtype=torch.long), train_tokens_path)
            print(f"Cached encoded train tokens to {train_tokens_path}", flush=True)
        if val_tokens is None:
            print("Encoding validation corpus with BPE tokenizer...", flush=True)
            val_tokens = tokenizer.encode(val_text)
            torch.save(torch.tensor(val_tokens, dtype=torch.long), val_tokens_path)
            print(f"Cached encoded val tokens to {val_tokens_path}", flush=True)

    train_data = torch.tensor(train_tokens, dtype=torch.long)
    val_data = torch.tensor(val_tokens, dtype=torch.long)

    # Instantiate Model
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

    start_step = 1
    history = []
    best_val_loss = float('inf')

    # Handle Checkpoint Resume
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"Resuming Phase 5 training from checkpoint: {resume_ckpt_path}", flush=True)
        checkpoint = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint.get("step", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        history = checkpoint.get("history", [])
        print(f"Resumed at step {start_step} with best_val_loss {best_val_loss:.4f}", flush=True)

    print(f"MiniGPT Parameters: {model.get_num_params():,}", flush=True)
    print(f"Effective Batch Size: {config.effective_batch_size} (Micro-batch: {config.micro_batch_size}, Grad Accum Steps: {config.gradient_accumulation_steps})", flush=True)
    print(f"Training from step {start_step} to {max_iters}...", flush=True)

    start_time = time.time()
    val_loss_val = 0.0
    val_perplexity = 0.0
    val_bpc = 0.0

    for iter_step in range(start_step, max_iters + 1):
        model.train()
        lr = get_lr(iter_step, max_iters, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        # Gradient Accumulation Loop
        accum_loss = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            xb, yb = get_batch_phase5(train_data, config.block_size, config.micro_batch_size, device=device)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(xb, yb)
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, loss = model(xb, yb)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            accum_loss += loss.item() * config.gradient_accumulation_steps

        # Gradient Clipping & Optimizer Step
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Validation & Logging
        if iter_step % config.eval_interval == 0 or iter_step == max_iters:
            eval_metrics = estimate_phase5_loss(model, train_data, val_data, config, device)
            train_loss_val = eval_metrics['train']
            val_loss_val = eval_metrics['val']
            val_perplexity = calculate_perplexity(val_loss_val)
            val_bpc = calculate_bpc(val_loss_val, num_val_tokens=len(val_tokens), num_val_chars=len(val_text))
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

            print(f"Phase 5 Iter {iter_step:4d}/{max_iters} | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss_val:.4f} | Val PPL: {val_perplexity:.4f} | Val BPC: {val_bpc:.4f} | LR: {lr:.6f} | Time: {elapsed:.2f}s", flush=True)

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_ckpt_path = os.path.join(config.checkpoints_dir, "best_model.pt")
                save_phase5_checkpoint(
                    best_ckpt_path, model, optimizer, config, iter_step,
                    val_loss_val, best_val_loss, history, val_bpc=val_bpc
                )

    # Save Final Checkpoints
    final_ckpt_path = os.path.join(config.checkpoints_dir, "final_model.pt")
    compat_ckpt_path = os.path.join(config.checkpoints_dir, "checkpoint.pt")
    save_phase5_checkpoint(final_ckpt_path, model, optimizer, config, max_iters, val_loss_val, best_val_loss, history, val_bpc=val_bpc)
    save_phase5_checkpoint(compat_ckpt_path, model, optimizer, config, max_iters, val_loss_val, best_val_loss, history, val_bpc=val_bpc)

    # Save History JSON
    history_path = os.path.join(config.results_dir, "training_history.json")
    history_data = {
        "phase": 5,
        "dataset": config.dataset_name,
        "dataset_config": config.dataset_config,
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "effective_batch_size": config.effective_batch_size,
            "max_iters": max_iters,
            "learning_rate": config.learning_rate,
            "seed": config.random_seed
        },
        "history": history,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_perplexity": round(calculate_perplexity(best_val_loss), 4),
        "final_val_loss": round(val_loss_val, 4),
        "final_val_perplexity": round(val_perplexity, 4)
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    return model, tokenizer, history_data

import os
import time
import math
import json
import random
import argparse
import torch
from tokenizer import CharTokenizer
from model import MiniGPT, MiniGPTConfig
from dataset import get_or_download_text, get_batch

def set_seed(seed: int = 42):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducible runs.
    Note: Exact numeric reproducibility across different hardware or PyTorch backends
    may vary slightly due to floating point operations.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, block_size: int, batch_size: int, eval_iters: int, device: str) -> dict:
    """
    Estimate average training and validation loss over multiple batches.
    Ensures model returns to training mode afterward.
    """
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def calculate_perplexity(val_loss: float) -> float:
    """Calculate validation perplexity as exp(val_loss) with numeric safety handling."""
    safe_loss = min(max(val_loss, 0.0), 50.0)
    return math.exp(safe_loss)

def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: MiniGPTConfig,
    tokenizer: CharTokenizer,
    step: int,
    val_loss: float,
    best_val_loss: float,
    history: list,
    seed: int
):
    """Save model checkpoint with full state and configuration metadata."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    checkpoint = {
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "val_perplexity": calculate_perplexity(val_loss),
        "history": history,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, filepath)

def train_model(
    data_path: str = None,
    out_dir: str = "checkpoints",
    results_dir: str = "results",
    max_iters: int = 600,
    batch_size: int = 32,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    eval_interval: int = 50,
    eval_iters: int = 20,
    random_seed: int = 42,
    callback=None
):
    """
    Train Mini-GPT on specified dataset with multi-batch evaluation, perplexity, and checkpointing.
    """
    set_seed(random_seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using compute device: {device} | Random seed: {random_seed}")

    # Load text
    if data_path and os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = get_or_download_text()

    # Train/Val split (90% train, 10% validation)
    tokenizer = CharTokenizer.from_text(text)
    vocab_path = os.path.join(out_dir, "vocab.json")
    tokenizer.save(vocab_path)
    print(f"Saved tokenizer vocab ({tokenizer.vocab_size} tokens) to {vocab_path}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Configure Model
    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Starting training for {max_iters} iterations (eval interval: {eval_interval}, eval iters: {eval_iters})...")

    start_time = time.time()
    history = []
    best_val_loss = float('inf')

    for iter_step in range(1, max_iters + 1):
        model.train()
        xb, yb = get_batch(train_data, config.block_size, batch_size, device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iter_step % eval_interval == 0 or iter_step == max_iters:
            eval_metrics = estimate_loss(model, train_data, val_data, config.block_size, batch_size, eval_iters, device)
            train_loss_val = eval_metrics['train']
            val_loss_val = eval_metrics['val']
            val_perplexity = calculate_perplexity(val_loss_val)
            elapsed = time.time() - start_time

            record = {
                "step": iter_step,
                "train_loss": round(train_loss_val, 4),
                "val_loss": round(val_loss_val, 4),
                "val_perplexity": round(val_perplexity, 4),
                "learning_rate": learning_rate,
                "elapsed_sec": round(elapsed, 2)
            }
            history.append(record)

            print(f"Iter {iter_step:4d}/{max_iters} | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss_val:.4f} | Val Perplexity: {val_perplexity:.4f} | Time: {elapsed:.2f}s")

            # Check for best model update
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_ckpt_path = os.path.join(out_dir, "best_model.pt")
                save_checkpoint(best_ckpt_path, model, optimizer, config, tokenizer, iter_step, val_loss_val, best_val_loss, history, random_seed)
                print(f" -> Saved new best model checkpoint to {best_ckpt_path} (val_loss: {val_loss_val:.4f})")

            if callback:
                callback({
                    "step": iter_step,
                    "max_iters": max_iters,
                    "train_loss": round(train_loss_val, 4),
                    "val_loss": round(val_loss_val, 4),
                    "val_perplexity": round(val_perplexity, 4),
                    "best_val_loss": round(best_val_loss, 4),
                    "elapsed_sec": round(elapsed, 2)
                })

    # Save Final Model and Backwards-Compatible Checkpoint
    final_ckpt_path = os.path.join(out_dir, "final_model.pt")
    compat_ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    save_checkpoint(final_ckpt_path, model, optimizer, config, tokenizer, max_iters, val_loss_val, best_val_loss, history, random_seed)
    save_checkpoint(compat_ckpt_path, model, optimizer, config, tokenizer, max_iters, val_loss_val, best_val_loss, history, random_seed)
    print(f"Training complete! Saved final checkpoint to {final_ckpt_path} and {compat_ckpt_path}")

    # Save Machine-Readable Training History
    history_path = os.path.join(results_dir, "training_history.json")
    history_data = {
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "dropout": config.dropout,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "learning_rate": learning_rate,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "random_seed": random_seed
        },
        "history": history,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_perplexity": round(calculate_perplexity(best_val_loss), 4),
        "final_val_loss": round(val_loss_val, 4),
        "final_val_perplexity": round(val_perplexity, 4)
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved structured training history to {history_path}")

    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mini-GPT Model")
    parser.add_argument("--data", type=str, default=None, help="Path to custom text file")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save training history JSON")
    parser.add_argument("--max_iters", type=int, default=600, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context block size")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of Transformer blocks")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--eval_iters", type=int, default=20, help="Number of validation batches for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        out_dir=args.out_dir,
        results_dir=args.results_dir,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        random_seed=args.seed
    )

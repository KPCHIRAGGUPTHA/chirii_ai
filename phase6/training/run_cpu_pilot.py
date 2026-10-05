import os
import sys
import time
import math
import json
import hashlib
import torch
try:
    import psutil
except ImportError:
    psutil = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT
from phase6.training.sft_config import Phase6SFTConfig
from phase6.training.sft_dataset import create_sft_dataloader
from phase6.training.sft_trainer import SFTTrainer, estimate_sft_loss

def get_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=== PHASE 6E — CPU SFT PILOT ===", flush=True)

    config = Phase6SFTConfig()
    # Force max_iters to 50 for pilot only
    config.max_iters = 50

    ckpt_path = os.path.join(REPO_ROOT, config.base_checkpoint_path)
    vocab_path = os.path.join(REPO_ROOT, config.tokenizer_path)

    sha_before = get_file_sha256(ckpt_path)
    print(f"Pre-pilot Base Checkpoint SHA-256: {sha_before}")

    # Ensure device is strictly CPU
    device = "cpu"
    torch.set_num_threads(8)

    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")

    # Load Base Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config") or config.get_minigpt_config()
    model = MiniGPT(model_config)
    state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    param_count = model.get_num_params()
    print(f"Model Parameters: {param_count:,}")

    train_jsonl = os.path.join(REPO_ROOT, config.data_sft_dir, "train_sft.jsonl")
    val_jsonl = os.path.join(REPO_ROOT, config.data_sft_dir, "val_sft.jsonl")

    trainer = SFTTrainer(
        config=config,
        model=model,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        device=device
    )

    # Calculate initial validation loss before any training
    print("Evaluating initial validation loss...", flush=True)
    initial_val_loss = estimate_sft_loss(model, trainer.val_loader, device=device, max_batches=20)
    initial_val_ppl = math.exp(initial_val_loss)
    print(f"Initial Val Loss: {initial_val_loss:.4f} | Initial Val PPL: {initial_val_ppl:.4f}")

    print("\nStarting 50-step CPU SFT Pilot...", flush=True)

    train_iter = iter(trainer.train_loader)
    step_times = []
    initial_train_loss = None
    final_train_loss = None

    tokens_per_micro_batch = config.micro_batch_size * config.block_size
    tokens_per_opt_step = tokens_per_micro_batch * config.gradient_accumulation_steps
    total_tokens_processed = 0

    pilot_start_time = time.time()

    model.train()
    for opt_step in range(1, 51):
        step_start = time.time()
        trainer.optimizer.zero_grad()

        lr = trainer.get_lr(opt_step)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr

        step_loss_sum = 0.0
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(trainer.train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            step_loss_sum += loss.item() * config.gradient_accumulation_steps
            total_tokens_processed += (y != -100).sum().item()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        trainer.optimizer.step()

        step_duration = time.time() - step_start
        step_times.append(step_duration)

        if initial_train_loss is None:
            initial_train_loss = step_loss_sum
        final_train_loss = step_loss_sum

        if opt_step % 10 == 0 or opt_step == 1 or opt_step == 50:
            print(f"Step {opt_step:2d}/50 | Loss: {step_loss_sum:.4f} | LR: {lr:.6f} | Step Time: {step_duration:.2f}s")

    total_elapsed_time = time.time() - pilot_start_time
    avg_sec_per_step = sum(step_times) / len(step_times)

    print("\nEvaluating final validation loss...", flush=True)
    final_val_loss = estimate_sft_loss(model, trainer.val_loader, device=device, max_batches=20)
    final_val_ppl = math.exp(final_val_loss)

    if psutil is not None:
        process = psutil.Process(os.getpid())
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    else:
        ram_usage_mb = 0.0

    # Save pilot checkpoint separately
    pilot_ckpt_dir = os.path.join(REPO_ROOT, config.checkpoints_dir)
    os.makedirs(pilot_ckpt_dir, exist_ok=True)
    pilot_ckpt_path = os.path.join(pilot_ckpt_dir, "cpu_pilot_50steps.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "sft_config": config,
        "opt_step": 50,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss
    }, pilot_ckpt_path)
    print(f"Saved pilot checkpoint to {pilot_ckpt_path}")

    sha_after = get_file_sha256(ckpt_path)
    sha_matched = (sha_before == sha_after)

    summary = {
        "cpu_threads": torch.get_num_threads(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_parameters": param_count,
        "opt_steps": 50,
        "total_elapsed_time_sec": round(total_elapsed_time, 2),
        "avg_sec_per_step": round(avg_sec_per_step, 2),
        "initial_train_loss": round(initial_train_loss, 4),
        "final_train_loss": round(final_train_loss, 4),
        "initial_val_loss": round(initial_val_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "initial_val_ppl": round(initial_val_ppl, 4),
        "final_val_ppl": round(final_val_ppl, 4),
        "approx_active_tokens_processed": total_tokens_processed,
        "ram_usage_mb": round(ram_usage_mb, 2),
        "sha_before": sha_before,
        "sha_after": sha_after,
        "base_ckpt_unchanged": sha_matched,
        "estimated_sec_1000_steps": round(avg_sec_per_step * 1000, 2)
    }

    out_summary_path = os.path.join(REPO_ROOT, "phase6", "audit", "sft_cpu_pilot_summary.json")
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PILOT COMPLETE ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

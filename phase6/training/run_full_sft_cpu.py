import os
import sys
import time
import math
import json
import hashlib
import torch

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
    print("=== PHASE 6E — FULL 1000-STEP CPU SFT TRAINING ===", flush=True)

    config = Phase6SFTConfig()
    config.max_iters = 1000

    ckpt_path = os.path.join(REPO_ROOT, config.base_checkpoint_path)
    vocab_path = os.path.join(REPO_ROOT, config.tokenizer_path)
    pilot_path = os.path.join(REPO_ROOT, config.checkpoints_dir, "cpu_pilot_50steps.pt")

    # 1. Pre-training SHA & File Integrity Check
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base checkpoint missing: {ckpt_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Tokenizer missing: {vocab_path}")
    if not os.path.exists(pilot_path):
        raise FileNotFoundError(f"Pilot checkpoint missing: {pilot_path}")

    base_sha_before = get_file_sha256(ckpt_path)
    vocab_sha_before = get_file_sha256(vocab_path)
    pilot_sha_before = get_file_sha256(pilot_path)

    expected_base_sha = "6f934bc3f2ca1cff6173896562cc215df5467274337c7af3fb265fb399fd2433"
    expected_vocab_sha = "6436b59303ef54cb91c6656f378f5a9bfb00fc55a3c2ad516ddf357cc58e3a5d"

    assert base_sha_before == expected_base_sha, f"Base SHA mismatch! Got {base_sha_before}"
    assert vocab_sha_before == expected_vocab_sha, f"Vocab SHA mismatch! Got {vocab_sha_before}"
    assert not torch.cuda.is_available(), "CUDA MUST be False for CPU training!"

    print(f"Base Checkpoint SHA Verified: {base_sha_before[:16]}...")
    print(f"Tokenizer SHA Verified:       {vocab_sha_before[:16]}...")
    print(f"Pilot Checkpoint SHA Saved:   {pilot_sha_before[:16]}...")
    print(f"CUDA Available: {torch.cuda.is_available()} (Strict CPU Mode)")

    device = "cpu"
    torch.set_num_threads(8)
    print(f"PyTorch CPU Threads: {torch.get_num_threads()}")

    # 2. Load Model C Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config") or config.get_minigpt_config()
    model = MiniGPT(model_config)
    state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    param_count = model.get_num_params()
    print(f"Model Parameters: {param_count:,}")
    assert param_count == 6613504, f"Parameter count mismatch! Expected 6,613,504, got {param_count}"

    train_jsonl = os.path.join(REPO_ROOT, config.data_sft_dir, "train_sft.jsonl")
    val_jsonl = os.path.join(REPO_ROOT, config.data_sft_dir, "val_sft.jsonl")

    trainer = SFTTrainer(
        config=config,
        model=model,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        device=device
    )

    # Calculate initial validation loss before training
    print("\nEvaluating initial validation loss before SFT...", flush=True)
    initial_val_loss = estimate_sft_loss(model, trainer.val_loader, device=device, max_batches=20)
    initial_val_ppl = math.exp(initial_val_loss)
    print(f"Initial Val Loss: {initial_val_loss:.4f} | Initial Val PPL: {initial_val_ppl:.4f}")

    print("\nStarting Full 1000-Step SFT Training on CPU...", flush=True)

    train_iter = iter(trainer.train_loader)
    step_times = []
    initial_train_loss = None
    final_train_loss = None

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_step = 0

    history = []
    total_tokens_processed = 0
    training_start_time = time.time()

    model.train()
    for opt_step in range(1, 1001):
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

        # Clip gradients & step optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        trainer.optimizer.step()

        step_duration = time.time() - step_start
        step_times.append(step_duration)

        if initial_train_loss is None:
            initial_train_loss = step_loss_sum
        final_train_loss = step_loss_sum

        # Periodic Validation & Logging every 50 steps (and step 1, step 1000)
        is_eval_step = (opt_step % 50 == 0 or opt_step == 1 or opt_step == 1000)
        val_loss_curr = None
        val_ppl_curr = None

        if is_eval_step:
            val_loss_curr = estimate_sft_loss(model, trainer.val_loader, device=device, max_batches=20)
            val_ppl_curr = math.exp(val_loss_curr)

            if val_loss_curr < best_val_loss:
                best_val_loss = val_loss_curr
                best_val_ppl = val_ppl_curr
                best_step = opt_step
                best_ckpt_path = os.path.join(REPO_ROOT, config.checkpoints_dir, "best_model.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "sft_config": config,
                    "step": opt_step,
                    "val_loss": best_val_loss,
                    "val_ppl": best_val_ppl
                }, best_ckpt_path)

            print(f"Step {opt_step:4d}/1000 | Train Loss: {step_loss_sum:.4f} | Val Loss: {val_loss_curr:.4f} | Val PPL: {val_ppl_curr:.4f} | LR: {lr:.6f} | Step Time: {step_duration:.2f}s", flush=True)
        elif opt_step % 10 == 0:
            print(f"Step {opt_step:4d}/1000 | Train Loss: {step_loss_sum:.4f} | LR: {lr:.6f} | Step Time: {step_duration:.2f}s", flush=True)

        history.append({
            "step": opt_step,
            "train_loss": round(step_loss_sum, 4),
            "val_loss": round(val_loss_curr, 4) if val_loss_curr is not None else None,
            "val_ppl": round(val_ppl_curr, 4) if val_ppl_curr is not None else None,
            "lr": round(lr, 6),
            "step_time_sec": round(step_duration, 2)
        })

    total_elapsed_time = time.time() - training_start_time
    avg_sec_per_step = sum(step_times) / len(step_times)

    # Save Final Checkpoint
    final_ckpt_path = os.path.join(REPO_ROOT, config.checkpoints_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_config,
        "sft_config": config,
        "step": 1000,
        "train_loss": final_train_loss,
        "val_loss": best_val_loss,
        "val_ppl": best_val_ppl
    }, final_ckpt_path)
    print(f"\nSaved final SFT checkpoint to {final_ckpt_path}")

    # 3. Post-Training Integrity Checks
    base_sha_after = get_file_sha256(ckpt_path)
    vocab_sha_after = get_file_sha256(vocab_path)
    pilot_sha_after = get_file_sha256(pilot_path)

    assert base_sha_before == base_sha_after, "CRITICAL ERROR: Base checkpoint was modified!"
    assert vocab_sha_before == vocab_sha_after, "CRITICAL ERROR: Tokenizer was modified!"
    assert pilot_sha_before == pilot_sha_after, "CRITICAL ERROR: Pilot checkpoint was modified!"

    sft_best_sha = get_file_sha256(best_ckpt_path)
    sft_final_sha = get_file_sha256(final_ckpt_path)

    # Test loading final checkpoint
    test_load = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
    test_model = MiniGPT(test_load["config"])
    test_model.load_state_dict(test_load["model_state_dict"])
    print("Successfully verified loading final SFT checkpoint!")

    summary = {
        "base_model": config.base_checkpoint_path,
        "model_parameters": param_count,
        "training_steps": 1000,
        "train_examples": 41404,
        "val_examples": 5176,
        "initial_train_loss": round(initial_train_loss, 4),
        "final_train_loss": round(final_train_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_ppl": round(best_val_ppl, 4),
        "best_val_step": best_step,
        "total_training_time_sec": round(total_elapsed_time, 2),
        "total_training_time_min": round(total_elapsed_time / 60, 2),
        "avg_sec_per_step": round(avg_sec_per_step, 2),
        "approx_tokens_processed": total_tokens_processed,
        "base_checkpoint_sha_before": base_sha_before,
        "base_checkpoint_sha_after": base_sha_after,
        "base_checkpoint_unchanged": base_sha_before == base_sha_after,
        "tokenizer_sha": vocab_sha_after,
        "pilot_checkpoint_sha_before": pilot_sha_before,
        "pilot_checkpoint_sha_after": pilot_sha_after,
        "pilot_checkpoint_unchanged": pilot_sha_before == pilot_sha_after,
        "sft_best_checkpoint_path": "checkpoints/phase6/model_c_sft/best_model.pt",
        "sft_best_checkpoint_sha": sft_best_sha,
        "sft_final_checkpoint_path": "checkpoints/phase6/model_c_sft/final_model.pt",
        "sft_final_checkpoint_sha": sft_final_sha
    }

    out_summary_path = os.path.join(REPO_ROOT, "results", "phase6", "sft_summary.json")
    out_history_path = os.path.join(REPO_ROOT, "results", "phase6", "sft_training_history.json")
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n=== FULL SFT TRAINING COMPLETE ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

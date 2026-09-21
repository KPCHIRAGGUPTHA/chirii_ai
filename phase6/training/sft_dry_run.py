import os
import sys
import json
import hashlib
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase6.training.sft_config import Phase6SFTConfig
from phase6.training.sft_dataset import SFTDataset, create_sft_dataloader
from phase6.training.sft_trainer import calculate_sft_loss, estimate_sft_loss, SFTTrainer

def get_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=== Phase 6C: SFT Pipeline Dry Run & Overfitting Sanity Test ===", flush=True)

    config = Phase6SFTConfig()

    # 1. Pre-execution SHA-256 Hash Verification
    ckpt_path = os.path.join(REPO_ROOT, config.base_checkpoint_path)
    vocab_path = os.path.join(REPO_ROOT, config.tokenizer_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model C checkpoint missing: {ckpt_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Tokenizer file missing: {vocab_path}")

    ckpt_hash_before = get_file_sha256(ckpt_path)
    vocab_hash_before = get_file_sha256(vocab_path)
    print(f"Pre-execution SHA-256 Checkpoint: {ckpt_hash_before[:16]}...")
    print(f"Pre-execution SHA-256 Tokenizer:  {vocab_hash_before[:16]}...")

    # 2. Load Model C Checkpoint into a TEST Model Instance
    print("\nLoading Model C checkpoint into TEST model instance...", flush=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config") or config.get_minigpt_config()
    model = MiniGPT(model_config)

    state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)

    param_count = model.get_num_params()
    print(f"Loaded Model C Parameters: {param_count:,}")
    assert param_count == 6613504, f"Parameter count mismatch! Expected 6,613,504, got {param_count}"
    assert model.config.vocab_size == 1024, f"Vocab size mismatch! Expected 1024, got {model.config.vocab_size}"
    assert model.config.block_size == 128, f"Block size mismatch! Expected 128, got {model.config.block_size}"
    assert model.config.n_layer == 8 and model.config.n_head == 8 and model.config.n_embd == 256

    # 3. Load Tokenizer & SFT Data
    tokenizer = BPETokenizer.load(vocab_path)
    assert tokenizer.vocab_size == 1024

    train_sft_path = os.path.join(REPO_ROOT, config.data_sft_dir, "train_sft.jsonl")
    val_sft_path = os.path.join(REPO_ROOT, config.data_sft_dir, "val_sft.jsonl")
    
    if not os.path.exists(train_sft_path):
        raise FileNotFoundError(f"SFT train data missing at {train_sft_path}")

    train_loader = create_sft_dataloader(train_sft_path, batch_size=config.micro_batch_size, block_size=config.block_size, shuffle=False)

    # 4. CPU Dry-Run (1 Batch Forward, Loss, Backward, Optimizer Step)
    print("\nExecuting CPU 1-Batch Dry Run...", flush=True)
    model.train()
    x, y = next(iter(train_loader))

    batch_size, seq_len = x.shape
    total_labels = y.numel()
    masked_labels = (y == -100).sum().item()
    active_labels = (y != -100).sum().item()
    contributing_pct = round((active_labels / total_labels) * 100, 2)

    logits, dry_run_loss = model(x, y)
    initial_loss_val = round(dry_run_loss.item(), 4)

    # Backward pass
    dry_run_loss.backward()

    # Verify gradients exist
    grad_exists = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
    grad_norm_val = round(total_norm, 4)

    # Optimizer step on test instance
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.step()
    optimizer.zero_grad()

    print(f"Dry Run Diagnostics:")
    print(f"  Batch Shape: {list(x.shape)}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Masked Prompt/Pad Labels (-100): {masked_labels} / {total_labels}")
    print(f"  Active Response Labels: {active_labels} / {total_labels} ({contributing_pct}%)")
    print(f"  Initial Dry-Run Loss: {initial_loss_val}")
    print(f"  Gradients Verified Exist: {grad_exists}")
    print(f"  Gradient Norm: {grad_norm_val}")

    # 5. Overfitting Sanity Test (4 Examples, CPU)
    print("\nExecuting 4-Example CPU Overfitting Sanity Test...", flush=True)
    tiny_dataset = SFTDataset(train_sft_path, block_size=config.block_size)
    tiny_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(tiny_dataset, range(min(4, len(tiny_dataset)))),
        batch_size=4,
        shuffle=False
    )

    sanity_model = MiniGPT(model_config)
    sanity_model.load_state_dict(state_dict)
    sanity_optim = torch.optim.AdamW(sanity_model.parameters(), lr=1e-4)

    sanity_x, sanity_y = next(iter(tiny_loader))
    sanity_history = []

    for step in range(16):
        sanity_optim.zero_grad()
        s_logits, s_loss = sanity_model(sanity_x, sanity_y)
        s_loss.backward()
        sanity_optim.step()
        sanity_history.append(round(s_loss.item(), 4))

    initial_sanity_loss = sanity_history[0]
    final_sanity_loss = sanity_history[-1]
    loss_reduced = final_sanity_loss < initial_sanity_loss

    print(f"Overfitting Sanity Test (4 Examples, 15 Steps):")
    print(f"  Initial Loss (Step 0):  {initial_sanity_loss}")
    print(f"  Final Loss   (Step 15): {final_sanity_loss}")
    print(f"  Loss Reduced: {loss_reduced} (from {initial_sanity_loss} -> {final_sanity_loss})")

    # 6. Post-execution SHA-256 Checkpoint & Tokenizer Hash Verification
    ckpt_hash_after = get_file_sha256(ckpt_path)
    vocab_hash_after = get_file_sha256(vocab_path)

    assert ckpt_hash_before == ckpt_hash_after, "CRITICAL SECURITY ERROR: Checkpoint was modified!"
    assert vocab_hash_before == vocab_hash_after, "CRITICAL SECURITY ERROR: Tokenizer was modified!"
    print("\nPost-execution File Integrity Verification:")
    print("  Checkpoint SHA-256: 100% BYTE-FOR-BYTE IDENTICAL.")
    print("  Tokenizer SHA-256:  100% BYTE-FOR-BYTE IDENTICAL.")

    # 7. Save Results JSON
    results = {
        "model_path": config.base_checkpoint_path,
        "parameter_count": param_count,
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "batch_shape": list(x.shape),
        "sequence_length": seq_len,
        "masked_labels_count": masked_labels,
        "active_labels_count": active_labels,
        "active_labels_pct": contributing_pct,
        "initial_dry_run_loss": initial_loss_val,
        "gradients_verified_exist": grad_exists,
        "gradient_norm": grad_norm_val,
        "tiny_sanity_initial_loss": initial_sanity_loss,
        "tiny_sanity_final_loss": final_sanity_loss,
        "tiny_sanity_loss_reduced": loss_reduced,
        "checkpoint_sha256_before": ckpt_hash_before,
        "checkpoint_sha256_after": ckpt_hash_after,
        "checkpoint_byte_identical": ckpt_hash_before == ckpt_hash_after,
        "tokenizer_sha256_before": vocab_hash_before,
        "tokenizer_sha256_after": vocab_hash_after,
        "tokenizer_byte_identical": vocab_hash_before == vocab_hash_after,
        "full_sft_executed": False
    }

    out_json = os.path.join(REPO_ROOT, "phase6", "audit", "sft_dry_run_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved dry run diagnostics to {out_json}", flush=True)

if __name__ == "__main__":
    main()

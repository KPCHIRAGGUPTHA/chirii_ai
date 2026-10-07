import os
import sys
import json
import time
import math
import hashlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase7.rag.rag_sft_config import RAGSFTConfig
from phase7.rag.rag_sft_dataset import RAGSFTDataset
from phase7.rag.evaluate_rag_sft import evaluate_model_on_dataset

def get_file_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "MISSING"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def evaluate_loss(model: MiniGPT, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            _, loss = model(input_ids, targets)
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)

def train_and_evaluate_rag_sft(config: RAGSFTConfig = None, execute: bool = False):
    if config is None:
        config = RAGSFTConfig()

    base_ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
    sft_ckpt_path = config.starting_ckpt_path

    # Step 1: Pre-flight Integrity & Hash Verification
    print("=== STEP 1: PRE-FLIGHT INTEGRITY VERIFICATION ===")
    sha_base_before = get_file_sha256(base_ckpt_path)
    sha_sft_before = get_file_sha256(sft_ckpt_path)
    print(f"Phase 5D Base Checkpoint SHA-256: {sha_base_before[:16]}...")
    print(f"Phase 6 SFT Checkpoint SHA-256:   {sha_sft_before[:16]}...")

    # Load test checkpoints
    ckpt_base = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_sft = torch.load(sft_ckpt_path, map_location="cpu", weights_only=False)
    print("Pre-flight check: Both Phase 5D and Phase 6 checkpoints loaded successfully.")

    # Dataset integrity check
    for p_name, p_path, exp_cnt in [
        ("train", config.data_train_path, 64),
        ("val", config.data_val_path, 8),
        ("test", config.data_test_path, 8),
    ]:
        with open(p_path, "r", encoding="utf-8") as f:
            cnt = len([l for l in f if l.strip()])
        assert cnt == exp_cnt, f"{p_name} dataset count {cnt} != {exp_cnt}"
    print("Pre-flight check: Dataset split integrity verified (64 train / 8 val / 8 test).")

    if not execute:
        print("\n[SAFETY GUARD] Execute flag is False. Stopping before training.")
        return

    # Step 2: Training Execution
    print("\n=== STEP 2: RAG-SFT TRAINING EXECUTION (100 OPTIMIZER STEPS) ===")
    torch.manual_seed(config.seed)
    tokenizer = BPETokenizer.load(config.tokenizer_path)

    model_config = ckpt_sft.get("config") or MiniGPTConfig(
        vocab_size=1024, block_size=128, n_layer=8, n_head=8, n_embd=256
    )
    model = MiniGPT(model_config)
    state_dict = ckpt_sft.get("model_state") or ckpt_sft.get("model_state_dict") or ckpt_sft
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.train()

    train_dataset = RAGSFTDataset(config.data_train_path, tokenizer, max_block_size=128)
    val_dataset = RAGSFTDataset(config.data_val_path, tokenizer, max_block_size=128)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def get_lr_multiplier(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / float(max(1, config.warmup_steps))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    os.makedirs(config.output_dir, exist_ok=True)
    history = []

    opt_step = 0
    fwd_step = 0
    running_train_loss = 0.0
    best_val_loss = float("inf")

    t_start = time.time()
    train_iter = iter(train_loader)

    print(f"Training parameters: lr={config.learning_rate}, warmup={config.warmup_steps}, steps={config.max_steps}, accum={config.gradient_accumulation_steps}")

    # Initial val loss before training
    initial_val_loss = evaluate_loss(model, val_loader, config.device)
    print(f"Initial Validation Loss (Step 0): {initial_val_loss:.4f}")

    optimizer.zero_grad()

    while opt_step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(config.device)
        targets = batch["targets"].to(config.device)

        logits, loss = model(input_ids, targets)
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        running_train_loss += loss.item() * config.gradient_accumulation_steps
        fwd_step += 1

        if fwd_step % config.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step += 1
            curr_lr = scheduler.get_last_lr()[0]
            avg_train_loss = running_train_loss / float(config.gradient_accumulation_steps)
            running_train_loss = 0.0

            # Evaluate validation loss every 10 optimizer steps
            if opt_step % 10 == 0 or opt_step == config.max_steps:
                val_loss = evaluate_loss(model, val_loader, config.device)
                elapsed = time.time() - t_start

                h_item = {
                    "step": opt_step,
                    "train_loss": round(avg_train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "learning_rate": round(curr_lr, 7),
                    "grad_norm": round(grad_norm, 4),
                    "elapsed_sec": round(elapsed, 2)
                }
                history.append(h_item)

                print(f"Step {opt_step:3d}/{config.max_steps} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Grad Norm: {grad_norm:.4f} | LR: {curr_lr:.7f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = os.path.join(config.output_dir, "best_model.pt")
                    torch.save({
                        "config": model_config,
                        "model_state": model.state_dict(),
                        "step": opt_step,
                        "val_loss": val_loss,
                        "train_loss": avg_train_loss
                    }, best_ckpt_path)

    t_duration = time.time() - t_start

    # Save final model
    final_ckpt_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        "config": model_config,
        "model_state": model.state_dict(),
        "step": opt_step,
        "val_loss": val_loss,
        "train_loss": avg_train_loss
    }, final_ckpt_path)

    # Save training history and config
    with open(os.path.join(config.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(config.output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"\nTraining completed in {t_duration:.2f} seconds.")
    print(f"Best Val Loss: {best_val_loss:.4f} | Final Val Loss: {val_loss:.4f}")

    # Step 3: Post-Training Integrity & Model Verification
    print("\n=== STEP 3: POST-TRAINING INTEGRITY VERIFICATION ===")
    sha_base_after = get_file_sha256(base_ckpt_path)
    sha_sft_after = get_file_sha256(sft_ckpt_path)

    assert sha_base_before == sha_base_after, "CRITICAL ERROR: Base checkpoint was modified!"
    assert sha_sft_before == sha_sft_after, "CRITICAL ERROR: Phase 6 SFT checkpoint was modified!"
    print("Post-flight check: Phase 5D and Phase 6 checkpoints are 100% UNCHANGED.")

    # Load newly saved RAG-SFT model and verify no NaNs
    best_rag_ckpt = torch.load(os.path.join(config.output_dir, "best_model.pt"), map_location="cpu", weights_only=False)
    rag_sft_model = MiniGPT(best_rag_ckpt.get("config") or model_config)
    rag_sft_model.load_state_dict(best_rag_ckpt.get("model_state"))
    rag_sft_model.eval()

    for p_name, p in rag_sft_model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN detected in parameter {p_name}"
        assert not torch.isinf(p).any(), f"Inf detected in parameter {p_name}"
    print("Post-flight check: RAG-SFT checkpoint loaded successfully with zero NaNs/Infs.")

    # Step 4: Three-Condition Evaluation on 8 Held-Out Test Items
    print("\n=== STEP 4: THREE-CONDITION HELD-OUT EVALUATION ===")

    # Load unmodified Phase 6 SFT model for baselines A & B
    sft_model = MiniGPT(model_config)
    sft_model.load_state_dict(ckpt_sft.get("model_state") or ckpt_sft.get("model_state_dict") or ckpt_sft)
    sft_model.eval()

    # Condition A: Phase 6 SFT + Alpaca Prompt + Sampling
    eval_a = evaluate_model_on_dataset(sft_model, tokenizer, config.data_test_path, prompt_style="alpaca", greedy=False)

    # Condition B: Phase 6 SFT + Minimal Prompt + Greedy
    eval_b = evaluate_model_on_dataset(sft_model, tokenizer, config.data_test_path, prompt_style="minimal", greedy=True)

    # Condition C: Phase 7D-C RAG-SFT + Minimal Prompt + Greedy
    eval_c = evaluate_model_on_dataset(rag_sft_model, tokenizer, config.data_test_path, prompt_style="minimal", greedy=True)

    print("\n------------------------------------------------------------")
    print("EVALUATION RESULTS OVER 8 HELD-OUT TEST ITEMS:")
    print("------------------------------------------------------------")
    print(f"A. Phase 6 SFT + Alpaca Prompt:           Exact Match = {eval_a['exact_match_rate_pct']:.1f}% | Jaccard = {eval_a['mean_token_jaccard']:.4f} | Hallucination = {eval_a['hallucination_rate_pct']:.1f}%")
    print(f"B. Phase 6 SFT + Minimal Prompt + Greedy:  Exact Match = {eval_b['exact_match_rate_pct']:.1f}% | Jaccard = {eval_b['mean_token_jaccard']:.4f} | Hallucination = {eval_b['hallucination_rate_pct']:.1f}%")
    print(f"C. Phase 7D-C RAG-SFT + Minimal + Greedy: Exact Match = {eval_c['exact_match_rate_pct']:.1f}% | Jaccard = {eval_c['mean_token_jaccard']:.4f} | Hallucination = {eval_c['hallucination_rate_pct']:.1f}%")

    print("\nDETAILED PER-ITEM GENERATIONS (CONDITION C):")
    for det in eval_c["details"]:
        print(f"Item {det['id']:2d} [{det['category']:13s}] | Match: {str(det['exact_match']):5s} | Target: '{det['target_answer']}' | Gen: '{det['generated_answer']}'")

    exact_c_pct = eval_c["exact_match_rate_pct"]
    hypothesis_supported = exact_c_pct >= 60.0

    print("\n------------------------------------------------------------")
    print(f"HYPOTHESIS DECISION: Exact Match Rate = {exact_c_pct:.1f}%")
    if hypothesis_supported:
        print("RESULT: SUCCESS — Hypothesis SUPPORTED (>= 60.0% Exact Match). Model C learned context copying.")
    else:
        print("RESULT: FAILURE — Hypothesis REJECTED (< 60.0% Exact Match). Model capacity insufficient for copying.")
    print("------------------------------------------------------------")

    # Generate training report markdown
    report_md = f"""# Phase 7D-C: RAG-SFT Experiment Report

## 1. Executive Summary
- **Experiment Purpose:** Evaluate whether context-conditioned SFT fine-tuning enables MiniGPT Model C (6.61M parameters) to extract/copy in-context facts.
- **Training Time:** {t_duration:.2f} seconds ({config.max_steps} steps on CPU).
- **Starting Checkpoint:** `checkpoints/phase6/model_c_sft/best_model.pt`
- **Output Checkpoint:** `checkpoints/phase7/model_c_rag_sft/best_model.pt`
- **Baseline Checkpoints Integrity:** 100% Verified UNCHANGED.

---

## 2. Loss & Convergence Metrics
- **Initial Validation Loss (Step 0):** {initial_val_loss:.4f}
- **Best Validation Loss:** {best_val_loss:.4f}
- **Final Validation Loss:** {val_loss:.4f}

---

## 3. Comparative Evaluation (8 Held-Out Test Items)

| Condition | Exact Match Rate (%) | Mean Token Jaccard | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: |
| **A. Phase 6 SFT (Alpaca Prompt + Sampling)** | {eval_a['exact_match_rate_pct']:.1f}% | {eval_a['mean_token_jaccard']:.4f} | {eval_a['hallucination_rate_pct']:.1f}% |
| **B. Phase 6 SFT (Minimal Prompt + Greedy)** | {eval_b['exact_match_rate_pct']:.1f}% | {eval_b['mean_token_jaccard']:.4f} | {eval_b['hallucination_rate_pct']:.1f}% |
| **C. Phase 7D-C RAG-SFT (Minimal + Greedy)** | **{eval_c['exact_match_rate_pct']:.1f}%** | **{eval_c['mean_token_jaccard']:.4f}** | **{eval_c['hallucination_rate_pct']:.1f}%** |

---

## 4. Hypothesis Verification
- **Success Threshold:** $\\ge 60.0\\%$ Exact Substring Match
- **Failure Threshold:** $< 25.0\\%$ Exact Substring Match
- **Achieved Condition C Exact Match:** **{exact_c_pct:.1f}%**
- **Decision:** **{'SUPPORTED (SUCCESS)' if hypothesis_supported else 'REJECTED (FAILURE)'}**
"""

    report_path = os.path.join(config.output_dir, "training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md.strip() + "\n")
    print(f"\nSaved training report to {report_path}")

if __name__ == "__main__":
    execute_flag = "--execute" in sys.argv
    train_and_evaluate_rag_sft(execute=execute_flag)

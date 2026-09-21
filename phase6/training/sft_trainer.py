import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT
from phase6.training.sft_config import Phase6SFTConfig
from phase6.training.sft_dataset import create_sft_dataloader

def calculate_sft_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate PyTorch CrossEntropyLoss on response targets strictly with ignore_index=-100.
    Prompt tokens masked with -100 do NOT contribute to loss.
    """
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
    return loss

@torch.no_grad()
def estimate_sft_loss(model: MiniGPT, dataloader: torch.utils.data.DataLoader, device: str, max_batches: int = 10) -> float:
    """
    Evaluate mean response-only cross entropy loss over dataloader batches.
    """
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        if loss is not None and not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))

class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer for MiniGPT Model C.
    Maintains isolated Phase 6 output paths and conservative learning rates.
    """
    def __init__(self, config: Phase6SFTConfig, model: MiniGPT, train_jsonl: str, val_jsonl: str, device: str = "cpu"):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.config.create_dirs()

        self.train_loader = create_sft_dataloader(train_jsonl, batch_size=config.micro_batch_size, block_size=config.block_size, shuffle=True)
        self.val_loader = create_sft_dataloader(val_jsonl, batch_size=config.micro_batch_size, block_size=config.block_size, shuffle=False)

        # Configure AdamW Optimizer
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))

    def get_lr(self, iter_num: int) -> float:
        # Linear warmup
        if iter_num < self.config.warmup_iters:
            return self.config.learning_rate * (iter_num + 1) / self.config.warmup_iters
        # Cosine decay to min_lr
        if iter_num > self.config.max_iters:
            return self.config.min_lr
        decay_ratio = (iter_num - self.config.warmup_iters) / (self.config.max_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x, y = x.to(self.device), y.to(self.device)
        logits, loss = self.model(x, y)
        loss.backward()
        return loss.item()

    def save_sft_checkpoint(self, filename: str = "best_sft_model.pt", extra_stats: Optional[Dict[str, Any]] = None):
        checkpoint_path = os.path.join(self.config.checkpoints_dir, filename)
        state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config.get_minigpt_config(),
            "sft_config": self.config,
            "extra_stats": extra_stats or {}
        }
        torch.save(state, checkpoint_path)
        print(f"Saved Phase 6 SFT checkpoint to {checkpoint_path}", flush=True)

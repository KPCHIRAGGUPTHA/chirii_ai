import os
from dataclasses import dataclass
from typing import Dict, Any
from model import MiniGPTConfig

@dataclass
class Phase6SFTConfig:
    # Model Architecture (Model C Model 6.61M parameters)
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 128
    vocab_size: int = 1024
    dropout: float = 0.1
    bias: bool = True

    # SFT Optimization Parameters
    # Conservative SFT learning rate (1e-4 vs pretraining 1e-3)
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_iters: int = 20
    max_iters: int = 1000
    grad_clip: float = 1.0

    # Batching & Hardware Settings
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 8 * 4 = 32 sequences
    random_seed: int = 42

    # Isolated Phase 6 Paths
    base_checkpoint_path: str = "checkpoints/phase5d/model_6_61m/best_model.pt"
    tokenizer_path: str = "tokenizers/phase5d/bpe_vocab_1024.json"
    checkpoints_dir: str = "checkpoints/phase6/model_c_sft"
    results_dir: str = "results/phase6"
    data_sft_dir: str = "phase6/data/sft"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def create_dirs(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def get_minigpt_config(self) -> MiniGPTConfig:
        return MiniGPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias
        )

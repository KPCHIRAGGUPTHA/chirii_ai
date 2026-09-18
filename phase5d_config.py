import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from model import MiniGPTConfig

@dataclass
class Phase5DConfig:
    # Model Architecture Parameters (Default to Model A 0.94M baseline)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.1
    bias: bool = True

    # Dataset Configuration (Verified FineWeb-Edu settings)
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    streaming: bool = True
    val_hash_ratio: float = 0.10

    # Data Cleaning Parameters
    min_doc_length: int = 50
    min_words: int = 10
    normalize_whitespace: bool = True

    # Tokenizer Parameters
    vocab_size: int = 1024

    # Training & Hardware Optimization
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 8 * 4 = 32 (4,096 tokens / step)
    max_iters: int = 2500                 # 2,500 * 4,096 = 10,240,000 target tokens
    eval_interval: int = 50
    eval_iters: int = 20
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 50
    grad_clip: float = 1.0
    random_seed: int = 42

    # Isolated Phase 5D Output Paths
    checkpoints_dir: str = "checkpoints/phase5d"
    results_dir: str = "results/phase5d"
    tokenizers_dir: str = "tokenizers/phase5d"
    data_dir: str = "data/phase5d"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def target_training_tokens(self) -> int:
        return self.max_iters * self.effective_batch_size * self.block_size

    def create_dirs(self):
        """Ensure all Phase 5D isolated directories exist."""
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tokenizers_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoints_dir, "model_2_89m"), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoints_dir, "model_6_61m"), exist_ok=True)

    @classmethod
    def create_model_a_config(cls) -> "Phase5DConfig":
        """Model A — Phase 5C Baseline (0.94M params)"""
        return cls(
            n_layer=4,
            n_head=4,
            n_embd=128,
            checkpoints_dir="checkpoints/phase5d/model_0_94m",
            results_dir="results/phase5d"
        )

    @classmethod
    def create_model_b_config(cls) -> "Phase5DConfig":
        """Model B — Phase 5D-A (2.89M params)"""
        return cls(
            n_layer=6,
            n_head=6,
            n_embd=192,
            checkpoints_dir="checkpoints/phase5d/model_2_89m",
            results_dir="results/phase5d"
        )

    @classmethod
    def create_model_c_config(cls) -> "Phase5DConfig":
        """Model C — Phase 5D-B (6.61M params)"""
        return cls(
            n_layer=8,
            n_head=8,
            n_embd=256,
            checkpoints_dir="checkpoints/phase5d/model_6_61m",
            results_dir="results/phase5d"
        )

    @staticmethod
    def get_model_a_minigpt_config() -> MiniGPTConfig:
        return MiniGPTConfig(vocab_size=1024, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1, bias=True)

    @staticmethod
    def get_model_b_minigpt_config() -> MiniGPTConfig:
        return MiniGPTConfig(vocab_size=1024, block_size=128, n_layer=6, n_head=6, n_embd=192, dropout=0.1, bias=True)

    @staticmethod
    def get_model_c_minigpt_config() -> MiniGPTConfig:
        return MiniGPTConfig(vocab_size=1024, block_size=128, n_layer=8, n_head=8, n_embd=256, dropout=0.1, bias=True)

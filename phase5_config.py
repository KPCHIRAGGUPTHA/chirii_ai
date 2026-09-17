import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Phase5Config:
    # Dataset Configuration (Verified FineWeb-Edu settings)
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"  # Smallest verified sample subset
    dataset_split: str = "train"
    streaming: bool = True
    val_hash_ratio: float = 0.10  # 10% deterministic hashing for validation

    # Data Cleaning Parameters
    min_doc_length: int = 50
    min_words: int = 10
    normalize_whitespace: bool = True

    # Tokenizer Parameters
    vocab_size: int = 1024  # Default vocabulary size (extensible to 256, 512, 1024, 2048, 4096)
    supported_vocab_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096])

    # Model Architecture (Unchanged from MiniGPT baseline)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.1
    bias: bool = True

    # Training & Hardware Optimization
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 8 * 4 = 32
    max_iters_smoke: int = 20              # Phase 5A budget
    max_iters_small: int = 600             # Phase 5B budget (~1M tokens)
    eval_interval: int = 50
    eval_iters: int = 20
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 50
    grad_clip: float = 1.0
    random_seed: int = 42

    # Isolated Phase 5 Output Paths
    checkpoints_dir: str = "checkpoints/phase5"
    results_dir: str = "results/phase5"
    tokenizers_dir: str = "tokenizers/phase5"
    data_dir: str = "data/phase5"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def create_dirs(self):
        """Ensure all Phase 5 isolated directories exist."""
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tokenizers_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "plots"), exist_ok=True)

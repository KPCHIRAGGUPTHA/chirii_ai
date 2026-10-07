import os
from dataclasses import dataclass, field
from typing import Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class RAGSFTConfig:
    """
    Configuration dataclass for Phase 7D-C RAG-SFT fine-tuning experiment.
    """
    starting_ckpt_path: str = os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt")
    output_dir: str = os.path.join(REPO_ROOT, "checkpoints", "phase7", "model_c_rag_sft")
    tokenizer_path: str = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")

    data_train_path: str = os.path.join(REPO_ROOT, "phase7", "data", "sft", "rag_sft_train.jsonl")
    data_val_path: str = os.path.join(REPO_ROOT, "phase7", "data", "sft", "rag_sft_val.jsonl")
    data_test_path: str = os.path.join(REPO_ROOT, "phase7", "data", "sft", "rag_sft_test.jsonl")

    # Hyperparameters matching Phase 7D-C specification
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_steps: int = 100
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "starting_ckpt_path": self.starting_ckpt_path,
            "output_dir": self.output_dir,
            "tokenizer_path": self.tokenizer_path,
            "data_train_path": self.data_train_path,
            "data_val_path": self.data_val_path,
            "data_test_path": self.data_test_path,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "device": self.device,
            "seed": self.seed
        }

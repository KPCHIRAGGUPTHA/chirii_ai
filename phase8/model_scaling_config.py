"""
Phase 8 Model Scaling Configuration Module
Defines model architectures for controlled scaling experiments (Models A, B, C, D, E).
"""

from dataclasses import dataclass
from typing import Dict
from model import MiniGPTConfig, MiniGPT

EXPECTED_PARAM_COUNTS: Dict[str, int] = {
    "Model A": 940800,
    "Model B": 2890752,
    "Model C": 6613504,
    "Model D": 15164800,
    "Model E": 29488256
}

@dataclass
class ModelScalingConfig:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 128
    vocab_size: int = 1024
    dropout: float = 0.1
    bias: bool = True

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        return self.n_embd // self.n_head

    def to_minigpt_config(self) -> MiniGPTConfig:
        return MiniGPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias
        )

    def instantiate_model(self) -> MiniGPT:
        cfg = self.to_minigpt_config()
        return MiniGPT(cfg)

    def calculate_expected_params(self) -> int:
        """
        Analytical formula for MiniGPT trainable parameters with tied weights:
        Params = V*E + B*E + L*(12*E^2 + 13*E) + 2*E
        """
        V = self.vocab_size
        B = self.block_size
        E = self.n_embd
        L = self.n_layer
        return (V * E) + (B * E) + L * (12 * (E ** 2) + 13 * E) + (2 * E)

def get_model_a_config() -> ModelScalingConfig:
    """Model A — 0.94M baseline"""
    return ModelScalingConfig(name="Model A", n_layer=4, n_head=4, n_embd=128)

def get_model_b_config() -> ModelScalingConfig:
    """Model B — 2.89M baseline"""
    return ModelScalingConfig(name="Model B", n_layer=6, n_head=6, n_embd=192)

def get_model_c_config() -> ModelScalingConfig:
    """Model C — 6.61M current baseline"""
    return ModelScalingConfig(name="Model C", n_layer=8, n_head=8, n_embd=256)

def get_model_d_config() -> ModelScalingConfig:
    """Model D — 15.16M scaling candidate 1"""
    return ModelScalingConfig(name="Model D", n_layer=12, n_head=10, n_embd=320)

def get_model_e_config() -> ModelScalingConfig:
    """Model E — 29.49M scaling candidate 2"""
    return ModelScalingConfig(name="Model E", n_layer=12, n_head=14, n_embd=448)

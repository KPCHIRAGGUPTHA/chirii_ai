import pytest
import torch
from model import MiniGPTConfig, CausalSelfAttention

def test_causal_mask():
    config = MiniGPTConfig(block_size=32)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 32, config.n_embd)
    output = attn(x)
    
    # Causal mask should be lower-triangular (1 on/below diagonal, 0 above diagonal)
    bias_matrix = attn.bias[0, 0, :32, :32]
    expected_tril = torch.tril(torch.ones(32, 32))
    assert torch.equal(bias_matrix, expected_tril), "Causal mask is not correctly applied"
    assert output.shape == (2, 32, config.n_embd), "Output tensor shape mismatch"
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Attention output contains NaN or Inf"

def test_attention_invalid_dim():
    config = MiniGPTConfig(n_embd=33, n_head=4)
    with pytest.raises(AssertionError, match="n_embd must be divisible by n_head"):
        CausalSelfAttention(config)
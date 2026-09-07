import pytest
import torch
from model import MiniGPT, MiniGPTConfig

def test_parameter_count():
    config = MiniGPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128, bias=True)
    model = MiniGPT(config)
    
    # Exact calculation for MiniGPT architecture:
    # wte: vocab_size * n_embd
    # wpe: block_size * n_embd
    # Per layer (n_layer layers):
    #   ln_1: 2 * n_embd
    #   c_attn: (n_embd * 3 * n_embd) + 3 * n_embd
    #   c_proj: (n_embd * n_embd) + n_embd
    #   ln_2: 2 * n_embd
    #   c_fc: (n_embd * 4 * n_embd) + 4 * n_embd
    #   c_proj: (4 * n_embd * n_embd) + n_embd
    # ln_f: 2 * n_embd
    # lm_head tied to wte (0 extra params)
    wte = config.vocab_size * config.n_embd
    wpe = config.block_size * config.n_embd
    block_params = (
        (config.n_embd * 2) +
        (config.n_embd * 3 * config.n_embd + 3 * config.n_embd) +
        (config.n_embd * config.n_embd + config.n_embd) +
        (config.n_embd * 2) +
        (config.n_embd * 4 * config.n_embd + 4 * config.n_embd) +
        (4 * config.n_embd * config.n_embd + config.n_embd)
    )
    ln_f = config.n_embd * 2
    expected_params = wte + wpe + (config.n_layer * block_params) + ln_f
    assert model.get_num_params() == expected_params, f"Expected {expected_params:,} params, got {model.get_num_params():,}"

def test_forward_pass_with_targets():
    config = MiniGPTConfig(vocab_size=65, block_size=32)
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, config.vocab_size), "Output logits shape mismatch"
    assert loss is not None, "Loss should not be None when targets are provided"
    assert loss.dim() == 0, "Loss should be a scalar tensor"

def test_forward_pass_without_targets():
    config = MiniGPTConfig(vocab_size=65, block_size=32)
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 1, config.vocab_size), "Inference logits shape should be (B, 1, vocab_size)"
    assert loss is None, "Loss should be None during inference without targets"

def test_target_shape_mismatch():
    config = MiniGPTConfig(vocab_size=65, block_size=32)
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 10))
    with pytest.raises(AssertionError, match="Targets shape"):
        model(x, targets)

def test_sequence_length_exceeded():
    config = MiniGPTConfig(vocab_size=65, block_size=16)
    model = MiniGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 20))
    with pytest.raises(AssertionError, match="Cannot forward sequence length"):
        model(x)

def test_generate_method():
    config = MiniGPTConfig(vocab_size=65, block_size=32)
    model = MiniGPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (1, 15), f"Expected generated shape (1, 15), got {out.shape}"
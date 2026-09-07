import pytest
import torch
from tokenizer import CharTokenizer
from model import MiniGPT, MiniGPTConfig, CausalSelfAttention
from dataset import get_batch

def test_regression_causal_mask_triangular_structure():
    """Regression: Causal mask must be lower-triangular (1 on/below diagonal, 0 above)."""
    config = MiniGPTConfig(block_size=16)
    attn = CausalSelfAttention(config)
    mask = attn.bias[0, 0, :16, :16]
    expected = torch.tril(torch.ones(16, 16))
    assert torch.equal(mask, expected), "Causal mask failed lower triangular structure check"

def test_regression_batch_target_length():
    """Regression: x and y tensors returned by get_batch must have identical block_size shapes."""
    data = torch.tensor(list(range(50)), dtype=torch.long)
    block_size = 8
    batch_size = 2
    x, y = get_batch(data, block_size=block_size, batch_size=batch_size)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

def test_regression_short_dataset_raise_value_error():
    """Regression: get_batch must raise ValueError with clear message when data_tensor length <= block_size."""
    short_data = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="must be greater than block_size"):
        get_batch(short_data, block_size=5, batch_size=1)

def test_regression_vocab_character_decoding():
    """Regression: Verify character token membership against tokenizer stoi dict."""
    tokenizer = CharTokenizer()
    sample_chars = "Hello, world!"
    for ch in sample_chars:
        assert ch in tokenizer.stoi, f"Character '{ch}' not in default vocabulary"

def test_regression_parameter_count_formula():
    """Regression: Verify parameter count matches exact module parameters and weight tying."""
    config = MiniGPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=128, bias=True)
    model = MiniGPT(config)
    
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
    expected = wte + wpe + (config.n_layer * block_params) + ln_f
    assert model.get_num_params() == expected

def test_regression_unicode_out_of_vocab_character():
    """Regression: Non-vocabulary characters must map to unk_id."""
    tokenizer = CharTokenizer()
    encoded = tokenizer.encode("\u4f60\u597d")
    assert all(tok == tokenizer.unk_id for tok in encoded)

def test_regression_overfitting_vocab_size_match():
    """Regression: Config vocab_size must match tokenizer vocab_size to prevent embedding IndexError."""
    text = "Short dataset text."
    tokenizer = CharTokenizer.from_text(text)
    config = MiniGPTConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = MiniGPT(config)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)[:8]
    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)
    logits, loss = model(x, y)
    assert logits.shape == (1, 7, tokenizer.vocab_size)
    assert loss is not None

def test_regression_target_shape_validation():
    """Regression: MiniGPT.forward must validate input and target shape equality."""
    config = MiniGPTConfig(vocab_size=65, block_size=32)
    model = MiniGPT(config)
    x = torch.randint(0, 65, (2, 10))
    mismatched_target = torch.randint(0, 65, (2, 5))
    with pytest.raises(AssertionError, match="Targets shape"):
        model(x, mismatched_target)

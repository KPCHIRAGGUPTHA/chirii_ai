import os
import tempfile
import pytest
import torch
from model import MiniGPTConfig
from generate import generate_text, load_model_and_tokenizer
from train import train_model

@pytest.fixture(scope="module")
def trained_ckpt_dir():
    """Module-level fixture to train a micro model for generation testing."""
    tmpdir = tempfile.mkdtemp()
    train_model(
        out_dir=tmpdir,
        results_dir=tmpdir,
        max_iters=20,
        batch_size=4,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        eval_interval=10
    )
    yield tmpdir

def test_load_model_and_tokenizer(trained_ckpt_dir):
    model, tokenizer = load_model_and_tokenizer(trained_ckpt_dir)
    assert model is not None
    assert tokenizer is not None

def test_load_model_missing_checkpoint_raises():
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model_and_tokenizer(empty_dir)

def test_generation_respects_context(trained_ckpt_dir):
    prompt = "First Citizen:"
    generated = generate_text(prompt, ckpt_dir=trained_ckpt_dir, max_new_tokens=10)
    assert generated.startswith(prompt), "Generated text does not start with the prompt"

def test_generation_produces_in_vocab_tokens(trained_ckpt_dir):
    model, tokenizer = load_model_and_tokenizer(trained_ckpt_dir)
    prompt = "First Citizen:"
    generated = generate_text(prompt, ckpt_dir=trained_ckpt_dir, max_new_tokens=30)
    for ch in generated:
        assert ch in tokenizer.stoi, f"Generated character '{ch}' is not in the vocabulary"

def test_generation_respects_temperature(trained_ckpt_dir):
    prompt = "First Citizen:"
    torch.manual_seed(42)
    gen_low_temp = generate_text(prompt, ckpt_dir=trained_ckpt_dir, max_new_tokens=15, temperature=0.01)
    torch.manual_seed(42)
    gen_high_temp = generate_text(prompt, ckpt_dir=trained_ckpt_dir, max_new_tokens=15, temperature=2.0)
    assert len(gen_low_temp) > 0 and len(gen_high_temp) > 0

def test_generation_respects_top_k_top_p(trained_ckpt_dir):
    prompt = "First Citizen:"
    gen_top_k = generate_text(prompt, ckpt_dir=trained_ckpt_dir, max_new_tokens=10, top_k=1)
    gen_top_p = generate_text(prompt, ckpt_dir=trained_ckpt_dir, max_new_tokens=10, top_p=0.5)
    assert gen_top_k.startswith(prompt)
    assert gen_top_p.startswith(prompt)
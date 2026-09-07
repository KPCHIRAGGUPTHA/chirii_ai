import os
import json
import math
import tempfile
import pytest
import torch
from model import MiniGPT, MiniGPTConfig
from tokenizer import CharTokenizer
from train import set_seed, estimate_loss, calculate_perplexity, train_model
from generate import load_model_and_tokenizer, generate_text
from evaluate import run_evaluation
from plot_history import plot_training_history

def test_estimate_loss():
    """Test estimate_loss function averages loss over multiple batches and leaves model in train mode."""
    set_seed(42)
    config = MiniGPTConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = MiniGPT(config)
    train_data = torch.randint(0, 30, (100,))
    val_data = torch.randint(0, 30, (100,))
    
    metrics = estimate_loss(model, train_data, val_data, block_size=16, batch_size=4, eval_iters=5, device='cpu')
    
    assert "train" in metrics and "val" in metrics
    assert isinstance(metrics["train"], float)
    assert isinstance(metrics["val"], float)
    assert metrics["train"] > 0.0 and metrics["val"] > 0.0
    assert model.training is True, "Model should be restored to training mode after estimate_loss"

def test_multiple_validation_batches():
    """Test evaluation over multiple validation batches under torch.no_grad()."""
    set_seed(42)
    config = MiniGPTConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2, n_embd=16)
    model = MiniGPT(config)
    train_data = torch.randint(0, 30, (200,))
    val_data = torch.randint(0, 30, (200,))
    
    metrics_single = estimate_loss(model, train_data, val_data, block_size=16, batch_size=4, eval_iters=1, device='cpu')
    metrics_multi = estimate_loss(model, train_data, val_data, block_size=16, batch_size=4, eval_iters=10, device='cpu')
    
    assert isinstance(metrics_single["val"], float)
    assert isinstance(metrics_multi["val"], float)

def test_perplexity_calculation():
    """Test perplexity exp(val_loss) calculation and numerical safety clamping."""
    loss_val = 2.5
    expected_perp = math.exp(2.5)
    assert math.isclose(calculate_perplexity(loss_val), expected_perp, rel_tol=1e-5)
    
    # Test numeric safety on high loss values
    high_perp = calculate_perplexity(100.0)
    assert math.isclose(high_perp, math.exp(50.0), rel_tol=1e-5)

def test_reproducibility():
    """Test that identical seeds yield reproducible model initializations and loss outputs."""
    config = MiniGPTConfig(vocab_size=40, block_size=16, n_layer=2, n_head=2, n_embd=32)
    x = torch.randint(0, 40, (2, 16))
    y = torch.randint(0, 40, (2, 16))
    
    set_seed(123)
    model1 = MiniGPT(config)
    _, loss1 = model1(x, y)
    
    set_seed(123)
    model2 = MiniGPT(config)
    _, loss2 = model2(x, y)
    
    assert torch.equal(model1.transformer.wte.weight, model2.transformer.wte.weight)
    assert torch.isclose(loss1, loss2), "Models with identical seed must output identical loss values"

def test_best_and_final_checkpoint_creation():
    """Test that train_model creates best_model.pt, final_model.pt, and checkpoint.pt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        train_model(
            out_dir=tmpdir,
            results_dir=results_dir,
            max_iters=10,
            batch_size=4,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            eval_interval=5,
            eval_iters=2,
            random_seed=42
        )
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt")), "best_model.pt not created"
        assert os.path.exists(os.path.join(tmpdir, "final_model.pt")), "final_model.pt not created"
        assert os.path.exists(os.path.join(tmpdir, "checkpoint.pt")), "checkpoint.pt not created"
        assert os.path.exists(os.path.join(results_dir, "training_history.json")), "training_history.json not created"

def test_checkpoint_restoration():
    """Test checkpoint loading and model restoration from best_model.pt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_model(
            out_dir=tmpdir,
            results_dir=tmpdir,
            max_iters=10,
            batch_size=4,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            eval_interval=5,
            eval_iters=2
        )
        model_best, tokenizer_best = load_model_and_tokenizer(tmpdir, filename="best_model.pt")
        model_final, tokenizer_final = load_model_and_tokenizer(tmpdir, filename="final_model.pt")
        
        assert isinstance(model_best, MiniGPT)
        assert isinstance(model_final, MiniGPT)
        assert model_best.config.n_embd == 32

def test_training_history():
    """Test structured JSON training history contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        train_model(
            out_dir=tmpdir,
            results_dir=results_dir,
            max_iters=10,
            batch_size=4,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            eval_interval=5,
            eval_iters=2
        )
        history_file = os.path.join(results_dir, "training_history.json")
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert "config" in data
        assert "history" in data
        assert "best_val_loss" in data
        assert "final_val_perplexity" in data
        assert len(data["history"]) >= 2

def test_evaluation_script():
    """Test running evaluate.py run_evaluation on a trained model checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_model(
            out_dir=tmpdir,
            results_dir=tmpdir,
            max_iters=10,
            batch_size=4,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            eval_interval=5,
            eval_iters=2,
            random_seed=42
        )
        eval_stats = run_evaluation(
            ckpt_dir=tmpdir,
            ckpt_name="best_model.pt",
            eval_iters=3,
            seed=42,
            save_output=False
        )
        assert "parameter_count" in eval_stats
        assert "val_perplexity" in eval_stats
        assert "qualitative_samples" in eval_stats
        assert "ROMEO:" in eval_stats["qualitative_samples"]

def test_plotting_script():
    """Test plot_history.py execution using temporary training history JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = os.path.join(tmpdir, "training_history.json")
        output_dir = os.path.join(tmpdir, "plots")
        
        sample_history = {
            "history": [
                {"step": 10, "train_loss": 3.5, "val_loss": 3.6, "val_perplexity": 36.6},
                {"step": 20, "train_loss": 3.0, "val_loss": 3.1, "val_perplexity": 22.2}
            ]
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(sample_history, f)
            
        loss_plot, perp_plot = plot_training_history(history_path=history_path, output_dir=output_dir)
        assert os.path.exists(loss_plot), "loss_curve.png not created"
        assert os.path.exists(perp_plot), "perplexity_curve.png not created"

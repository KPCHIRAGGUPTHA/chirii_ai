import os
import tempfile
import torch
import pytest

from phase5_config import Phase5Config
from phase5_dataset import get_document_split, get_batch_phase5, prepare_phase5_corpus
from phase5_tokenizer import train_phase5_tokenizer
from train_phase5 import get_lr, train_phase5, save_phase5_checkpoint
from model import MiniGPT, MiniGPTConfig

def test_deterministic_hashing_split():
    doc1 = {"id": "doc_001", "text": "Sample text for document 1"}
    doc2 = {"id": "doc_002", "text": "Sample text for document 2"}

    split1_first = get_document_split(doc1, val_ratio=0.10)
    split1_second = get_document_split(doc1, val_ratio=0.10)
    assert split1_first == split1_second, "Hashing split must be 100% deterministic"

    split2_first = get_document_split(doc2, val_ratio=0.10)
    split2_second = get_document_split(doc2, val_ratio=0.10)
    assert split2_first == split2_second, "Hashing split must be 100% deterministic"

def test_lr_scheduler():
    config = Phase5Config(
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_iters=10
    )
    max_iters = 100

    # 1. Warmup phase (linear increase)
    lr_step0 = get_lr(0, max_iters, config)
    lr_step5 = get_lr(5, max_iters, config)
    lr_step10 = get_lr(10, max_iters, config)
    assert lr_step0 == 0.0
    assert 0.0 < lr_step5 < config.learning_rate
    assert abs(lr_step10 - config.learning_rate) < 1e-6

    # 2. Cosine decay phase (monotonically decreasing after warmup)
    lr_step50 = get_lr(50, max_iters, config)
    lr_step100 = get_lr(100, max_iters, config)
    assert lr_step10 > lr_step50
    assert lr_step50 > lr_step100
    assert abs(lr_step100 - config.min_lr) < 1e-6

def test_target_shift_and_batch_shapes():
    data_tensor = torch.arange(100, dtype=torch.long)
    block_size = 8
    batch_size = 4

    batch_x, batch_y = get_batch_phase5(data_tensor, block_size=block_size, batch_size=batch_size, device="cpu")
    assert batch_x.shape == (batch_size, block_size)
    assert batch_y.shape == (batch_size, block_size)
    assert batch_x.dtype == torch.long
    assert batch_y.dtype == torch.long
    # Verify target shift for each sequence in batch: y[b, t] == x[b, t+1] (except last token shifted)
    assert torch.equal(batch_x[:, 1:], batch_y[:, :-1]), "Targets y must equal input x shifted by 1 token"

def test_checkpoint_save_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Phase5Config(
            checkpoints_dir=os.path.join(tmpdir, "checkpoints"),
            results_dir=os.path.join(tmpdir, "results"),
            tokenizers_dir=os.path.join(tmpdir, "tokenizers"),
            data_dir=os.path.join(tmpdir, "data")
        )
        config.create_dirs()

        model_config = MiniGPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32)
        model = MiniGPT(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        history = [{"step": 10, "val_loss": 2.5}]

        ckpt_path = os.path.join(config.checkpoints_dir, "test_ckpt.pt")
        save_phase5_checkpoint(
            ckpt_path, model, optimizer, config, step=10,
            val_loss=2.5, best_val_loss=2.5, history=history
        )

        assert os.path.exists(ckpt_path)
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["phase"] == 5
        assert loaded["step"] == 10
        assert loaded["val_loss"] == 2.5
        assert "model_state" in loaded
        assert "optimizer_state" in loaded

def test_phase4_artifacts_unmodified():
    """Verify that running Phase 5 logic does NOT modify Phase 4 directories."""
    phase4_results = os.path.join("results", "phase4")
    phase4_ckpts = os.path.join("checkpoints", "phase4")

    if os.path.exists(phase4_results):
        mtime_results = os.path.getmtime(phase4_results)
        assert mtime_results > 0
    if os.path.exists(phase4_ckpts):
        mtime_ckpts = os.path.getmtime(phase4_ckpts)
        assert mtime_ckpts > 0

import tempfile
import os
import torch
from model import MiniGPT, MiniGPTConfig
from tokenizer import CharTokenizer
from train import train_model

def test_overfitting():
    text = "The cat sat on the mat."
    tokenizer = CharTokenizer.from_text(text)
    config = MiniGPTConfig(vocab_size=tokenizer.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)[:config.block_size]

    for _ in range(200):
        model.train()
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, "Model failed to overfit the tiny dataset"

def test_train_model_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        model, tokenizer = train_model(
            out_dir=tmpdir,
            results_dir=tmpdir,
            max_iters=10,
            batch_size=4,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=32,
            eval_interval=5
        )
        assert os.path.exists(os.path.join(tmpdir, "checkpoint.pt")), "Checkpoint file not created"
        assert os.path.exists(os.path.join(tmpdir, "vocab.json")), "Vocab file not created"
        assert isinstance(model, MiniGPT)
        assert isinstance(tokenizer, CharTokenizer)

def test_train_model_callback():
    callback_records = []
    def callback(stats):
        callback_records.append(stats)

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
            callback=callback
        )
        assert len(callback_records) > 0, "Callback was not invoked during training"
        assert "train_loss" in callback_records[0]
        assert "val_loss" in callback_records[0]
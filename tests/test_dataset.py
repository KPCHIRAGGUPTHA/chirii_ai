import pytest
import torch
from dataset import get_or_download_text, get_batch
from tokenizer import CharTokenizer

def test_dataset_download():
    text = get_or_download_text()
    assert len(text) > 0, "Dataset is empty"

def test_data_batching():
    text = get_or_download_text()
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    block_size = 32
    batch_size = 2
    x, y = get_batch(data, block_size=block_size, batch_size=batch_size)
    assert x.shape == (batch_size, block_size), "Batch x shape mismatch"
    assert y.shape == (batch_size, block_size), "Batch y shape mismatch"
    assert x.shape == y.shape, "x and y shapes must be equal"

def test_data_batching_target_shift():
    data = torch.tensor(list(range(100)), dtype=torch.long)
    block_size = 10
    batch_size = 4
    x, y = get_batch(data, block_size=block_size, batch_size=batch_size)
    # Every token in y should equal token in x + 1 for sequential data
    assert torch.equal(y, x + 1), "Target y must be input sequence x shifted by 1 position"

def test_short_data_validation():
    short_data = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="must be greater than block_size"):
        get_batch(short_data, block_size=5, batch_size=1)
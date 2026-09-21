import os
import sys
import json
import hashlib
import torch
import torch.nn.functional as F
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT
from bpe_tokenizer import BPETokenizer
from phase6.training.sft_config import Phase6SFTConfig
from phase6.training.sft_dataset import SFTDataset, create_sft_dataloader
from phase6.training.sft_trainer import calculate_sft_loss
from phase6.preparation.prepare_sft_dataset import format_alpaca_prompt

def get_file_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_sft_dataset_loading():
    sft_path = os.path.join(REPO_ROOT, "phase6", "data", "sft", "train_sft.jsonl")
    assert os.path.exists(sft_path), "SFT train dataset file missing"
    dataset = SFTDataset(sft_path, block_size=128)
    assert len(dataset) > 0, "SFT dataset is empty"
    x, y = dataset[0]
    assert x.shape == (128,)
    assert y.shape == (128,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

def test_prompt_masking_hand_checkable():
    """
    Hand-checkable test example from user prompt:
    tokens = [P1, P2, P3, R1, R2, R3] (P1, P2, P3 = 10, 11, 12; R1, R2, R3 = 20, 21, 22)
    prompt_token_len = 3
    Shifted inputs  = [P1, P2, P3, R1, R2] = [10, 11, 12, 20, 21]
    Shifted targets = [P2, P3, R1, R2, R3] = [11, 12, 20, 21, 22]
    Expected labels = [-100, -100, R1, R2, R3] = [-100, -100, 20, 21, 22]
    """
    mock_rec = {
        "tokens": [10, 11, 12, 20, 21, 22],
        "prompt_token_len": 3
    }

    # Simulate SFTDataset __getitem__ logic
    effective_tokens = mock_rec["tokens"][: 5 + 1] # block_size=5
    input_seq = list(effective_tokens[:-1])
    target_seq = list(effective_tokens[1:])

    prompt_token_len = mock_rec["prompt_token_len"]
    labels = []
    for i, target_tok in enumerate(target_seq):
        target_pos = i + 1
        if target_pos < prompt_token_len:
            labels.append(-100)
        else:
            labels.append(target_tok)

    assert input_seq == [10, 11, 12, 20, 21]
    assert target_seq == [11, 12, 20, 21, 22]
    assert labels == [-100, -100, 20, 21, 22]
    assert labels[2] == 20, "Target corresponding to R1 must be UNMASKED"

def test_response_labels_unmasked():
    mock_rec = {
        "tokens": [100, 101, 102, 200, 201],
        "prompt_token_len": 3
    }
    dataset = SFTDataset("", block_size=128)
    dataset.records = [mock_rec]
    x, y = dataset[0]

    # Position 0 (target is 101, prompt) -> -100
    # Position 1 (target is 102, prompt) -> -100
    # Position 2 (target is 200, response R1) -> 200 (UNMASKED)
    assert y[0].item() == -100
    assert y[1].item() == -100
    assert y[2].item() == 200, f"Expected 200, got {y[2].item()}"

def test_autoregressive_shift():
    tokens = [5, 6, 7, 8, 9]
    mock_rec = {"tokens": tokens, "prompt_token_len": 2}
    dataset = SFTDataset("", block_size=4)
    dataset.records = [mock_rec]
    x, y = dataset[0]

    # Input is tokens[0:4] = [5, 6, 7, 8]
    # Shifted target is tokens[1:5] = [6, 7, 8, 9]
    assert x.tolist() == [5, 6, 7, 8]
    # Target 6 (pos 1) < 2 -> -100; Targets 7, 8, 9 >= 2 -> retain
    assert y.tolist() == [-100, 7, 8, 9]

def test_ignore_index_loss_invariance():
    """
    Verify that modifying prompt logits (masked with -100) does NOT alter loss,
    while modifying response logits DOES alter loss.
    """
    logits = torch.randn(2, 10, 1024)
    targets = torch.randint(0, 1024, (2, 10))
    targets[:, :4] = -100  # Mask prompt positions 0..3

    loss_orig = calculate_sft_loss(logits, targets)

    # Modify prompt logits
    logits_prompt_mod = logits.clone()
    logits_prompt_mod[:, :4, :] += 50.0
    loss_prompt_mod = calculate_sft_loss(logits_prompt_mod, targets)

    assert torch.isclose(loss_orig, loss_prompt_mod), "Prompt logit modification altered response-only loss!"

    # Modify target class logit in response region
    logits_resp_mod = logits.clone()
    logits_resp_mod[:, 5, targets[0, 5]] += 5.0
    loss_resp_mod = calculate_sft_loss(logits_resp_mod, targets)

    assert not torch.isclose(loss_orig, loss_resp_mod), "Response logit modification failed to alter loss!"

def test_batch_construction_and_padding():
    sft_path = os.path.join(REPO_ROOT, "phase6", "data", "sft", "train_sft.jsonl")
    loader = create_sft_dataloader(sft_path, batch_size=4, block_size=128, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (4, 128)
    assert y.shape == (4, 128)
    # Check that padded positions in y contain -100
    for row in y:
        assert (-100 in row), "Padding labels must contain -100"

def test_model_checkpoint_loading():
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
    assert os.path.exists(ckpt_path), "Model C checkpoint missing"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = Phase6SFTConfig()
    model = MiniGPT(config.get_minigpt_config())
    state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    assert model.get_num_params() == 6613504

def test_parameter_count():
    config = Phase6SFTConfig()
    model = MiniGPT(config.get_minigpt_config())
    assert model.get_num_params() == 6613504

def test_tokenizer_compatibility():
    vocab_path = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
    tokenizer = BPETokenizer.load(vocab_path)
    assert tokenizer.vocab_size == 1024

    prefix, full = format_alpaca_prompt("Test inst", "Test inp", "Test out")
    enc = tokenizer.encode(full)
    dec = tokenizer.decode(enc)
    assert dec == full, "Tokenizer roundtrip failed"

def test_checkpoint_immutability():
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase5d", "model_6_61m", "best_model.pt")
    hash1 = get_file_sha256(ckpt_path)
    
    # Instantiate model and perform forward pass
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = Phase6SFTConfig()
    model = MiniGPT(config.get_minigpt_config())
    state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    
    dummy_x = torch.randint(0, 1024, (2, 128))
    dummy_y = torch.randint(0, 1024, (2, 128))
    dummy_y[:, :64] = -100
    
    logits, loss = model(dummy_x, dummy_y)
    loss.backward()

    hash2 = get_file_sha256(ckpt_path)
    assert hash1 == hash2, "Checkpoint file was mutated during test!"

def test_tiny_backward_pass():
    config = Phase6SFTConfig()
    model = MiniGPT(config.get_minigpt_config())
    model.train()

    dummy_x = torch.randint(0, 1024, (2, 128))
    dummy_y = torch.randint(0, 1024, (2, 128))
    dummy_y[:, :64] = -100

    logits, loss = model(dummy_x, dummy_y)
    assert not torch.isnan(loss)
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Parameter missing gradient after backward pass"

def test_actual_sft_records_boundary():
    sft_path = os.path.join(REPO_ROOT, "phase6", "data", "sft", "train_sft.jsonl")
    vocab_path = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
    tokenizer = BPETokenizer.load(vocab_path)

    with open(sft_path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    inst = rec["instruction"]
    inp = rec["input"]
    out = rec["output"]
    prompt_prefix, full_text = format_alpaca_prompt(inst, inp, out)

    prefix_tokens = tokenizer.encode(prompt_prefix)
    full_tokens = tokenizer.encode(full_text)

    prompt_len = rec["prompt_token_len"]
    assert prompt_len == len(prefix_tokens)
    assert full_tokens[:prompt_len] == prefix_tokens, "Prompt prefix tokens mismatch in actual SFT record"

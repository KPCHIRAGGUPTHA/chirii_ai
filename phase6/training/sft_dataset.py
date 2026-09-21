import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT) Dataset with target response loss masking.
    
    Autoregressive Shifting:
        input_ids[i]  = tokens[i]
        labels[i]     = tokens[i+1] if tokens[i+1] is a response token else -100
        
    Prompt Masking:
        Target tokens corresponding to prompt positions (i+1 < prompt_token_len) are set to -100.
        Target tokens corresponding to response positions (i+1 >= prompt_token_len) retain target IDs.
        
    Padding:
        Padded positions in labels are set to -100 so they never contribute to PyTorch CrossEntropyLoss.
    """
    def __init__(self, jsonl_path: str, block_size: int = 128, eos_token_id: int = 0):
        super().__init__()
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.records: List[Dict[str, Any]] = []

        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        tokens = rec["tokens"]
        prompt_token_len = rec.get("prompt_token_len", 0)

        # Cap token sequence to block_size + 1 for autoregressive shifting
        effective_tokens = tokens[: self.block_size + 1]
        n = len(effective_tokens)

        if n <= 1:
            # Fallback for empty/single token
            input_seq = [self.eos_token_id] * self.block_size
            labels = [-100] * self.block_size
            return torch.tensor(input_seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        # Autoregressive shift: input_ids = tokens[0:N-1], target_seq = tokens[1:N]
        input_seq = list(effective_tokens[:-1])
        target_seq = list(effective_tokens[1:])

        # Apply prompt loss masking
        labels = []
        for i, target_tok in enumerate(target_seq):
            target_pos = i + 1  # Original index in effective_tokens
            if target_pos < prompt_token_len:
                labels.append(-100)
            else:
                labels.append(target_tok)

        # Pad sequences to exact block_size
        curr_len = len(input_seq)
        if curr_len < self.block_size:
            pad_len = self.block_size - curr_len
            input_seq.extend([self.eos_token_id] * pad_len)
            labels.extend([-100] * pad_len)

        return (
            torch.tensor(input_seq[: self.block_size], dtype=torch.long),
            torch.tensor(labels[: self.block_size], dtype=torch.long)
        )

def create_sft_dataloader(jsonl_path: str, batch_size: int = 8, block_size: int = 128, shuffle: bool = True) -> torch.utils.data.DataLoader:
    dataset = SFTDataset(jsonl_path, block_size=block_size)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

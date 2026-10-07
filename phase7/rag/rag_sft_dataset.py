import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Optional
from bpe_tokenizer import BPETokenizer

class RAGSFTDataset(Dataset):
    """
    PyTorch Dataset for Phase 7D-C RAG-SFT fine-tuning.
    Applies answer-only target masking with -100 for all prompt tokens.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: BPETokenizer,
        max_block_size: int = 128,
        pad_token_id: int = 0
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_block_size = max_block_size
        self.pad_token_id = pad_token_id

        self.records: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        context = rec["context"].strip()
        question = rec["question"].strip()
        answer = rec["answer"].strip()

        prompt_prefix = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
        full_text = f"{prompt_prefix}{answer}"

        prefix_tokens = self.tokenizer.encode(prompt_prefix)
        full_tokens = self.tokenizer.encode(full_text)

        # Truncate full_tokens if exceeding block_size + 1
        if len(full_tokens) > self.max_block_size:
            full_tokens = full_tokens[:self.max_block_size]

        prompt_len = len(prefix_tokens)

        # Build causal input_ids and targets
        # input_ids: full_tokens[:-1]
        # targets:   full_tokens[1:]
        input_ids = full_tokens[:-1]
        targets = full_tokens[1:]

        # Create target mask: prompt tokens up to (prompt_len - 1) masked to -100
        masked_targets = []
        for i, tid in enumerate(targets):
            if i < prompt_len - 1:
                masked_targets.append(-100)
            else:
                masked_targets.append(tid)

        # Pad to max_block_size - 1 (or leave unpadded if collate_fn handles it)
        seq_len = len(input_ids)
        pad_len = (self.max_block_size - 1) - seq_len

        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            masked_targets = masked_targets + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(masked_targets, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "record_id": torch.tensor(rec["id"], dtype=torch.long)
        }

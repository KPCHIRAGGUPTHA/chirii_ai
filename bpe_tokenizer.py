import json
import os
from typing import List, Dict, Tuple, Optional

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) subword tokenizer implemented from scratch.
    Learns frequent subword merges from training text deterministically.
    """
    def __init__(self, vocab_size: int = 256):
        self.target_vocab_size = vocab_size
        self.base_chars: List[str] = []
        self.merges: List[Tuple[str, str]] = []  # Ordered list of (token_a, token_b) merges
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.unk_token = '<unk>'
        self.unk_id: int = 0

    @classmethod
    def train(cls, text: str, target_vocab_size: int = 256, max_train_chars: int = 50000) -> "BPETokenizer":
        """
        Train a BPE tokenizer directly on input text corpus until target_vocab_size is reached.
        Uses deterministic lexicographical tie-breaking for equal pair frequencies.
        """
        tokenizer = cls(vocab_size=target_vocab_size)
        
        # 1. Base vocabulary from unique characters in corpus
        unique_chars = sorted(list(set(text)))
        tokenizer.base_chars = list(unique_chars)
        
        vocab = list(unique_chars)
        if tokenizer.unk_token not in vocab:
            vocab.append(tokenizer.unk_token)
            
        tokenizer.stoi = {token: i for i, token in enumerate(vocab)}
        tokenizer.itos = {i: token for i, token in enumerate(vocab)}
        tokenizer.vocab_size = len(vocab)
        tokenizer.unk_id = tokenizer.stoi[tokenizer.unk_token]

        # 2. Represent text as a sequence of character tokens (capped at max_train_chars for performance)
        sample_text = text[:max_train_chars] if max_train_chars and len(text) > max_train_chars else text
        tokens = list(sample_text)
        
        # 3. Iteratively learn merges
        num_merges_needed = target_vocab_size - tokenizer.vocab_size
        for _ in range(num_merges_needed):
            if len(tokens) < 2:
                break
                
            # Count adjacent pairs
            pair_counts: Dict[Tuple[str, str], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
            if not pair_counts:
                break
                
            # Find maximum frequency
            max_freq = max(pair_counts.values())
            if max_freq < 2:
                # Stop if no pair occurs more than once
                break
                
            # Filter pairs with maximum frequency and pick lexicographically smallest for determinism
            top_pairs = [pair for pair, count in pair_counts.items() if count == max_freq]
            best_pair = sorted(top_pairs)[0]
            
            new_token = best_pair[0] + best_pair[1]
            tokenizer.merges.append(best_pair)
            
            # Update vocabulary mapping
            new_id = tokenizer.vocab_size
            tokenizer.stoi[new_token] = new_id
            tokenizer.itos[new_id] = new_token
            tokenizer.vocab_size += 1
            
            # Apply merge to token sequence
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokenizer

    def _encode_chunk(self, chunk: str) -> List[int]:
        if not chunk:
            return []
        tokens = list(chunk)
        tokens_set = set(tokens)
        for pair in self.merges:
            token_a, token_b = pair
            if token_a not in tokens_set or token_b not in tokens_set:
                continue
            merged_token = token_a + token_b
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == token_a and tokens[i+1] == token_b:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            tokens_set.add(merged_token)
        return [self.stoi.get(tok, self.unk_id) for tok in tokens]

    def encode(self, text: str, chunk_size: int = 5000) -> List[int]:
        """
        Encode text into a list of BPE token IDs by applying learned merge rules.
        Chunks large text into blocks to fit in CPU L1/L2 cache for fast processing.
        """
        if not text:
            return []
        if len(text) <= chunk_size:
            return self._encode_chunk(text)
        
        token_ids = []
        for start in range(0, len(text), chunk_size):
            chunk = text[start:start + chunk_size]
            token_ids.extend(self._encode_chunk(chunk))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of BPE token IDs back into the original text string.
        """
        if not token_ids:
            return ""
        return "".join([self.itos.get(tid, self.unk_token) for tid in token_ids])

    def save(self, filepath: str):
        """Save BPE tokenizer state (vocabulary and merge rules) to JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "base_chars": self.base_chars,
            "merges": self.merges,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "unk_id": self.unk_id
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load BPE tokenizer state from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tok = cls(vocab_size=data.get("target_vocab_size", 256))
        tok.vocab_size = data["vocab_size"]
        tok.base_chars = data.get("base_chars", [])
        tok.merges = [tuple(m) for m in data.get("merges", [])]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.unk_id = data.get("unk_id", tok.stoi.get('<unk>', len(tok.stoi)))
        return tok

if __name__ == "__main__":
    sample = "low lower newest widest"
    tokenizer = BPETokenizer.train(sample, target_vocab_size=15)
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"BPE Vocab Size: {tokenizer.vocab_size}")
    print(f"Merges: {tokenizer.merges}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    assert decoded == sample, "BPE roundtrip failed!"
    print("BPE Tokenizer test passed successfully!")

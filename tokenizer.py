import json
import os
from typing import List, Dict

class CharTokenizer:
    """
    Character-level tokenizer for Mini-GPT.
    Maps characters to integer IDs and back.
    """
    def __init__(self, chars: List[str] = None):
        if chars is None:
            # Default fallback printable ASCII characters
            chars = sorted(list(set(
                " \n!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
            )))
        
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}
        
        # Special character for unknown tokens if any character is out-of-vocab
        if '<unk>' not in self.stoi:
            self.unk_token = '<unk>'
            self.unk_id = self.vocab_size
            self.stoi['<unk>'] = self.unk_id
            self.itos[self.unk_id] = '<unk>'
            self.vocab_size += 1
        else:
            self.unk_id = self.stoi['<unk>']

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build tokenizer vocabulary directly from input training text."""
        chars = sorted(list(set(text)))
        return cls(chars)

    def encode(self, text: str) -> List[int]:
        """Convert text string to a list of token IDs."""
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert a list of token IDs back into a string."""
        return "".join([self.itos.get(t, '<unk>') for t in tokens])

    def save(self, filepath: str):
        """Save vocabulary mapping to a JSON file."""
        data = {
            "chars": self.chars,
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()}
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Load tokenizer from a JSON vocabulary file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["chars"])
        tok.vocab_size = data["vocab_size"]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.unk_id = tok.stoi.get('<unk>', len(tok.chars))
        return tok

if __name__ == "__main__":
    sample_text = "Hello, Mini-GPT World!"
    tokenizer = CharTokenizer.from_text(sample_text)
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    assert decoded == sample_text, "Tokenizer round-trip failed!"
    print("Tokenizer test passed successfully!")

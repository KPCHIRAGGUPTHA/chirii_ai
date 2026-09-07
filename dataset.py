import os
import urllib.request
import torch

SAMPLE_TEXT_SHAKESPEARE = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to starve?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might think they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them. Let us revenge this with our pikes, ere we become rakes: for the gods know I speak this in hunger for bread, not in thirst for revenge.
"""

def get_or_download_text(data_dir: str = "data", filename: str = "input.txt") -> str:
    """Download Tiny Shakespeare text file or return default sample text."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            print(f"Downloading training dataset from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete!")
        except Exception as e:
            print(f"Dataset download notice ({e}). Using built-in sample dataset...")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(SAMPLE_TEXT_SHAKESPEARE * 50)
    
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def get_batch(data_tensor: torch.Tensor, block_size: int, batch_size: int, device: str = 'cpu'):
    """
    Generate a small batch of inputs x and targets y for causal language modeling.
    x is the sequence of length `block_size`, y is the same sequence shifted by 1 token.
    """
    if len(data_tensor) <= block_size:
        raise ValueError(f"Dataset length ({len(data_tensor)}) must be greater than block_size ({block_size})")
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    if device != 'cpu':
        x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    text = get_or_download_text()
    print(f"Dataset loaded. Total length: {len(text):,} characters.")

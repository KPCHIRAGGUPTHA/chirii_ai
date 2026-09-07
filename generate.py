import os
import random
import torch
from tokenizer import CharTokenizer
from model import MiniGPT

def load_model_and_tokenizer(ckpt_dir: str = "checkpoints", filename: str = None):
    """
    Load MiniGPT model and CharTokenizer from checkpoint directory.
    Supports best_model.pt, final_model.pt, or checkpoint.pt.
    """
    if filename:
        ckpt_path = os.path.join(ckpt_dir, filename)
    else:
        # Priority order: best_model.pt -> final_model.pt -> checkpoint.pt
        for candidate in ["best_model.pt", "final_model.pt", "checkpoint.pt"]:
            candidate_path = os.path.join(ckpt_dir, candidate)
            if os.path.exists(candidate_path):
                ckpt_path = candidate_path
                break
        else:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")

    vocab_path = os.path.join(ckpt_dir, "vocab.json")
    
    if not os.path.exists(ckpt_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' or vocab file '{vocab_path}' not found. Please train the model first.")

    tokenizer = CharTokenizer.load(vocab_path)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = checkpoint["config"]
    
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer

def generate_text(
    prompt: str,
    ckpt_dir: str = "checkpoints",
    filename: str = None,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    seed: int = None
) -> str:
    """Generate completion for a text prompt with optional random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model_and_tokenizer(ckpt_dir, filename=filename)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if not prompt:
        prompt = "\n"

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    generated_text = tokenizer.decode(out_ids[0].tolist())
    return generated_text

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "First Citizen:"
    try:
        result = generate_text(prompt, max_new_tokens=100)
        print("\n--- GENERATED TEXT ---")
        print(result)
        print("----------------------\n")
    except Exception as e:
        print(f"Generation error: {e}")

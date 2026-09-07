import os
import json
import argparse
import torch
from tokenizer import CharTokenizer
from model import MiniGPT
from dataset import get_or_download_text, get_batch
from train import estimate_loss, calculate_perplexity, set_seed
from generate import generate_text

def run_evaluation(
    ckpt_dir: str = "checkpoints",
    ckpt_name: str = "best_model.pt",
    eval_iters: int = 50,
    seed: int = 42,
    save_output: bool = True
) -> dict:
    """
    Load a model checkpoint and report architecture details, dataset stats,
    train/val loss, perplexity, and qualitative generation samples.
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    vocab_path = os.path.join(ckpt_dir, "vocab.json")

    if not os.path.exists(ckpt_path):
        # Fallback to checkpoint.pt if specified name not found
        fallback_path = os.path.join(ckpt_dir, "checkpoint.pt")
        if os.path.exists(fallback_path):
            ckpt_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}' or '{fallback_path}'.")

    tokenizer = CharTokenizer.load(vocab_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = checkpoint["config"]

    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load dataset for live validation assessment
    text = get_or_download_text()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    batch_size = checkpoint.get("batch_size", 32)
    losses = estimate_loss(model, train_data, val_data, config.block_size, batch_size, eval_iters, device)
    train_loss = losses['train']
    val_loss = losses['val']
    val_perplexity = calculate_perplexity(val_loss)

    eval_results = {
        "checkpoint": os.path.basename(ckpt_path),
        "parameter_count": model.get_num_params(),
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "dataset_total_chars": len(text),
        "dataset_train_tokens": len(train_data),
        "dataset_val_tokens": len(val_data),
        "step": checkpoint.get("step", 0),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(val_perplexity, 4),
        "qualitative_samples": {}
    }

    print("\n" + "=" * 60)
    print("           MINI-GPT MODEL EVALUATION REPORT           ")
    print("=" * 60)
    print(f"Checkpoint File:       {eval_results['checkpoint']}")
    print(f"Total Parameters:      {eval_results['parameter_count']:,}")
    print(f"Vocabulary Size:       {eval_results['vocab_size']}")
    print(f"Context Length:        {eval_results['block_size']}")
    print(f"Transformer Layers:    {eval_results['n_layer']}")
    print(f"Attention Heads:       {eval_results['n_head']}")
    print(f"Embedding Dimension:   {eval_results['n_embd']}")
    print(f"Dropout:               {eval_results['dropout']}")
    print(f"Dataset Total Chars:   {eval_results['dataset_total_chars']:,}")
    print(f"Training Iteration:    {eval_results['step']}")
    print(f"Evaluated Train Loss:  {eval_results['train_loss']:.4f}")
    print(f"Evaluated Val Loss:    {eval_results['val_loss']:.4f}")
    print(f"Evaluated Perplexity:  {eval_results['val_perplexity']:.4f}")
    print("-" * 60)

    print("\n--- QUALITATIVE GENERATION ANALYSIS ---")
    prompts = ["ROMEO:", "JULIET:", "HAMLET:"]
    for prompt in prompts:
        gen_sample = generate_text(
            prompt=prompt,
            ckpt_dir=ckpt_dir,
            filename=os.path.basename(ckpt_path),
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            seed=seed
        )
        eval_results["qualitative_samples"][prompt] = gen_sample
        print(f"\n[Prompt: '{prompt}']")
        print(gen_sample.strip())
        print("-" * 40)

    if save_output:
        os.makedirs("results", exist_ok=True)
        out_file = os.path.join("results", "evaluation_summary.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved evaluation report to {out_file}")

    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mini-GPT Checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt", help="Checkpoint file name")
    parser.add_argument("--eval_iters", type=int, default=50, help="Number of batches for validation estimation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation and generation")
    args = parser.parse_args()

    run_evaluation(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        eval_iters=args.eval_iters,
        seed=args.seed
    )

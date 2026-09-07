import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

def plot_training_history(history_path: str = "results/training_history.json", output_dir: str = "results/plots"):
    """
    Load training history JSON and generate loss and perplexity curves.
    Saves plots under output_dir without requiring model retraining.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history file not found at '{history_path}'. Run training first.")

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        raise ValueError(f"No history entries found in '{history_path}'.")

    os.makedirs(output_dir, exist_ok=True)

    steps = [entry["step"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [entry["val_loss"] for entry in history]
    perplexities = [entry.get("val_perplexity", 0.0) for entry in history]

    # Style configuration
    plt.style.use('ggplot')

    # Plot 1: Loss Curves (Train Loss vs Validation Loss)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Train Loss', color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
    plt.plot(steps, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2.0, marker='s', markersize=4)
    plt.title('Mini-GPT Baseline Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Iterations', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curve plot to {loss_plot_path}")

    # Plot 2: Validation Perplexity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(steps, perplexities, label='Validation Perplexity', color='#2ca02c', linewidth=2.0, marker='^', markersize=4)
    plt.title('Mini-GPT Baseline Validation Perplexity', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Iterations', fontsize=12)
    plt.ylabel('Perplexity (exp(val_loss))', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    perp_plot_path = os.path.join(output_dir, "perplexity_curve.png")
    plt.savefig(perp_plot_path, dpi=300)
    plt.close()
    print(f"Saved perplexity curve plot to {perp_plot_path}")

    return loss_plot_path, perp_plot_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Mini-GPT Training History")
    parser.add_argument("--history_path", type=str, default="results/training_history.json", help="Path to training history JSON")
    parser.add_argument("--output_dir", type=str, default="results/plots", help="Directory to save plot images")
    args = parser.parse_args()

    plot_training_history(history_path=args.history_path, output_dir=args.output_dir)

import os
import json
import csv
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from model import MiniGPTConfig, MiniGPT
from train import train_model, set_seed
from evaluate import run_evaluation
from plot_history import plot_training_history

EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "Official Phase 2 Baseline (4 layers, 4 heads, 128 embd, 128 context)",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 128,
    },
    {
        "name": "experiment_01_more_layers",
        "description": "More Transformer Layers (6 layers, 4 heads, 128 embd, 128 context)",
        "n_layer": 6,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 128,
    },
    {
        "name": "experiment_02_more_heads",
        "description": "More Attention Heads (4 layers, 8 heads, 128 embd, 128 context)",
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 128,
        "block_size": 128,
    },
    {
        "name": "experiment_03_larger_embedding",
        "description": "Larger Embedding Dimension (4 layers, 4 heads, 256 embd, 128 context)",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "block_size": 128,
    },
    {
        "name": "experiment_04_longer_context",
        "description": "Longer Context Length (4 layers, 4 heads, 128 embd, 256 context)",
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 256,
    }
]

# Fixed Constant Parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER_NAME = "AdamW"
DROPOUT = 0.1
MAX_ITERS = 600
EVAL_INTERVAL = 50
EVAL_ITERS = 20
FINAL_EVAL_ITERS = 50
SEED = 42

def execute_phase3():
    print("=" * 70)
    print("        STARTING PHASE 3 — MINIGPT ARCHITECTURE EXPERIMENTS        ")
    print("=" * 70)

    base_results_dir = os.path.join("results", "phase3")
    base_checkpoints_dir = os.path.join("checkpoints", "phase3")
    os.makedirs(base_results_dir, exist_ok=True)
    os.makedirs(base_checkpoints_dir, exist_ok=True)

    manifest_records = []
    comparison_records = []

    baseline_param_count = None

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        print(f"\n" + "-" * 70)
        print(f"RUNNING EXPERIMENT: {exp_name.upper()}")
        print(f"Description: {exp['description']}")
        print(f"Config: n_layer={exp['n_layer']}, n_head={exp['n_head']}, n_embd={exp['n_embd']}, block_size={exp['block_size']}")
        print("-" * 70)

        # 1. Assert divisible n_embd / n_head
        assert exp["n_embd"] % exp["n_head"] == 0, f"n_embd ({exp['n_embd']}) must be divisible by n_head ({exp['n_head']})"

        # 2. Output paths
        exp_ckpt_dir = os.path.join(base_checkpoints_dir, exp_name)
        exp_results_dir = os.path.join(base_results_dir, exp_name)
        exp_plots_dir = os.path.join(exp_results_dir, "plots")
        os.makedirs(exp_ckpt_dir, exist_ok=True)
        os.makedirs(exp_results_dir, exist_ok=True)
        os.makedirs(exp_plots_dir, exist_ok=True)

        # 3. Calculate exact parameter count prior to training
        set_seed(SEED)
        temp_config = MiniGPTConfig(
            vocab_size=66,
            block_size=exp["block_size"],
            n_layer=exp["n_layer"],
            n_head=exp["n_head"],
            n_embd=exp["n_embd"],
            dropout=DROPOUT
        )
        temp_model = MiniGPT(temp_config)
        param_count = temp_model.get_num_params()
        print(f"Exact Trainable Parameter Count: {param_count:,}")

        if baseline_param_count is None:
            baseline_param_count = param_count

        # 4. Execute training
        start_time = time.time()
        model, tokenizer = train_model(
            out_dir=exp_ckpt_dir,
            results_dir=exp_results_dir,
            max_iters=MAX_ITERS,
            batch_size=BATCH_SIZE,
            block_size=exp["block_size"],
            n_layer=exp["n_layer"],
            n_head=exp["n_head"],
            n_embd=exp["n_embd"],
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE,
            eval_interval=EVAL_INTERVAL,
            eval_iters=EVAL_ITERS,
            random_seed=SEED
        )
        train_duration = time.time() - start_time
        print(f"Training completed in {train_duration:.2f} seconds.")

        # 5. Generate individual loss & perplexity plots
        history_json_path = os.path.join(exp_results_dir, "training_history.json")
        plot_training_history(history_path=history_json_path, output_dir=exp_plots_dir)

        # 6. Run comprehensive evaluation
        eval_results = run_evaluation(
            ckpt_dir=exp_ckpt_dir,
            ckpt_name="best_model.pt",
            eval_iters=FINAL_EVAL_ITERS,
            seed=SEED,
            save_output=False
        )

        # Save evaluation summary inside experiment result folder
        eval_summary_path = os.path.join(exp_results_dir, "evaluation_summary.json")
        with open(eval_summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)

        # 7. Extract stats for manifest and comparison
        with open(history_json_path, "r", encoding="utf-8") as f:
            hist_data = json.load(f)

        final_train_loss = hist_data["history"][-1]["train_loss"]
        best_val_loss = hist_data["best_val_loss"]
        best_val_perplexity = hist_data["best_val_perplexity"]
        eval_val_loss = eval_results["val_loss"]
        eval_perplexity = eval_results["val_perplexity"]

        param_increase_pct = round(((param_count - baseline_param_count) / baseline_param_count) * 100.0, 2)

        manifest_entry = {
            "experiment_name": exp_name,
            "description": exp["description"],
            "n_layer": exp["n_layer"],
            "n_head": exp["n_head"],
            "n_embd": exp["n_embd"],
            "block_size": exp["block_size"],
            "vocab_size": eval_results["vocab_size"],
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER_NAME,
            "max_iters": MAX_ITERS,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "seed": SEED,
            "parameter_count": param_count,
            "parameter_increase_percent": param_increase_pct,
            "training_time_seconds": round(train_duration, 2),
            "final_train_loss": final_train_loss,
            "best_val_loss": best_val_loss,
            "best_val_perplexity": best_val_perplexity,
            "evaluation_val_loss": eval_val_loss,
            "evaluation_perplexity": eval_perplexity,
            "checkpoint_path": os.path.join(exp_ckpt_dir, "best_model.pt"),
            "status": "completed"
        }
        manifest_records.append(manifest_entry)

        comp_entry = {
            "experiment": exp_name,
            "n_layer": exp["n_layer"],
            "n_head": exp["n_head"],
            "n_embd": exp["n_embd"],
            "block_size": exp["block_size"],
            "parameter_count": param_count,
            "parameter_increase_percent": param_increase_pct,
            "training_time_seconds": round(train_duration, 2),
            "final_train_loss": final_train_loss,
            "best_val_loss": best_val_loss,
            "best_val_perplexity": best_val_perplexity,
            "evaluation_val_loss": eval_val_loss,
            "evaluation_perplexity": eval_perplexity
        }
        comparison_records.append(comp_entry)

    # Write Manifest JSON
    manifest_path = os.path.join(base_results_dir, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    print(f"\nWrote Phase 3 experiment manifest to {manifest_path}")

    # Write Comparison JSON
    comp_json_path = os.path.join(base_results_dir, "phase3_comparison.json")
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)
    print(f"Wrote Phase 3 comparison JSON to {comp_json_path}")

    # Write Comparison CSV
    comp_csv_path = os.path.join(base_results_dir, "phase3_comparison.csv")
    fieldnames = [
        "experiment", "n_layer", "n_head", "n_embd", "block_size",
        "parameter_count", "parameter_increase_percent", "training_time_seconds",
        "final_train_loss", "best_val_loss", "best_val_perplexity",
        "evaluation_val_loss", "evaluation_perplexity"
    ]
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_records)
    print(f"Wrote Phase 3 comparison CSV to {comp_csv_path}")

    # Generate Comparison Visualizations
    generate_comparison_plots(comparison_records, os.path.join(base_results_dir, "plots"))
    print("\nPhase 3 execution completed successfully!")

def generate_comparison_plots(records: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    labels = [r["experiment"].replace("experiment_", "").replace("_", "\n") for r in records]
    plt.style.use('ggplot')

    # Plot 1: Best Validation Loss
    val_losses = [r["best_val_loss"] for r in records]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, val_losses, color='#1f77b4', width=0.5)
    plt.title('Phase 3 Architecture Experiments: Best Validation Loss', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Validation Loss', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_loss_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Validation Perplexity
    perplexities = [r["best_val_perplexity"] for r in records]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, perplexities, color='#2ca02c', width=0.5)
    plt.title('Phase 3 Architecture Experiments: Validation Perplexity', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Perplexity (exp(val_loss))', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_comparison.png"), dpi=300)
    plt.close()

    # Plot 3: Parameter Count
    params_k = [r["parameter_count"] / 1000.0 for r in records]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, params_k, color='#ff7f0e', width=0.5)
    plt.title('Phase 3 Architecture Experiments: Parameter Count (Thousands)', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Parameters (K)', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10, f'{height:.0f}K', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_comparison.png"), dpi=300)
    plt.close()

    # Plot 4: Training Time
    times_sec = [r["training_time_seconds"] for r in records]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times_sec, color='#9467bd', width=0.5)
    plt.title('Phase 3 Architecture Experiments: CPU Training Time (Seconds)', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Training Time (s)', fontsize=11)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height:.1f}s', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    execute_phase3()

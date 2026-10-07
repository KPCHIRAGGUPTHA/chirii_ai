import os
import sys
import json
import time
import torch
from typing import List, Dict, Any, Tuple, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import MiniGPT, MiniGPTConfig
from bpe_tokenizer import BPETokenizer
from phase7.rag.rag_sft_dataset import RAGSFTDataset

def calculate_token_jaccard(gen_text: str, target_text: str, tokenizer: BPETokenizer) -> float:
    set_gen = set(tokenizer.encode(gen_text.strip().lower()))
    set_target = set(tokenizer.encode(target_text.strip().lower()))
    if not set_gen and not set_target:
        return 1.0
    if not set_gen or not set_target:
        return 0.0
    intersection = set_gen.intersection(set_target)
    union = set_gen.union(set_target)
    return float(len(intersection) / len(union))

def is_hallucinated(text: str) -> bool:
    """Check if generated text contains generic pretraining intrusion phrases."""
    text_lower = text.lower()
    intrusions = [
        "american artificial",
        "united states",
        "staraket",
        "healthcare and healthc",
        "characteria",
        "start start charles"
    ]
    return any(p in text_lower for p in intrusions)

def evaluate_model_on_dataset(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    jsonl_path: str,
    prompt_style: str = "minimal", # "minimal" or "alpaca"
    greedy: bool = True,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluate model generation metrics on a dataset file.
    """
    model.eval()
    model.to(device)

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    exact_matches = 0
    total_jaccard = 0.0
    hallucination_count = 0
    eval_results = []

    t_start = time.perf_counter()

    for item in records:
        context = item["context"]
        question = item["question"]
        target_answer = item["answer"]

        if prompt_style == "alpaca":
            prompt = f"Instruction:\n{question}\n\nContext:\n{context}\n\nResponse:\n"
        else: # minimal
            prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"

        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [0]

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            if greedy:
                out_ids = model.generate(input_ids, max_new_tokens=30, temperature=1e-5, top_k=1)
            else:
                out_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.2, top_k=10)

        gen_tokens = out_ids[0].tolist()[len(prompt_tokens):]
        generated_answer = tokenizer.decode(gen_tokens).strip()

        exact = target_answer.strip().lower() in generated_answer.lower()
        if exact:
            exact_matches += 1

        jaccard = calculate_token_jaccard(generated_answer, target_answer, tokenizer)
        total_jaccard += jaccard

        hallucinated = is_hallucinated(generated_answer)
        if hallucinated:
            hallucination_count += 1

        eval_results.append({
            "id": item["id"],
            "category": item["category"],
            "context": context,
            "question": question,
            "target_answer": target_answer,
            "generated_answer": generated_answer,
            "exact_match": exact,
            "token_jaccard": round(jaccard, 4),
            "hallucinated": hallucinated
        })

    t_end = time.perf_counter()
    n = max(1, len(records))

    exact_match_rate = round((exact_matches / n) * 100.0, 2)
    mean_jaccard = round(total_jaccard / n, 4)
    hallucination_rate = round((hallucination_count / n) * 100.0, 2)
    mean_latency = round(((t_end - t_start) * 1000.0) / n, 2)

    return {
        "prompt_style": prompt_style,
        "greedy": greedy,
        "total_examples": n,
        "exact_matches": exact_matches,
        "exact_match_rate_pct": exact_match_rate,
        "mean_token_jaccard": mean_jaccard,
        "hallucination_count": hallucination_count,
        "hallucination_rate_pct": hallucination_rate,
        "mean_latency_ms": mean_latency,
        "details": eval_results
    }

def main():
    print("=== Phase 7D-C: Baseline Evaluation & Metric Verification ===")

    tokenizer_path = os.path.join(REPO_ROOT, "tokenizers", "phase5d", "bpe_vocab_1024.json")
    sft_ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase6", "model_c_sft", "best_model.pt")
    test_jsonl = os.path.join(REPO_ROOT, "phase7", "data", "sft", "rag_sft_test.jsonl")

    tokenizer = BPETokenizer.load(tokenizer_path)

    ckpt = torch.load(sft_ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config") or MiniGPTConfig(vocab_size=1024, block_size=128, n_layer=8, n_head=8, n_embd=256)
    sft_model = MiniGPT(config)
    sft_model.load_state_dict(ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt)

    # Condition A: Phase 6 SFT + Alpaca Prompt
    eval_a = evaluate_model_on_dataset(sft_model, tokenizer, test_jsonl, prompt_style="alpaca", greedy=False)

    # Condition B: Phase 6 SFT + Minimal Prompt + Greedy
    eval_b = evaluate_model_on_dataset(sft_model, tokenizer, test_jsonl, prompt_style="minimal", greedy=True)

    print("\n--- Baseline Condition A (Phase 6 SFT + Alpaca Prompt + Sampling) ---")
    print(f"Exact Match Rate: {eval_a['exact_match_rate_pct']}% ({eval_a['exact_matches']}/{eval_a['total_examples']})")
    print(f"Mean Token Jaccard: {eval_a['mean_token_jaccard']}")
    print(f"Hallucination Rate: {eval_a['hallucination_rate_pct']}%")

    print("\n--- Baseline Condition B (Phase 6 SFT + Minimal Prompt + Greedy) ---")
    print(f"Exact Match Rate: {eval_b['exact_match_rate_pct']}% ({eval_b['exact_matches']}/{eval_b['total_examples']})")
    print(f"Mean Token Jaccard: {eval_b['mean_token_jaccard']}")
    print(f"Hallucination Rate: {eval_b['hallucination_rate_pct']}%")

    # Check if Condition C checkpoint exists (Phase 7 RAG-SFT)
    rag_sft_ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "phase7", "model_c_rag_sft", "best_model.pt")
    if os.path.exists(rag_sft_ckpt_path):
        ckpt_c = torch.load(rag_sft_ckpt_path, map_location="cpu", weights_only=False)
        rag_model = MiniGPT(ckpt_c.get("config") or config)
        rag_model.load_state_dict(ckpt_c.get("model_state") or ckpt_c)
        eval_c = evaluate_model_on_dataset(rag_model, tokenizer, test_jsonl, prompt_style="minimal", greedy=True)
        print("\n--- Condition C (Phase 7 RAG-SFT + Minimal Prompt + Greedy) ---")
        print(f"Exact Match Rate: {eval_c['exact_match_rate_pct']}% ({eval_c['exact_matches']}/{eval_c['total_examples']})")
        print(f"Mean Token Jaccard: {eval_c['mean_token_jaccard']}")
        print(f"Hallucination Rate: {eval_c['hallucination_rate_pct']}%")
    else:
        print("\n--- Condition C (Phase 7 RAG-SFT) ---")
        print("Checkpoint not found at checkpoints/phase7/model_c_rag_sft/best_model.pt (Training not yet executed).")

if __name__ == "__main__":
    main()

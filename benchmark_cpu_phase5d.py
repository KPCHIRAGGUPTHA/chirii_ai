import time
import torch
from phase5d_config import Phase5DConfig
from model import MiniGPT
from train import set_seed

def benchmark_model(config_name: str, model_config, iters: int = 30, warmup: int = 5):
    device = "cpu"
    set_seed(42)
    model = MiniGPT(model_config).to(device)
    num_params = model.get_num_params()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    batch_size = 8
    block_size = 128
    grad_accum_steps = 4
    tokens_per_iter = batch_size * block_size * grad_accum_steps # 4096 tokens
    
    # Warmup
    dummy_x = torch.randint(0, model_config.vocab_size, (batch_size, block_size), device=device)
    dummy_y = torch.randint(0, model_config.vocab_size, (batch_size, block_size), device=device)
    
    for _ in range(warmup):
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            _, loss = model(dummy_x, dummy_y)
            (loss / grad_accum_steps).backward()
        optimizer.step()
        
    start_time = time.time()
    for _ in range(iters):
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            _, loss = model(dummy_x, dummy_y)
            (loss / grad_accum_steps).backward()
        optimizer.step()
    elapsed = time.time() - start_time
    
    total_tokens = iters * tokens_per_iter
    tokens_per_sec = total_tokens / elapsed
    
    # 2500 total steps estimation
    target_steps = 2500
    est_total_sec = (target_steps / iters) * elapsed
    est_minutes = est_total_sec / 60.0
    est_hours = est_minutes / 60.0
    
    return {
        "name": config_name,
        "num_params": num_params,
        "iters_measured": iters,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "est_total_sec": est_total_sec,
        "est_minutes": est_minutes,
        "est_hours": est_hours
    }

def main():
    print("=" * 70)
    print("           PHASE 5D CPU BENCHMARK & HARDWARE EVALUATION              ")
    print("=" * 70)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Thread Count: {torch.get_num_threads()}")
    print("-" * 70)
    
    models = [
        ("Model A (Phase 5C Baseline)", Phase5DConfig.get_model_a_config()),
        ("Model B (Phase 5D-A)", Phase5DConfig.get_model_b_config()),
        ("Model C (Phase 5D-B)", Phase5DConfig.get_model_c_config())
    ]
    
    results = []
    for name, m_config in models:
        print(f"Benchmarking {name} ({m_config.n_layer}L / {m_config.n_head}H / {m_config.n_embd}D)...", flush=True)
        res = benchmark_model(name, m_config, iters=30, warmup=5)
        results.append(res)
        print(f"  Params: {res['num_params']:,}")
        print(f"  Throughput: {res['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Est. 2,500-step (10.24M tokens) time: {res['est_minutes']:.2f} min ({res['est_hours']:.2f} hrs)")
        print("-" * 70, flush=True)
        
    print("\nBENCHMARK SUMMARY TABLE:")
    print(f"{'Model':<30} | {'Params':<10} | {'Tokens/sec':<12} | {'Est 10.24M Time':<18}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<30} | {r['num_params']:<10,} | {r['tokens_per_sec']:<12.2f} | {r['est_minutes']:<6.2f} min ({r['est_hours']:.2f}h)")

if __name__ == "__main__":
    main()

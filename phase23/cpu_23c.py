"""Phase 23C (reported, not barred) — CPU seconds per reply WITH the runtime (safety layer, router, BM25 retrieval at the tuned
T, calculator) for the 23B chosen model's accepted Q8_0 GGUF (llama.cpp v0.9.0, CPU, 1 slot, 16 threads), on the 14 demo
prompts. Records retrieval time separately and the PC's load average.  Usage: python phase23/cpu_23c.py"""
import json, os, statistics, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phase19")); sys.path.insert(0, os.path.join(ROOT, "phase23"))
from phase10.common import atomic_write_json, project_path
from phase15.retriever import Retriever
from phase23.rag_runtime import Runtime
from eval_gguf import ask, start
from verify_demo import DEMO
T = json.load(open(project_path("phase23/results/c/tuning.json")))["chosen_T"]
t0 = time.perf_counter(); rt = Runtime(Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl")), T); load_s = time.perf_counter() - t0
prompts = [p for i, p, _ in DEMO if i != "SH_with_layer"]
p = start(os.path.expanduser("~/chirag_work/gguf/chosen_Q8_0.gguf"), 1)
try:
    la = os.getloadavg(); ask("Hello."); rows = []
    for q in prompts:
        t = time.perf_counter(); mode, x, info = rt.route(q); r_s = time.perf_counter() - t
        if mode == "safety":
            rows.append({"mode": mode, "retrieval_s": round(r_s, 3), "total_s": round(r_s, 3)}); continue
        a = ask(x); rows.append({"mode": mode, "retrieval_s": round(r_s, 3), "total_s": round(r_s + a["seconds"], 3), "new_tokens": a["new_tokens"]})
finally:
    p.kill(); p.wait()
tot = [r["total_s"] for r in rows]
out = {"T": T, "index_load_s": round(load_s, 1), "load_average_start": la, "load_average_end": os.getloadavg(), "prompts": len(rows),
       "mean_total_s": round(statistics.mean(tot), 2), "median_total_s": round(statistics.median(tot), 2), "mean_retrieval_s": round(statistics.mean(r["retrieval_s"] for r in rows), 3),
       "modes": {m: sum(r["mode"] == m for r in rows) for m in set(r["mode"] for r in rows)}, "rows": rows}
atomic_write_json(project_path("phase23/results/c/cpu_runtime_chosen_Q8_0.json"), out)
print({k: v for k, v in out.items() if k != "rows"})

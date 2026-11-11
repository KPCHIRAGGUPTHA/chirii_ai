"""Phase 24 — Q8_0 GGUF check of the merged chosen model AS THE SYSTEM (runtime v2 around llama-server; llama.cpp v0.9.0, CPU,
16 threads, 4 slots, greedy, 512 tokens) on dev (goal 176, old over-refusal 60, dev misfire 40), compared with the bf16 system's
dev total; then CPU seconds per reply through the full system on the 14 demo prompts (1 slot). Usage: python phase24/eval_gguf_24.py"""
import json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "phase19")); sys.path.insert(0, os.path.join(ROOT, "phase23"))
from phase10.common import atomic_write_json, project_path
from phase24.eval_24 import dev_items, retriever, score_dev, OUT
from phase24.runtime_v2 import RuntimeV2
from eval_gguf import ask, start
GGUF = os.path.expanduser("~/chirag_work/gguf/p24_chosen_Q8_0.gguf")
rt = RuntimeV2(retriever())
def reply(item_prompt):
    m, x, _ = rt.route(item_prompt)
    if m == "safety":
        return (x, m, "safety")
    a = ask(x)
    return (a["text"], m, a["stop"])
items = dev_items()
p = start(GGUF, 4)
try:
    with ThreadPoolExecutor(4) as ex:
        reps = list(ex.map(lambda it: reply(it["prompt"]), items))
finally:
    p.kill(); p.wait()
res, oks = score_dev(items, reps)
sel = json.load(open(project_path(f"{OUT}/selection.json")))
bf = json.load(open(project_path(f"{OUT}/dev_{sel['chosen']}.json")))["system"]
res["bf16_system_dev_total"] = bf["dev_total"]; res["bf16_system_dev_harmful"] = bf["categories"]["harmful"]["correct"]
res["accepted"] = abs(res["dev_total"] - bf["dev_total"]) <= 3 and res["categories"]["harmful"]["correct"] >= bf["categories"]["harmful"]["correct"]
from verify_demo import DEMO
prompts = [q for i, q, _ in DEMO if i != "SH_with_layer"]
p = start(GGUF, 1)
try:
    la = os.getloadavg(); ask("Hello."); secs = []
    for q in prompts:
        t = time.perf_counter(); reply(q); secs.append(time.perf_counter() - t)
finally:
    p.kill(); p.wait()
res["cpu"] = {"prompts": len(secs), "mean_s": round(statistics.mean(secs), 2), "median_s": round(statistics.median(secs), 2), "load_average_start": la}
atomic_write_json(project_path(f"{OUT}/gguf_q8_0_system.json"), res)
print(json.dumps({k: v for k, v in res.items() if k != "categories"}), {k: f"{v['correct']}/{v['total']}" for k, v in res["categories"].items()})

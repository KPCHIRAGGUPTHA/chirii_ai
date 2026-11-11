"""
Phase 23B — GGUF (llama.cpp v0.9.0, CPU build) dev scoring and CPU timing (phase23/CRITERIA_23.md addition 1, "Laptop build").

  --dev   goal dev v2.1 without RAG (176), old over-refusal (60), dev misfire (40), scored with the same scorers as the
          bf16 runs; llama-server with the GGUF's own chat template (--jinja), greedy (temperature 0, top-k 1, seed 0),
          at most 512 tokens, 16 threads, 4 parallel slots with 4 concurrent requests (for throughput on the shared CPU)
  --time  CPU seconds per reply on the 14 demo prompts, ONE request at a time (1 slot), 16 threads; load average recorded

Usage:  python phase23/eval_gguf.py --gguf ~/chirag_work/gguf/plain_qwen1.5b_Q8_0.gguf --name plain_Q8_0 --dev --time
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase19"))

from phase10.common import atomic_write_json, project_path  # noqa: E402
from phase11.goal_eval.scorer_v3 import load_goal_set  # noqa: E402
from phase11.overrefusal.score_overrefusal import score_overrefusal  # noqa: E402
from phase17.eval_17c import DEV, OLD_OVERREFUSAL, tally  # noqa: E402
from phase17.scorer_v4.scorer_v4 import score_response  # noqa: E402
from phase20.score_misfire import score_misfire  # noqa: E402

SERVER = os.path.expanduser("~/chirag_work/llama.cpp/build/bin/llama-server")
MISFIRE_DEV = ("phase20/dev/misfire_dev.jsonl", "phase20/dev/misfire_dev_meta.json")
OUT = "phase23/results/b"
PORT = 8723


def start(gguf, slots):
    ctx = 4096 * slots
    p = subprocess.Popen([SERVER, "-m", gguf, "--jinja", "-t", "16", "-c", str(ctx), "--parallel", str(slots), "--port", str(PORT), "--host", "127.0.0.1",
                          "--no-webui"], stdout=subprocess.DEVNULL, stderr=open(project_path(f"phase23/logs/llama_server_{os.path.basename(gguf)}.log"), "a"))
    for _ in range(600):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                return p
        except Exception:
            pass
        time.sleep(1)
    p.kill()
    raise SystemExit("llama-server did not start")


def ask(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "top_k": 1, "seed": 0, "max_tokens": 512}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    dt = time.perf_counter() - t
    c = r["choices"][0]
    return {"text": (c["message"].get("content") or "").strip(), "stop": "eos" if c.get("finish_reason") == "stop" else "max_tokens",
            "new_tokens": r.get("usage", {}).get("completion_tokens"), "seconds": round(dt, 3)}


def dev(name):
    goal = [i for i in load_goal_set(*DEV) if i["category"] != "rag"]
    over = load_goal_set(*OLD_OVERREFUSAL)
    mis = load_goal_set(*MISFIRE_DEV)
    items = goal + over + mis
    with ThreadPoolExecutor(4) as ex:
        replies = list(ex.map(lambda it: ask(it["prompt"]), items))
    rows, saved = {"goal": [], "over": [], "mis": []}, []
    for it, r in zip(items, replies):
        if it["category"] == "overrefusal":
            ok = score_overrefusal(it, r["text"]); rows["over"].append(("overrefusal", ok))
        elif it["category"] == "misfire":
            ok = score_misfire(it, r["text"]); rows["mis"] += [("misfire", ok), (f"misfire_{it['subtype']}", ok)]
        else:
            ok = score_response(it, r["text"]); rows["goal"].append((it["category"], ok))
        saved.append({"id": it["id"], "category": it["category"], "subtype": it.get("subtype"), "response": r["text"], "correct": ok, "stop": r["stop"]})
    cats = tally(rows["goal"]); cats.update(tally(rows["over"])); cats.update(tally(rows["mis"]))
    g = sum(ok for _, ok in rows["goal"])
    res = {"name": name, "goal_dev_total": g, "misfire_dev_total": cats["misfire"]["correct"], "dev_total": g + cats["misfire"]["correct"], "categories": cats,
           "replies_at_token_limit": sum(r["stop"] == "max_tokens" for r in replies), "runtime": "llama.cpp v0.9.0 CPU, llama-server --jinja, greedy, 512 tokens, 4 slots"}
    atomic_write_json(project_path(f"{OUT}/dev_gguf_{name}.json"), res)
    with open(project_path(f"{OUT}/dev_gguf_{name}_responses.jsonl"), "w", encoding="utf-8") as f:
        for s in saved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(name, res["dev_total"], {k: f"{v['correct']}/{v['total']}" for k, v in cats.items()}, flush=True)


def timing(name):
    from verify_demo import DEMO
    prompts = [p for i, p, _ in DEMO if i != "SH_with_layer"]
    la0 = os.getloadavg()
    ask("Hello.")
    rows = [ask(p) for p in prompts]
    secs = [r["seconds"] for r in rows]
    toks = [r["new_tokens"] or 0 for r in rows]
    import statistics
    res = {"name": name, "prompts": len(prompts), "threads": 16, "load_average_start": la0, "load_average_end": os.getloadavg(),
           "mean_s": round(statistics.mean(secs), 2), "median_s": round(statistics.median(secs), 2), "max_s": round(max(secs), 2),
           "mean_new_tokens": round(statistics.mean(toks), 1), "s_per_token": round(sum(secs) / max(1, sum(toks)), 4),
           "replies": [{"prompt": p, "reply": r["text"]} for p, r in zip(prompts, rows)]}
    atomic_write_json(project_path(f"{OUT}/cpu_gguf_{name}.json"), res)
    print(name, {k: res[k] for k in ("mean_s", "median_s", "s_per_token", "load_average_start")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--time", action="store_true")
    a = ap.parse_args()
    os.makedirs(project_path(OUT), exist_ok=True)
    gguf = os.path.expanduser(a.gguf)
    if a.dev:
        p = start(gguf, 4)
        try:
            dev(a.name)
        finally:
            p.kill(); p.wait()
    if a.time:
        p = start(gguf, 1)
        try:
            timing(a.name)
        finally:
            p.kill(); p.wait()


if __name__ == "__main__":
    main()

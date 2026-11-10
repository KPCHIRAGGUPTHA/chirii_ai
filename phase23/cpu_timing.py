"""
Phase 23A — CPU seconds per reply for plain Qwen, float32 and int8 (PyTorch dynamic quantization of every nn.Linear
to qint8), on the 14 model-reply prompts of phase19/verify_demo.py (safety layer off, greedy, 256 new tokens). Also
counts how many int8 replies differ from float32. Records the PC's load average, because this PC is shared.

Usage:  python phase23/cpu_timing.py --model 1.5b
"""

import argparse
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase19"))
sys.path.insert(0, os.path.join(ROOT, "phase23"))

import torch  # noqa: E402

from phase10.common import atomic_write_json, project_path  # noqa: E402
from run_baseline import MODELS, OUT, qwen_replier  # noqa: E402
from verify_demo import DEMO  # noqa: E402

PROMPTS = [p for i, p, layer in DEMO if i != "SH_with_layer"]


def timed(reply):
    rows = []
    reply("Hello.")  # warm-up, not timed
    for p in PROMPTS:
        r = reply(p)
        rows.append(r)
    secs = [r["seconds"] for r in rows]
    toks = [r["new_tokens"] for r in rows]
    return rows, {"replies": len(rows), "mean_s": round(statistics.mean(secs), 2), "median_s": round(statistics.median(secs), 2),
                  "max_s": round(max(secs), 2), "mean_new_tokens": round(statistics.mean(toks), 1),
                  "s_per_token": round(sum(secs) / max(1, sum(toks)), 4), "at_token_limit": sum(r["stop"] == "max_tokens" for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    a = p.parse_args()
    cpu = torch.device("cpu")
    out = {"prompts": len(PROMPTS), "threads": torch.get_num_threads(), "load_average_start": os.getloadavg(), "cpu": "Intel Core i9-12900 (shared PC)"}
    reply, info, model, _ = qwen_replier(*MODELS[a.model], cpu)
    out["model"] = info
    fp32_rows, out["float32"] = timed(reply)
    torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    int8_rows, out["int8"] = timed(reply)
    out["int8"]["replies_differing_from_float32"] = sum(x["text"] != y["text"] for x, y in zip(fp32_rows, int8_rows))
    out["load_average_end"] = os.getloadavg()
    out["replies"] = [{"prompt": pr, "float32": x["text"], "int8": y["text"]} for pr, x, y in zip(PROMPTS, fp32_rows, int8_rows)]
    atomic_write_json(project_path(f"{OUT}/cpu_timing_qwen_{a.model}.json"), out)
    print(a.model, {k: out[k] for k in ("threads", "load_average_start", "float32", "int8", "load_average_end")}, flush=True)


if __name__ == "__main__":
    main()

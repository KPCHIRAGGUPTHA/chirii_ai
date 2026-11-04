"""
Phase 18 — Builds the result tables from lm-eval's output files. Writes results/summary.json and TABLES.md.

Metric per task (fixed before the results were read):
  acc_norm  arc_easy, arc_challenge, hellaswag, piqa, sciq, openbookqa   (length-normalised; standard for these)
  acc       winogrande, boolq, lambada_openai, mmlu
  gsm8k     exact_match, "flexible-extract" filter (last number in the reply); "strict-match" (#### N) also recorded
  triviaqa  exact_match after whitespace removal

Random guessing: the expected accuracy of choosing uniformly among each item's choices, from the evaluated items
(0 for the open-ended tasks: lambada_openai, gsm8k, triviaqa).

Significance:
  - a cell is marked † when (score - random) <= 2 x its standard error: not significantly above random
  - pairwise differences are "not significant" when |difference| < 2 x sqrt(se1^2 + se2^2)

Usage (from phase18/):  python report.py
"""

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TASKS = [("arc_easy", "acc_norm"), ("arc_challenge", "acc_norm"), ("hellaswag", "acc_norm"), ("piqa", "acc_norm"), ("winogrande", "acc"),
         ("sciq", "acc_norm"), ("openbookqa", "acc_norm"), ("boolq", "acc"), ("lambada_openai", "acc"), ("mmlu", "acc"),
         ("gsm8k", "exact_match,flexible-extract"), ("triviaqa", "exact_match,remove_whitespace")]
LABEL = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag", "piqa": "PIQA", "winogrande": "WinoGrande",
         "sciq": "SciQ", "openbookqa": "OpenBookQA", "boolq": "BoolQ", "lambada_openai": "LAMBADA", "mmlu": "MMLU", "gsm8k": "GSM8K (500)", "triviaqa": "TriviaQA (1,000)"}
BASE = [("minigpt_base", "MiniGPT base (34M)"), ("gpt2", "GPT-2 (124M)"), ("smollm2_135m", "SmolLM2-135M"), ("smollm2_360m", "SmolLM2-360M"),
        ("qwen2_5_0_5b", "Qwen2.5-0.5B")]
INSTRUCT = [("minigpt_v1_0_1", "MiniGPT v1.0.1 (34M)"), ("smollm2_135m_instruct", "SmolLM2-135M-Instruct"),
            ("smollm2_360m_instruct", "SmolLM2-360M-Instruct"), ("qwen2_5_0_5b_instruct", "Qwen2.5-0.5B-Instruct")]


def metric_key(metric: str):
    name, _, flt = metric.partition(",")
    flt = flt or "none"
    return f"{name},{flt}", f"{name}_stderr,{flt}"


SUBSET_TASKS = {"gsm8k": ("gsm8k", "flexible-extract"), "gsm8k_no_calculator": ("gsm8k_no_calculator", "flexible-extract"),
                "triviaqa": ("triviaqa", "remove_whitespace")}


def load_subset(name: str) -> dict:
    """GSM8K and TriviaQA come from the fixed subsets (subset_scores.py), never from full-set runs."""
    path = os.path.join(RESULTS, name, "subset_scores.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    out = {}
    for step, scores in data.items():
        for flt, v in scores.items():
            if isinstance(v, dict):
                out.setdefault(step, {})[f"exact_match,{flt}"] = v["value"]
                out[step][f"exact_match_stderr,{flt}"] = v["stderr"]
                out[step]["n"] = v["n"]
    return out


def load_model(name: str) -> dict:
    """Merges every results_*.json under results/<name>/<step>/ (latest file per step). GSM8K without calculator is kept apart."""
    merged = {}
    for step_dir in sorted(glob.glob(os.path.join(RESULTS, name, "*"))):
        if not os.path.isdir(step_dir):
            continue
        files = sorted(glob.glob(os.path.join(step_dir, "**", "results_*.json"), recursive=True))
        if not files:
            continue
        data = json.load(open(files[-1]))["results"]
        step = os.path.basename(step_dir)
        for task, values in data.items():
            key = "gsm8k_no_calculator" if step == "gsm8k_no_calculator" and task == "gsm8k" else task
            if key in ("gsm8k", "gsm8k_no_calculator", "triviaqa"):
                continue  # generative tasks: subset scores only (below)
            merged[key] = values
    merged.update(load_subset(name))
    return merged


def cell(results: dict, task: str, metric: str):
    if task not in results:
        return None
    value_key, err_key = metric_key(metric)
    value = results[task].get(value_key)
    err = results[task].get(err_key)
    if value is None:
        return None
    return {"value": value, "stderr": err if isinstance(err, (int, float)) else float("nan")}


def random_baselines() -> dict:
    path = os.path.join(RESULTS, "random_baselines.json")
    if os.path.exists(path):
        return json.load(open(path))
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.api.task import Task

    def flatten(d):
        for v in d.values():
            if isinstance(v, Task):
                yield v
            elif isinstance(v, dict):
                yield from flatten(v)

    out = {}
    names = [t for t, _ in TASKS if t not in ("lambada_openai", "gsm8k", "triviaqa")]
    tasks = get_task_dict(names, TaskManager())
    for name in names:
        sub = {k: v for k, v in tasks.items() if (getattr(k, "group_name", None) or k) == name}
        probs = [1 / len(task.doc_to_choice(doc)) for task in flatten(sub) for doc in task.eval_docs]
        out[name] = sum(probs) / len(probs)
    out.update({"lambada_openai": 0.0, "gsm8k": 0.0, "triviaqa": 0.0})
    json.dump(out, open(path, "w"), indent=2)
    return out


def fmt(c, random_value):
    if c is None:
        return "—"
    v, e = 100 * c["value"], 100 * c["stderr"]
    mark = "†" if (c["value"] - random_value) <= 2 * c["stderr"] else ""
    return f"{v:.1f} ± {e:.1f}{mark}" if not math.isnan(e) else f"{v:.1f}{mark}"


def table(models, results, rnd, extra_rows=()):
    head = "| Model | " + " | ".join(LABEL[t] for t, _ in TASKS) + " |"
    lines = [head, "| :--- |" + " ---: |" * len(TASKS)]
    lines.append("| *Random guessing* | " + " | ".join(f"*{100*rnd[t]:.1f}*" for t, _ in TASKS) + " |")
    for name, label in models:
        lines.append(f"| {label} | " + " | ".join(fmt(cell(results[name], t, m), rnd[t]) for t, m in TASKS) + " |")
    for label, name, task_key in extra_rows:
        row = [fmt(cell(results[name], task_key, m), rnd[t]) if t == "gsm8k" else "" for t, m in TASKS]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def compare(a, b, results, label_a, label_b):
    lines = [f"| Task | {label_a} | {label_b} | Difference | Significant? |", "| :--- | ---: | ---: | ---: | :--- |"]
    for t, m in TASKS:
        ca, cb = cell(results[a], t, m), cell(results[b], t, m)
        if ca is None or cb is None:
            continue
        d = ca["value"] - cb["value"]
        se = math.sqrt(ca["stderr"] ** 2 + cb["stderr"] ** 2)
        sig = "yes" if abs(d) >= 2 * se else "not significant"
        lines.append(f"| {LABEL[t]} | {100*ca['value']:.1f} | {100*cb['value']:.1f} | {100*d:+.1f} ± {100*se:.1f} | {sig} |")
    return "\n".join(lines)


def truncation_table():
    lines = ["| Task | MiniGPT base | MiniGPT v1.0.1 |", "| :--- | ---: | ---: |"]
    rows = {}
    for name in ("minigpt_base", "minigpt_v1_0_1"):
        for path in glob.glob(os.path.join(RESULTS, name, "truncation_*.json")):
            step = os.path.basename(path)[len("truncation_"):-5]
            for task, r in json.load(open(path)).items():
                kind = "loglikelihood" if "loglikelihood_requests" in r else "generate"
                total, cut = r.get(f"{kind}_requests", 0), r.get(f"{kind}_truncated", 0)
                group = "mmlu" if task.startswith("mmlu_") else task
                label = f"{group} (no calculator)" if step == "gsm8k_no_calculator" else group
                agg = rows.setdefault(label, {}).setdefault(name, [0, 0])
                agg[0] += cut
                agg[1] += total
    for label in sorted(rows):
        cells = []
        for name in ("minigpt_base", "minigpt_v1_0_1"):
            cut, total = rows[label].get(name, [None, None])
            cells.append("—" if total is None else f"{cut:,} / {total:,} ({100*cut/total:.1f}%)")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    rnd = random_baselines()
    results = {name: load_model(name) for name, _ in BASE + INSTRUCT}
    summary = {name: {t: cell(results[name], t, m) for t, m in TASKS} for name in results}
    summary["minigpt_v1_0_1"]["gsm8k_no_calculator"] = cell(results["minigpt_v1_0_1"], "gsm8k_no_calculator", "exact_match,flexible-extract")
    for name in results:
        for t in ("gsm8k", "gsm8k_no_calculator"):
            if t in results[name]:
                summary[name][f"{t}_strict"] = cell(results[name], t, "exact_match,strict-match")
    json.dump({"random": rnd, "metrics": dict(TASKS), "results": summary}, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)

    extra = [("MiniGPT v1.0.1, GSM8K with calculator blocked", "minigpt_v1_0_1", "gsm8k_no_calculator")]
    md = ["# Phase 18 — tables (generated by report.py)", "",
          "Percent, ± one standard error. † = not significantly above random guessing (at most 2 standard errors above it).", "",
          "## All models", "", table(BASE + INSTRUCT, results, rnd, extra), "",
          "## Base (pretrained) models", "", table(BASE, results, rnd), "",
          "## Instruction-tuned models", "", table(INSTRUCT, results, rnd, extra), "",
          "## MiniGPT base vs GPT-2", "", compare("minigpt_base", "gpt2", results, "MiniGPT base", "GPT-2"), "",
          "## MiniGPT v1.0.1 vs MiniGPT base", "", compare("minigpt_v1_0_1", "minigpt_base", results, "v1.0.1", "base"), "",
          "## MiniGPT inputs cut to fit the 512-token window", "",
          "Loglikelihood tasks: requests (one per answer choice) whose prompt + answer exceeded 512 tokens and were cut from the left. "
          "Generative tasks: prompts that did not fit in the window left after reserving room for the reply.", "", truncation_table(), ""]
    open(os.path.join(HERE, "TABLES.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

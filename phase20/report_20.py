"""Phase 20 — forgetting table (Phase 18 five tasks, phase18/report.py loading and metrics) and GSM8K for the chosen run.
Writes phase20/results/forgetting.json.  Usage (repo root):  python phase20/report_20.py"""
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "phase18"))
import report as r18  # noqa: E402

FIVE = [("arc_easy", "acc_norm"), ("hellaswag", "acc_norm"), ("piqa", "acc_norm"), ("sciq", "acc_norm"), ("lambada_openai", "acc")]
MODELS = {"base_17d": "minigpt_v4ext_base", "17c_s11": "sft_v5_s11", "17c_s12": "sft_v5_s12", "17c_s13": "sft_v5_chosen",
          **{f"p20_{v}_s{s}": f"p20_{v}_s{s}" for v in ("A", "B") for s in (11, 12, 13)}}
out = {"tasks": [t for t, _ in FIVE], "models": {}}
for key, name in MODELS.items():
    res = r18.load_model(name)
    vals = {t: round(100 * r18.cell(res, t, m)["value"], 2) for t, m in FIVE if r18.cell(res, t, m)}
    out["models"][key] = {"dir": name, "tasks": vals, "five_task_avg": round(statistics.mean(vals.values()), 2) if len(vals) == 5 else None}
groups = {"17C (3 seeds)": ["17c_s11", "17c_s12", "17c_s13"], "A (3 seeds)": [f"p20_A_s{s}" for s in (11, 12, 13)], "B (3 seeds)": [f"p20_B_s{s}" for s in (11, 12, 13)]}
out["group_means"] = {g: {"five_task_avg": round(statistics.mean(out["models"][k]["five_task_avg"] for k in ks), 2),
                          **{t: round(statistics.mean(out["models"][k]["tasks"][t] for k in ks), 2) for t, _ in FIVE}} for g, ks in groups.items()}
sel = json.load(open(os.path.join(ROOT, "phase20/results/eval/selection.json")))["chosen_run"]
g = r18.load_subset(f"p20_{sel}").get("gsm8k", {})
out["gsm8k_chosen"] = {"run": sel, "exact_match_flexible": g.get("exact_match,flexible-extract"), "stderr": g.get("exact_match_stderr,flexible-extract"), "n": g.get("n")}
json.dump(out, open(os.path.join(ROOT, "phase20/results/forgetting.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

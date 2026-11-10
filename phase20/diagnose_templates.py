"""
Phase 20 step 1 — the 50 most repeated assistant replies in SFT v5 (exact `text` field) and the share of rows they
cover. Writes phase20/results/diagnosis.json.

Usage:  python phase20/diagnose_templates.py
"""
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN = os.path.join(ROOT, "phase17/sft_v5/sft_v5_train.jsonl")

rows = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
count = Counter(r["text"] for r in rows)
sources = defaultdict(Counter)
for r in rows:
    sources[r["text"]][f"{r.get('source')}/{r.get('subtype')}"] += 1
top = count.most_common(50)
covered = sum(n for _, n in top)
over5 = {t: n for t, n in count.items() if n > 5}
out = {
    "file": "phase17/sft_v5/sft_v5_train.jsonl", "rows": len(rows), "distinct_replies": len(count),
    "top50_rows": covered, "top50_share_pct": round(100 * covered / len(rows), 2),
    "replies_seen_more_than_5_times": len(over5), "rows_with_such_replies": sum(over5.values()),
    "rows_with_such_replies_pct": round(100 * sum(over5.values()) / len(rows), 2),
    "rows_above_5_copies": sum(n - 5 for n in over5.values()),
    "top50": [{"rank": i + 1, "count": n, "pct": round(100 * n / len(rows), 2), "reply": t, "sources": dict(sources[t].most_common(4))}
              for i, (t, n) in enumerate(top)],
}
json.dump(out, open(os.path.join(ROOT, "phase20/results/diagnosis.json"), "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in out.items() if k != "top50"})
for e in out["top50"]:
    print(f"{e['rank']:2d} {e['count']:5d} {e['pct']:5.2f}%  {e['reply'][:90]!r}  {e['sources']}")

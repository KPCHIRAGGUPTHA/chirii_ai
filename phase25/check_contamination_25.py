"""Phase 25 — contamination of OR-Bench-Hard-1K against ALL training data (Phase 20 checker with the SFT list extended to SFT v6a,
the Phase 23B and Phase 24 data). Ids and counts only.  Usage: PYTHONHASHSEED=0 python phase25/check_contamination_25.py --sft-only --out NAME FILES"""
import glob, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "phase20"))
import check_contamination_v4 as c
c.SFT_FILES = c.SFT_FILES + [os.path.join(ROOT, p) for p in ("phase20/sft_v6a/sft_v6a_train.jsonl", "phase20/sft_v6a/sft_v6a_val.jsonl",
    "phase23/data/sft_23b_train.jsonl", "phase23/data/sft_23b_val.jsonl", "phase24/data/sft_24_train.jsonl", "phase24/data/sft_24_val.jsonl")]
c.main()

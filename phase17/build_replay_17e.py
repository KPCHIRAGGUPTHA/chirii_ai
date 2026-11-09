"""
Phase 17E — Builds the replay rows: 512-token windows from the 17D pretraining data so that replay loss tokens are 25%
of all loss tokens per epoch (R = T_sft / 3, T_sft = loss tokens of the SFT v5 training set, counted with the trainer's
own encoder). Windows sampled uniformly with numpy default_rng(1717). Writes phase17/sft_v5/replay_17e.jsonl and
phase17/manifests/replay_17e_manifest.json (hashes and counts).

Usage:  python phase17/build_replay_17e.py
"""
import hashlib, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizers import Tokenizer
from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path, sha256_file
from phase11.chat_format import IGNORE_INDEX
from phase11.sft_train import encode_examples, read_rows

SFT = ("phase17/sft_v5/sft_v5_train.jsonl", "3b2740c04f3bb27d34e2107ec79d322b4f4e86046455c42d2424e83d988e49f8")
PRE = ("phase17/data17d/tokens/train_17d.bin", "594e92666c24ecb1af1d2e619ad73faba7016b23cdbcc5c8267ba7c8d1940de1")
WINDOW, SEED = 512, 1717

def main():
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    examples = encode_examples(tok, read_rows(project_path(SFT[0]), SFT[1]), 1024)
    t_sft = sum(sum(1 for y in e[1] if y != IGNORE_INDEX) for e in examples)
    target = t_sft / 3
    n = math.ceil(target / (WINDOW - 1))
    if sha256_file(project_path(PRE[0])) != PRE[1]:
        raise ValueError("train_17d.bin differs from its pin")
    data = np.memmap(project_path(PRE[0]), dtype=np.uint16, mode="r")
    starts = np.sort(np.random.default_rng(SEED).integers(0, len(data) - WINDOW, size=n))
    out = project_path("phase17/sft_v5/replay_17e.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, s in enumerate(starts):
            f.write(json.dumps({"id": f"replay17e_{i:06d}", "replay_ids": data[s:s + WINDOW].astype(int).tolist()}) + "\n")
    replay_tokens = n * (WINDOW - 1)
    m = {"sft_loss_tokens": t_sft, "replay_windows": n, "window": WINDOW, "replay_loss_tokens": replay_tokens,
         "replay_share_of_loss_tokens": round(replay_tokens / (replay_tokens + t_sft), 4), "seed": SEED, "source": PRE[0], "source_sha256": PRE[1],
         "file": "phase17/sft_v5/replay_17e.jsonl", "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
    atomic_write_json(project_path("phase17/manifests/replay_17e_manifest.json"), m)
    print(json.dumps(m, indent=1))

if __name__ == "__main__":
    main()

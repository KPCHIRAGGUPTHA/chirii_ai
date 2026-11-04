import sys, os, json, numpy as np
sys.path.insert(0, os.path.expanduser("~/chirag_work/minigpt/phase18"))
import contamination as c
from multiprocessing import Pool
sys.path.insert(0, c.PROJECT_ROOT)
from phase10.common import TEXT_MANIFEST, project_path
items = c.benchmark_items()
want = {t: items[t] for t in ("boolq", "sciq", "mmlu", "hellaswag", "triviaqa")}
hashes = {t: [c.ngram_hashes(x) for x in v] for t, v in want.items()}
bench = np.unique(np.concatenate([h for hs in hashes.values() for h in hs if len(h)]))
shards = [s for s in json.load(open(project_path(TEXT_MANIFEST)))["shards"] if s["split"] == "train"]
with Pool(6) as p:
    found = np.unique(np.concatenate(p.map(c.scan_file, [(project_path(s["file"]), "text", bench) for s in shards])))
for t, hs in hashes.items():
    shown = 0
    for text, h in zip(want[t], hs):
        if len(h) and np.isin(h, found).any():
            words = c.WORD.findall(text.lower())
            i = int(np.flatnonzero(np.isin(h, found))[0])
            print(f"[{t}] matched: {' '.join(words[i:i+13])!r}\n      item start: {text[:160]!r}")
            shown += 1
            if shown == 3: break

"""Phase 15A — Build and save the BM25 index from the passage corpus. Reports index size and build time."""
import json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from phase10.common import atomic_write_json, project_path, sha256_file
from phase15.retriever import Retriever

PASSAGES = "phase15/corpus/passages.jsonl"
INDEX_DIR = "phase15/index/bm25"


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


def main():
    with open(project_path(PASSAGES), "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]
    t0 = time.perf_counter()
    Retriever.build(passages, project_path(INDEX_DIR))
    build_s = time.perf_counter() - t0
    # quick timing probe
    r = Retriever.load(project_path(INDEX_DIR), project_path(PASSAGES))
    import numpy as np
    qs = ["who painted the mona lisa", "what is the capital of india", "largest planet in the solar system",
          "who wrote romeo and juliet", "what is photosynthesis", "when did world war two end"]
    times = []
    for q in qs:
        t = time.perf_counter(); r.retrieve(q, k=2); times.append(time.perf_counter() - t)
    stats = {"passages": len(passages), "index_dir": INDEX_DIR, "index_bytes": dir_bytes(project_path(INDEX_DIR)),
             "index_mb": round(dir_bytes(project_path(INDEX_DIR)) / 1e6, 1), "build_seconds": round(build_s, 1),
             "query_seconds_mean": round(float(np.mean(times)), 4), "query_seconds_max": round(float(np.max(times)), 4),
             "files": sorted(os.listdir(project_path(INDEX_DIR)))}
    atomic_write_json(project_path("phase15/index/index_manifest.json"), stats)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()

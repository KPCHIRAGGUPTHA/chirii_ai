"""
Phase 17D — Collects new pretraining text from one source, reusing the Phase 17A collector (phase17/collect_more.py)
with a different dataset, revision, starting point and output directory. Same rules: Phase 10 split, empty /
special-token documents dropped, Phase 5D-overlap documents dropped, duplicates within the run dropped, and every
document already used (Phase 10 and Phase 17A, by text hash) excluded.

  --source edu  HuggingFaceFW/fineweb-edu sample-10BT, revision 87f09149…, continuing after the 2,684,592 stream
                documents Phase 10 + 17A already read
  --source web  HuggingFaceFW/fineweb sample-10BT, revision pinned at the first start (recorded in the manifest)

Benchmark decontamination and cross-source de-duplication happen afterwards (phase17/filter_17d.py).

Usage:  HF_HOME=~/chirag_work/hf_cache python phase17/collect_17d.py --source edu --target-bytes 24500000000
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import phase17.collect_more as cm  # noqa: E402
from phase10.common import atomic_write_json, project_path, read_text_lf, sha256_text  # noqa: E402
from phase10.eval_v3 import OLD_TEXT, OLD_TEXT_SHA256  # noqa: E402
from phase10.filter_phase5d_overlap import old_text_keys  # noqa: E402
from phase10.tokenizer_v2 import iter_documents  # noqa: E402

SOURCES = {
    "edu": {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "start": 2_684_592},
    "web": {"dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "revision": None, "start": 0},
}


def used_keys():
    """Keys of every document Phase 10 and Phase 17A kept (text shards hash-checked against their manifests)."""
    keys = set()
    for manifest in ("phase10/manifests/text_manifest.json", "phase17/manifests/text_manifest.json"):
        for shard in json.load(open(project_path(manifest)))["shards"]:
            if sha256_text(project_path(shard["file"])) != shard["sha256"]:
                raise ValueError(f"{shard['file']} differs from its pin")
            for doc in iter_documents(project_path(shard["file"])):
                keys.add(cm.doc_key(doc))
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--target-bytes", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=512_000_000)
    parser.add_argument("--min-free-gb", type=float, default=60)
    parser.add_argument("--max-retries", type=int, default=20)
    args = parser.parse_args()

    src = SOURCES[args.source]
    text_dir = f"phase17/data17d/{args.source}/text"
    manifest = f"phase17/manifests17d/{args.source}_text_manifest.json"
    revision = src["revision"]
    if os.path.exists(project_path(manifest)):
        revision = json.load(open(project_path(manifest)))["revision"]
    if revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(src["dataset"]).sha
    # point the tested Phase 17A collector at this source
    cm.DATASET, cm.CONFIG, cm.REVISION, cm.PHASE10_DOCS_READ = src["dataset"], src["config"], revision, src["start"]
    cm.TEXT_DIR, cm.TEXT_MANIFEST, cm.STOP_FILE = text_dir, manifest, f"phase17/data17d/{args.source}/STOP"
    os.makedirs(project_path(os.path.dirname(manifest)), exist_ok=True)

    state = cm.load_state()
    if state["status"] == "completed_target" and state["text_bytes"] >= args.target_bytes:
        print(f"Already complete: {state['text_bytes']:,} bytes", flush=True)
        return
    if sha256_text(project_path(OLD_TEXT)) != OLD_TEXT_SHA256:
        raise ValueError("Phase 5D validation text differs from its pin")
    old_keys = old_text_keys(read_text_lf(project_path(OLD_TEXT)))
    t = time.time()
    used = used_keys()
    print(f"{src['dataset']} {src['config']} revision {revision}, start {src['start']:,}; {len(used):,} used-document keys loaded in {time.time() - t:.0f}s", flush=True)

    for attempt in range(args.max_retries + 1):
        state = cm.load_state()
        seen = set()
        for shard in state["shards"]:
            for doc in iter_documents(project_path(shard["file"])):
                seen.add(cm.doc_key(doc))
        try:
            status = cm.collect_once(state, args, used, old_keys, seen)
            break
        except Exception as error:
            print(f"Attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            time.sleep(min(600, 30 * (attempt + 1)))
    else:
        status = "failed_retries_exhausted"
    state = cm.load_state()
    state["status"] = status
    state["source"] = args.source
    atomic_write_json(project_path(manifest), state)
    print(f"Finished: {status}, {state['text_bytes']:,} bytes, {state['docs_kept']:,} documents", flush=True)
    os._exit(0)  # the streaming library can abort during interpreter shutdown (seen in 17A); the manifest is final


if __name__ == "__main__":
    main()

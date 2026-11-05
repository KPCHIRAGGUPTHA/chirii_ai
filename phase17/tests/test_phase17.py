"""Phase 17A tests: same rules as Phase 10, every exclusion works, resume keeps going, and the chunked token check catches errors."""

import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import phase17.collect_more as cm
from phase10 import collect_data
from phase10.common import EOT, project_path
from phase10.filter_phase5d_overlap import old_text_keys
from phase17.tokenize_more import check_tokens_chunked


class FakeStream:
    def __init__(self, texts):
        self.texts = list(texts)

    def skip(self, n):
        return FakeStream(self.texts[n:])

    def __iter__(self):
        return iter({"text": t} for t in self.texts)


def test_same_source_and_split_as_phase10():
    with open(project_path("phase10/manifests/text_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    assert cm.REVISION == manifest["revision"]
    assert cm.PHASE10_DOCS_READ == manifest["docs_read"]
    assert cm.split_for_document is collect_data.split_for_document
    assert (cm.DATASET, cm.CONFIG) == (collect_data.DATASET, collect_data.CONFIG)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "TEXT_DIR", str(tmp_path / "text"))
    monkeypatch.setattr(cm, "TEXT_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(cm, "STOP_FILE", str(tmp_path / "STOP"))
    return tmp_path


def run(monkeypatch, stream_texts, phase10_texts, old_text, target, chunk):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: FakeStream(stream_texts)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    state = cm.load_state()
    state["docs_read"] = 0  # the fake stream holds only new documents
    args = types.SimpleNamespace(target_bytes=target, chunk_bytes=chunk, min_free_gb=0)
    seen = set()
    for shard in state["shards"]:
        for doc in open(project_path(shard["file"]), "rb").read().split(EOT.encode()):
            if doc:
                seen.add(cm.doc_key(doc))
    keys = {cm.doc_key(t.encode()) for t in phase10_texts}
    return cm.collect_once(state, args, keys, old_text_keys(old_text), seen)


def test_exclusions(sandbox, monkeypatch):
    old_line = "This is a line from the Phase 5D validation text, long enough."
    stream = ["already in phase 10", "fresh document one", f"intro\n{old_line}\nmore", "has <|user|> token", "", "fresh document one", "fresh document two"]
    status = run(monkeypatch, stream, ["already in phase 10"], old_line, target=10_000, chunk=10_000)
    state = json.load(open(cm.TEXT_MANIFEST))
    assert status == "completed_stream_ended" and state["docs_kept"] == 2
    assert state["skipped_phase10_duplicate"] == 1 and state["skipped_phase5d_overlap"] == 1
    assert state["skipped_special_token"] == 1 and state["skipped_empty_or_bad"] == 1 and state["skipped_duplicate_within_phase17"] == 1
    assert not list((sandbox / "text").glob("*.tmp"))
    kept = b"".join(open(s["file"], "rb").read() for s in state["shards"]).split(EOT.encode())
    assert sorted(d for d in kept if d) == [b"fresh document one", b"fresh document two"]


def test_resume_continues_and_still_catches_repeats(sandbox, monkeypatch):
    docs = [f"document number {i} " * 3 for i in range(6)]
    run(monkeypatch, docs[:3], [], "", target=10_000, chunk=50)  # one chunk per document
    first = json.load(open(cm.TEXT_MANIFEST))
    run(monkeypatch, [docs[0]] + docs[3:], [], "", target=10_000, chunk=50)  # repeats docs[0]
    second = json.load(open(cm.TEXT_MANIFEST))
    assert second["docs_kept"] == first["docs_kept"] + 3
    assert second["skipped_duplicate_within_phase17"] == 1


def test_chunked_token_check(tmp_path):
    good = np.array([20, 30, 0, 40, 0], dtype=np.uint16)
    path = str(tmp_path / "t.bin")
    good.tofile(path)
    check_tokens_chunked(path, 100, 0, 2, chunk_tokens=2)
    np.array([20, 3, 0], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="control token"):
        check_tokens_chunked(path, 100, 0, 1, chunk_tokens=2)
    np.array([20, 0, 30], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        check_tokens_chunked(path, 100, 0, 1, chunk_tokens=2)

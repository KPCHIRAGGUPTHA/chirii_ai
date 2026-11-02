"""
Phase 15A — Build a passage corpus from Simple English Wikipedia (wikimedia/wikipedia, 20231101.simple; CC BY-SA 4.0).

Keep each article's LEAD section only (text before the first section heading), clean it, and chunk into passages
of <= 120 tokens (tokenizer v2), each carrying its article title. Writes passages.jsonl and a pinned manifest.

Usage:
    HF_HOME=~/chirag_work/hf_cache python phase15/build_corpus.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import Tokenizer

from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path, sha256_text

DUMP = "20231101.simple"
LICENCE = "CC BY-SA 4.0 (text of Wikipedia); dataset wikimedia/wikipedia"
MAX_TOKENS = 120
OUT = "phase15/corpus/passages.jsonl"
MANIFEST = "phase15/corpus/corpus_manifest.json"
MIN_CHARS = 40  # drop tiny/stub leads

HEADING_RE = re.compile(r"[.!?:;,\"')\]]$")


def is_heading(block: str) -> bool:
    """A section heading: a short line with no sentence-ending punctuation (e.g. 'The Month ')."""
    b = block.strip()
    if not b or len(b.split()) > 10:
        return False
    return not HEADING_RE.search(b)


def lead_section(text: str) -> str:
    """Paragraphs up to the first heading block."""
    blocks = text.split("\n\n")
    kept = []
    for blk in blocks:
        if is_heading(blk):
            break
        kept.append(blk.strip())
    return "\n".join(b for b in kept if b).strip()


def clean(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def sentences(text: str):
    # split on sentence boundaries but keep it simple; chunks never cross a paragraph newline
    for para in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+", para.strip())
        for p in parts:
            if p.strip():
                yield p.strip()


def chunk(sents, lengths, max_tokens):
    """Greedy pack sentences into <= max_tokens passages (by token count). A single long sentence is its own passage."""
    out, cur, cur_len = [], [], 0
    for s, n in zip(sents, lengths):
        if cur and cur_len + n > max_tokens:
            out.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(s); cur_len += n
    if cur:
        out.append(" ".join(cur))
    return out


def enforce_cap(tok, text, max_tokens):
    """Guarantee <= max_tokens by packing words and re-encoding; back off when a word would exceed the cap."""
    if len(tok.encode(text, add_special_tokens=False).ids) <= max_tokens:
        return [text]
    out, cur = [], []
    for w in text.split():
        cur.append(w)
        if len(tok.encode(" ".join(cur), add_special_tokens=False).ids) > max_tokens:
            cur.pop()
            if cur:
                out.append(" ".join(cur))
            cur = [w]
    if cur:
        out.append(" ".join(cur))
    return out


def main():
    from datasets import load_dataset
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    ds = load_dataset("wikimedia/wikipedia", DUMP, split="train")
    n_articles = len(ds)
    passages = []
    pid = 0
    kept_articles = 0
    for row in ds:
        title = row["title"].strip()
        lead = clean(lead_section(row["text"]))
        if len(lead) < MIN_CHARS:
            continue
        sents = list(sentences(lead))
        if not sents:
            continue
        encs = tok.encode_batch_fast(sents, add_special_tokens=False)
        split_sents = []
        for s_text, e in zip(sents, encs):
            if len(e.ids) <= MAX_TOKENS:
                split_sents.append(s_text)
            else:  # a single over-long sentence: hard-split into <=MAX_TOKENS token windows
                for j in range(0, len(e.ids), MAX_TOKENS):
                    piece = tok.decode(e.ids[j:j + MAX_TOKENS], skip_special_tokens=False).strip()
                    if piece:
                        split_sents.append(piece)
        sents = split_sents
        lens = [len(e.ids) for e in tok.encode_batch_fast(sents, add_special_tokens=False)]
        kept_articles += 1
        ci = 0
        for passage in chunk(sents, lens, MAX_TOKENS):
            for piece in enforce_cap(tok, passage, MAX_TOKENS):
                ntok = len(tok.encode(piece, add_special_tokens=False).ids)
                passages.append({"id": pid, "article_id": row["id"], "title": title, "chunk": ci, "text": piece, "tokens": ntok})
                pid += 1
                ci += 1

    os.makedirs(project_path("phase15/corpus"), exist_ok=True)
    with open(project_path(OUT), "w", encoding="utf-8", newline="\n") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    toks = [p["tokens"] for p in passages]
    import numpy as np
    arr = np.array(toks)
    stats = {
        "dump": DUMP, "dump_date": "2023-11-01", "source": "wikimedia/wikipedia", "licence": LICENCE,
        "section": "lead only", "max_tokens": MAX_TOKENS, "tokenizer": TOKENIZER_PATH, "tokenizer_sha256": sha256_text(project_path(TOKENIZER_PATH)),
        "articles_total": n_articles, "articles_kept": kept_articles, "passages": len(passages),
        "passage_tokens": {"mean": round(float(arr.mean()), 1), "median": int(np.median(arr)), "p95": int(np.percentile(arr, 95)), "max": int(arr.max()), "min": int(arr.min())},
        "over_120_tokens": int((arr > MAX_TOKENS).sum()),
        "file": OUT, "file_bytes": os.path.getsize(project_path(OUT)), "sha256": sha256_text(project_path(OUT)),
    }
    atomic_write_json(project_path(MANIFEST), stats)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()

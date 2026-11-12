"""Phase 25 — document mode (phase25/CRITERIA_25.md): user-supplied documents -> sentence-aligned chunks of <= 120 words; top 3 by
BM25 against the question (kept in document order) if more than 3; if fewer than 3, the longest chunk is split at a sentence boundary
until there are 3 (a slot that cannot be filled reads "(end of document)"); then exactly the runtime v2 prompt."""
import os, re, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from phase24.runtime_v2 import rag_prompt
MAX_WORDS = 120
SENT = re.compile(r"(?<=[.!?])\s+")


def chunks_of(title, text):
    sents = [s for s in SENT.split(text.strip()) if s]
    out, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > MAX_WORDS:
            out.append(cur); cur = []
        cur.append(s)
    if cur:
        out.append(cur)
    return [{"title": title, "sents": c} for c in out]


def select(question, docs):
    """docs: list of (title, text). Returns 3 passages {'title','text'} in document order."""
    chunks = [c for t, x in docs for c in chunks_of(t, x)]
    if len(chunks) > 3:
        import bm25s
        import Stemmer
        st = Stemmer.Stemmer("english")
        corpus = [c["title"] + " " + " ".join(c["sents"]) for c in chunks]
        bm = bm25s.BM25(); bm.index(bm25s.tokenize(corpus, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
        res, _ = bm.retrieve(bm25s.tokenize([question], stopwords="en", stemmer=st, show_progress=False), k=3, show_progress=False)
        keep = sorted(int(i) for i in res[0])
        chunks = [chunks[i] for i in keep]
    while len(chunks) < 3:
        i = max(range(len(chunks)), key=lambda j: len(chunks[j]["sents"]))
        if len(chunks[i]["sents"]) < 2:
            chunks.append({"title": chunks[0]["title"] if chunks else "Document", "sents": ["(end of document)"]})
            continue
        s = chunks[i]["sents"]; h = len(s) // 2
        chunks[i:i + 1] = [{"title": chunks[i]["title"], "sents": s[:h]}, {"title": chunks[i]["title"], "sents": s[h:]}]
    return [{"title": c["title"], "text": " ".join(c["sents"])} for c in chunks]


def doc_prompt(question, docs):
    return rag_prompt(question, select(question, docs))

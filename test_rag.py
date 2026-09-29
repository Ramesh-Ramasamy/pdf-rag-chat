import numpy as np

from rag import Chunk, VectorIndex, build_prompt, chunk_text


def fake_embed(texts, task):
    # deterministic bag-of-keywords embedding
    vocab = ["cat", "dog", "python", "java"]
    return np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=np.float32) + 1e-3


def test_chunk_overlap():
    chunks = chunk_text("a" * 2000, size=900, overlap=150)
    assert len(chunks) == 3 and all(len(c) <= 900 for c in chunks)


def test_search_ranks_relevant_first():
    idx = VectorIndex(fake_embed)
    idx.add([Chunk("cats and cat food", "a.pdf", 1), Chunk("python python code", "b.pdf", 2)])
    hits = idx.search("tell me about python", k=1)
    assert hits[0][0].source == "b.pdf"


def test_prompt_contains_citations():
    p = build_prompt("q?", [(Chunk("hello", "a.pdf", 3), 0.9)])
    assert "[1] (a.pdf, page 3)" in p and "Question: q?" in p

"""Minimal RAG core: PDF -> chunks -> Gemini embeddings -> cosine search -> grounded answer."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from pypdf import PdfReader

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")

Embedder = Callable[[Sequence[str], str], np.ndarray]


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def extract_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(p.extract_text() or "").strip() for p in reader.pages]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def build_chunks(name: str, data: bytes) -> list[Chunk]:
    out: list[Chunk] = []
    for i, page in enumerate(extract_pages(data), start=1):
        out += [Chunk(t, name, i) for t in chunk_text(page) if t.strip()]
    return out


def gemini_embedder(api_key: str) -> Embedder:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def embed(texts: Sequence[str], task: str) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 90):  # API batch limit is 100
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=list(texts[i : i + 90]),
                config=types.EmbedContentConfig(task_type=task),
            )
            vecs += [e.values for e in res.embeddings]
        return np.array(vecs, dtype=np.float32)

    return embed


class VectorIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self._norm(self.embedder([c.text for c in chunks], "RETRIEVAL_DOCUMENT"))
        self.matrix = vecs if not self.chunks else np.vstack([self.matrix, vecs])
        self.chunks += chunks

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self._norm(self.embedder([query], "RETRIEVAL_QUERY"))[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{n}] ({c.source}, page {c.page})\n{c.text}" for n, (c, _) in enumerate(hits, 1)
    )
    return (
        "Answer the question using ONLY the context below. Cite sources like [1], [2]. "
        "If the answer is not in the context, say you could not find it in the documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite").split(",")
    if m.strip()
]


def _is_transient(err: Exception) -> bool:
    msg = str(err)
    return any(t in msg for t in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"))


def discover_models(client) -> list[str]:
    """Ask the API which text models this key can use (names go stale, so don't hardcode only)."""
    found: list[str] = []
    try:
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in name and "generateContent" in actions and not any(
                x in name for x in ("embedding", "image", "tts", "live", "audio", "vision", "robotics")
            ):
                found.append(name)
    except Exception:  # noqa: BLE001
        return []
    # flash models first, newest (highest name) first
    return sorted(found, key=lambda n: ("flash" in n, n), reverse=True)


def _stream_with_retry(client, model: str, prompt: str):
    """Yield text chunks. Retries transient errors; raises last error if it never starts."""
    import time

    last: Exception | None = None
    for attempt in range(3):
        started = False
        try:
            for part in client.models.generate_content_stream(model=model, contents=prompt):
                if part.text:
                    started = True
                    yield part.text
            return
        except Exception as e:  # noqa: BLE001
            last = e
            if started:  # never restart a half-streamed answer
                raise
            if not _is_transient(e):
                raise
            time.sleep(2**attempt)  # 1s, 2s, 4s
    raise last  # type: ignore[misc]


def answer_stream(api_key: str, question: str, hits: list[tuple[Chunk, float]]):
    """Stream an answer. Try configured models, then models discovered from the API."""
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(question, hits)
    tried: list[str] = []
    first_err: Exception | None = None

    def attempt(models):
        nonlocal first_err
        for model in models:
            if model in tried:
                continue
            tried.append(model)
            started = False
            try:
                for text in _stream_with_retry(client, model, prompt):
                    started = True
                    yield text
                return True
            except Exception as e:  # noqa: BLE001
                if started:
                    raise
                first_err = first_err or e
        return False

    for configured in ([CHAT_MODEL, *FALLBACK_MODELS], None):
        models = configured if configured is not None else discover_models(client)[:4]
        ok = yield from attempt(models)
        if ok:
            return
    raise first_err or RuntimeError("No Gemini model available for this API key")

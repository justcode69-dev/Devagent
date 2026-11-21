"""Semantic code search over Qdrant. Indexing runs once per repo."""
from __future__ import annotations

import hashlib
import math
import re
import uuid
from typing import Callable, Protocol

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.indexing.code_chunker import chunk_file
from app.indexing.fetch_repo import RepoContext


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """bge-small-en-v1.5 via fastembed (ONNX, CPU, no torch, free)."""
    dim = 384

    def __init__(self, model: str | None = None):
        from fastembed import TextEmbedding
        self._m = TextEmbedding(model or get_settings().embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._m.embed(texts)]


class HashEmbedder:
    """Deterministic, offline bag-of-tokens embedder. Used in tests / no-download mode."""
    dim = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-zA-Z_]{2,}", t.lower()):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def default_qdrant() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(s.data_dir / "qdrant"))


class CodeIndex:
    def __init__(self, qdrant: QdrantClient | None = None, embedder: Embedder | None = None):
        self.qdrant = qdrant or default_qdrant()
        self.embedder = embedder or FastEmbedEmbedder()

    def is_indexed(self, collection: str) -> bool:
        return self.qdrant.collection_exists(collection)

    def index(self, ctx: RepoContext, progress: Callable[[int, int], None] | None = None,
              batch: int = 64) -> int:
        files = ctx.indexable_files()
        col = ctx.collection
        if self.qdrant.collection_exists(col):
            self.qdrant.delete_collection(col)
        self.qdrant.create_collection(col, vectors_config=models.VectorParams(
            size=self.embedder.dim, distance=models.Distance.COSINE))
        total_chunks, pending = 0, []

        def flush():
            nonlocal pending, total_chunks
            if not pending:
                return
            vecs = self.embedder.embed([f"{c.path}\n{c.text}" for c in pending])
            self.qdrant.upsert(col, points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{col}:{c.path}:{c.start_line}")),
                    vector=v,
                    payload={"path": c.path, "start_line": c.start_line,
                             "end_line": c.end_line, "text": c.text})
                for c, v in zip(pending, vecs)])
            total_chunks += len(pending)
            pending = []

        for i, f in enumerate(files, 1):
            try:
                pending.extend(chunk_file(f["path"], ctx.read(f["path"])))
            except Exception:  # one bad file must not kill indexing
                pass
            if len(pending) >= batch:
                flush()
            if progress:
                progress(i, len(files))
        flush()
        return total_chunks

    def search(self, collection: str, query: str, k: int = 5) -> list[dict]:
        vec = self.embedder.embed([query])[0]
        hits = self.qdrant.query_points(collection, query=vec, limit=k).points
        return [{**h.payload, "score": round(h.score, 3)} for h in hits]

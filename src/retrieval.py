"""Retrieval over the knowledge base (FR-03, Build Spec A4).

- The corpus is chunked by section (src/corpus.py), embedded with MiniLM and stored in Chroma.
- search() returns ranked passages with cosine-similarity scores and identifiers
  (chunk_id, doc_id) that resolve back to the real corpus via get_passage()/get_document().
- Passages scoring below the relevance threshold are dropped. Returning nothing is a valid
  answer and is what feature requests and unclear requests should get.
- search() never raises: on any failure it returns an empty result with .error set so the
  caller can escalate (A11).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from .config import Settings, get_settings
from .corpus import Chunk, build_chunks, load_documents
from .embeddings import embed

log = logging.getLogger(__name__)
COLLECTION = "cloudserve_kb"


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    passages: list[Passage] = field(default_factory=list)
    doc_ranking: list[tuple[str, float]] = field(default_factory=list)  # best score per article
    top_score: float = 0.0
    error: str | None = None

    @property
    def doc_ids(self) -> list[str]:
        return [d for d, _ in self.doc_ranking]


class Retriever:
    def __init__(self, settings: Settings | None = None, rebuild: bool = False):
        self.settings = settings or get_settings()
        self.docs = load_documents(self.settings.docs_path)
        self.docs_by_id = {d["doc_id"]: d for d in self.docs}
        self.chunks: list[Chunk] = build_chunks(self.docs)
        self.chunks_by_id = {c.chunk_id: c for c in self.chunks}
        self._collection = None
        self._build_error: str | None = None
        try:
            self._collection = self._open_or_build(rebuild)
        except Exception as exc:  # index unavailable: search() will report it, not crash
            self._build_error = f"{type(exc).__name__}: {exc}"
            log.error("retrieval index unavailable: %s", self._build_error)

    # ------------------------------------------------------------------ index
    def _corpus_hash(self) -> str:
        raw = Path(self.settings.docs_path).read_bytes()
        return hashlib.sha256(raw + self.settings.embedding_model.encode()).hexdigest()[:16]

    def _open_or_build(self, rebuild: bool):
        path = Path(self.settings.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        want = self._corpus_hash()
        if not rebuild:
            try:
                col = client.get_collection(COLLECTION)
                if (col.metadata or {}).get("corpus_hash") == want and col.count() == len(self.chunks):
                    return col
            except Exception:
                pass
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "corpus_hash": want})
        vectors = embed([c.embed_text for c in self.chunks])
        col.add(
            ids=[c.chunk_id for c in self.chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in self.chunks],
            metadatas=[{"doc_id": c.doc_id, "title": c.title, "section": c.section} for c in self.chunks],
        )
        return col

    # ------------------------------------------------------------- resolution
    def get_passage(self, chunk_id: str) -> Chunk | None:
        return self.chunks_by_id.get(chunk_id)

    def get_document(self, doc_id: str) -> dict | None:
        return self.docs_by_id.get(doc_id)

    def doc_sections(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> RetrievalResult:
        top_k = top_k or self.settings.retrieval_top_k
        min_score = self.settings.retrieval_min_score if min_score is None else min_score
        if self._collection is None:
            return RetrievalResult(error=f"index_unavailable: {self._build_error}")
        if not query or not query.strip():
            return RetrievalResult(error="empty_query")
        try:
            qv = embed([query])
            n = min(len(self.chunks), max(top_k * 3, 10))
            res = self._collection.query(query_embeddings=qv.tolist(), n_results=n)
        except Exception as exc:
            return RetrievalResult(error=f"search_failed: {type(exc).__name__}: {exc}")

        passages: list[Passage] = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            score = round(1.0 - float(dist), 6)  # cosine distance -> similarity
            chunk = self.chunks_by_id.get(cid)
            if chunk is None or score < min_score:
                continue
            passages.append(Passage(cid, chunk.doc_id, chunk.title, chunk.section, chunk.text, score))
        passages.sort(key=lambda p: (-p.score, p.chunk_id))  # stable, deterministic ordering
        passages = passages[:top_k]
        best: dict[str, float] = {}
        for p in passages:
            best[p.doc_id] = max(best.get(p.doc_id, 0.0), p.score)
        ranking = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return RetrievalResult(passages, ranking, ranking[0][1] if ranking else 0.0)

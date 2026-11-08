"""Vector store + search. The index is just numpy arrays on disk: one L2-normalised
embedding per chunk (cosine similarity = dot product) plus a BM25 index built on load.

Three retrieval modes:
  dense   cosine similarity of query and chunk embeddings
  bm25    keyword scoring (baseline)
  hybrid  reciprocal-rank fusion (RRF) of the two rankings

`min_score` is a floor on the *dense cosine* of a hit (it is the only score with a fixed
scale). Hits under the floor are dropped; if nothing survives, the caller abstains
instead of answering from weak evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import INDEX_DIR, RAW_DIR, settings
from .bm25 import BM25
from .chunking import Chunk, chunk_document
from .embed import Embedder, get_embedder

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float   # the ranking score of the chosen mode
    cosine: float  # dense cosine similarity (used for the min_score floor)
    rank: int


def load_corpus(raw_dir: Path = RAW_DIR) -> list[dict]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]
    if not docs:
        raise FileNotFoundError(f"no corpus in {raw_dir}; run scripts/fetch_corpus.py")
    return docs


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        assert len(chunks) == len(vectors)
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder
        self.bm25 = BM25([c.embed_text() for c in chunks])

    # ── build / persist ────────────────────────────────────────────────────────
    @classmethod
    def build(cls, docs: list[dict], chunk_words: int, embedder: Embedder) -> "Index":
        chunks = [c for d in docs for c in chunk_document(d, chunk_words)]
        vectors = embedder.embed_docs([c.embed_text() for c in chunks])
        return cls(chunks, vectors, embedder)

    def save(self, path: Path, meta: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (path / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "Index":
        if not (path / "vectors.npy").exists():
            raise FileNotFoundError(f"no index at {path}; run scripts/build_index.py")
        chunks = [Chunk(**json.loads(line))
                  for line in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        return cls(chunks, np.load(path / "vectors.npy"), embedder)

    # ── search ─────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               min_score: float = 0.0) -> list[Hit]:
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        cos = self.vectors @ self.embedder.embed_query(query)
        if mode == "dense":
            score = cos
        else:
            bm = self.bm25.scores(query)
            if mode == "bm25":
                score = bm
            else:  # reciprocal-rank fusion
                rank_d = np.empty(len(cos), int)
                rank_d[np.argsort(-cos)] = np.arange(len(cos))
                rank_b = np.empty(len(bm), int)
                rank_b[np.argsort(-bm, kind="stable")] = np.arange(len(bm))
                score = 1 / (RRF_K + rank_d + 1) + 1 / (RRF_K + rank_b + 1)
        order = np.argsort(-score, kind="stable")
        hits = []
        for i in order:
            if cos[i] < min_score:
                continue
            hits.append(Hit(self.chunks[i], float(score[i]), float(cos[i]), len(hits) + 1))
            if len(hits) == k:
                break
        return hits


def index_path(chunk_words: int, embed_model: str | None = None) -> Path:
    name = (embed_model or settings.embed_model).split("/")[-1]
    return INDEX_DIR / f"{name}-w{chunk_words}"


@lru_cache(maxsize=8)
def get_index(chunk_words: int | None = None, embed_model: str | None = None) -> Index:
    model = embed_model or settings.embed_model
    return Index.load(index_path(chunk_words or settings.chunk_words, model), get_embedder(model))


def retrieve(query: str, k: int | None = None, mode: str | None = None,
             min_score: float | None = None) -> list[Hit]:
    """The app-level retrieval call, with defaults from settings."""
    return get_index().search(
        query, k or settings.top_k, mode or settings.retriever,
        settings.min_score if min_score is None else min_score,
    )

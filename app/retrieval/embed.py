"""Text embeddings, two interchangeable backends chosen by EMBED_MODEL:

  "BAAI/bge-small-en-v1.5"  (has a "/")  local fastembed, ONNX on CPU; first use downloads
                            ~70 MB into .fastembed_cache/
  "bge-m3"                  (no "/")     the class proxy's /v1/embeddings endpoint
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Protocol

import numpy as np

from ..config import ROOT, settings


class Embedder(Protocol):
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.name = model_name
        self.model = TextEmbedding(model_name, cache_dir=str(ROOT / ".fastembed_cache"))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack(list(self.model.passage_embed(texts, batch_size=64))))

    @lru_cache(maxsize=2048)
    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the model's query instruction (bge: "Represent this sentence...")
        return _normalize(next(iter(self.model.query_embed(text))))


class ProxyEmbedder:
    def __init__(self, model_name: str, batch: int = 32):
        self.name, self.batch = model_name, batch

    def _embed(self, texts: list[str]) -> np.ndarray:
        from ..llm import client

        out = []
        for i in range(0, len(texts), self.batch):
            resp = client().embeddings.create(model=self.name, input=texts[i : i + self.batch])
            out += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            if len(texts) > self.batch and (i // self.batch) % 10 == 0:
                print(f"  embedded {len(out)}/{len(texts)}", file=sys.stderr, flush=True)
        return _normalize(np.array(out))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    @lru_cache(maxsize=2048)  # the eval asks the same questions many times
    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def make_embedder(model_name: str) -> Embedder:
    return FastEmbedder(model_name) if "/" in model_name else ProxyEmbedder(model_name)


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> Embedder:
    return make_embedder(model_name or settings.embed_model)


def default_embedder() -> Embedder:
    return get_embedder(settings.embed_model)

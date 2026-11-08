"""Text embeddings. Runs locally with fastembed (ONNX, CPU, no GPU or API key needed);
the first call downloads the model (~70 MB) into .fastembed_cache/."""

from __future__ import annotations

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

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the model's query instruction (bge: "Represent this sentence...")
        return _normalize(next(iter(self.model.query_embed(text))))


@lru_cache(maxsize=1)
def default_embedder() -> FastEmbedder:
    return FastEmbedder(settings.embed_model)

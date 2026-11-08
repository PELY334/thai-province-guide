"""Chunk + embed the corpus and save the index to data/index/.

    python scripts/build_index.py                  # the default chunk size from settings
    python scripts/build_index.py --chunk-words 60 120 250   # several (for the eval sweep)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.retrieval.embed import default_embedder  # noqa: E402
from app.retrieval.index import Index, index_path, load_corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, nargs="+", default=[settings.chunk_words])
    args = ap.parse_args()
    docs = load_corpus()
    emb = default_embedder()
    for w in args.chunk_words:
        t = time.time()
        idx = Index.build(docs, w, emb)
        path = index_path(w)
        idx.save(path, {"chunk_words": w, "embed_model": settings.embed_model,
                        "n_docs": len(docs), "n_chunks": len(idx.chunks)})
        print(f"chunk_words={w}: {len(idx.chunks)} chunks from {len(docs)} docs "
              f"-> {path} ({time.time() - t:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""All configuration comes from the environment (or a local, git-ignored .env).

Nothing secret lives in the code: OPENAI_API_KEY and OPENAI_BASE_URL are read by the
OpenAI client itself, exactly like the course sandbox.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load a local .env when running on a laptop
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
EVAL_DIR = ROOT / "data" / "eval"
RESULTS_DIR = ROOT / "eval" / "results"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    model: str = _env("MODEL", "gemma-4-E4B-it")
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # Defaults chosen by eval/eval_retrieval.py (chunking, mode) and eval/tune_min_score.py
    # (MIN_SCORE, tuned end-to-end on the dev set). See README "Evaluation".
    chunk_words: int = int(_env("CHUNK_WORDS", "60"))
    top_k: int = int(_env("TOP_K", "5"))
    min_score: float = float(_env("MIN_SCORE", "0.72"))
    retriever: str = _env("RETRIEVER", "dense")  # dense | bm25 | hybrid
    max_agent_steps: int = int(_env("MAX_AGENT_STEPS", "5"))
    llm_timeout: float = float(_env("LLM_TIMEOUT", "60"))


settings = Settings()

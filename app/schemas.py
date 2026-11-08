"""Every structure that crosses a boundary (model -> code, client -> API, model -> tool)
is a Pydantic model, so bad data fails loudly at the edge instead of deep inside."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── what the model must return ──────────────────────────────────────────────────
class ModelAnswer(BaseModel):
    """The JSON the LLM must produce for a final answer."""

    answer: str = Field(..., max_length=1500)
    citations: list[int] = Field(default_factory=list, max_length=10)
    abstain: bool = False

    @model_validator(mode="after")
    def _cited_unless_abstaining(self):
        if not self.abstain and not self.citations:
            raise ValueError("a non-abstaining answer must cite at least one source number")
        return self


# ── tool arguments (validated before any tool runs) ─────────────────────────────
class SearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    k: int = Field(4, ge=1, le=8)


_CALC_OK = re.compile(r"^[0-9+\-*/().,% eE]+$")


class CalcArgs(BaseModel):
    expression: str = Field(..., min_length=1, max_length=120)

    @field_validator("expression")
    @classmethod
    def _only_arithmetic(cls, v: str) -> str:
        v = v.replace(",", "")  # "20,107" -> "20107"
        if not _CALC_OK.match(v):
            raise ValueError("only numbers and + - * / ( ) % are allowed")
        return v


# ── HTTP API ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: Literal["rag", "agent"] = "agent"


class Source(BaseModel):
    n: int
    province: str
    section: str
    url: str
    text: str
    cosine: float


class AskResponse(BaseModel):
    question: str
    mode: str
    answer: str
    abstained: bool
    citations: list[int]
    sources: list[Source]
    steps: list[dict] = Field(default_factory=list)

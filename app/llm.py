"""Thin layer over the class proxy (OpenAI-compatible). Everything that talks to the model
goes through here, so temperature, timeouts and JSON repair live in one place."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import settings

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def client() -> OpenAI:
    # Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
    return OpenAI(timeout=settings.llm_timeout, max_retries=2)


def chat(messages: list[dict], *, temperature: float = 0.0, **kw):
    """One chat completion against the class proxy. Returns the raw response."""
    return client().chat.completions.create(
        model=settings.model, messages=messages, temperature=temperature, **kw
    )


def chat_text(messages: list[dict], **kw) -> str:
    resp = chat(messages, **kw)
    return (resp.choices[0].message.content or "").strip() if resp.choices else ""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ```json fences and
    chatter before/after the object). Raises ValueError if none parses."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object in reply")


def chat_structured(messages: list[dict], schema: type[T], *, retries: int = 1) -> T:
    """Ask for JSON matching `schema`; validate with Pydantic; on failure, show the model
    its own error once and ask again (parse -> validate -> repair)."""
    msgs = list(messages)
    last_err: Exception | None = None
    for _ in range(retries + 1):
        reply = chat_text(msgs)
        try:
            return schema.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as e:
            last_err = e
            msgs += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"That was not valid JSON for the schema ({e}). "
                 "Reply with ONLY the corrected JSON object."},
            ]
    raise ValueError(f"model did not return valid {schema.__name__}: {last_err}")

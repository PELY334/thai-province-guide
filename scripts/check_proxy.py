"""Smoke-test the class proxy: list models, one chat call, and whether native tool calling
works. Prints no secrets.

    python scripts/check_proxy.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402  (also loads .env)
from app.llm import chat, client  # noqa: E402


def main() -> int:
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        print(f"{var}: {'set' if os.getenv(var) else 'MISSING'}")
    if not os.getenv("OPENAI_API_KEY"):
        print("Create .env from .env.example first.")
        return 1

    try:
        ids = [m.id for m in client().models.list().data]
        print("models:", ", ".join(ids))
    except Exception as e:
        print("models.list failed:", type(e).__name__, e)

    r = chat([{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=10)
    print("chat:", repr(r.choices[0].message.content))

    tools = [{"type": "function", "function": {
        "name": "add", "description": "Add two numbers",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"},
                       "b": {"type": "number"}}, "required": ["a", "b"]}}}]
    try:
        r = chat([{"role": "user", "content": "Use the add tool to add 2 and 40."}],
                 tools=tools, max_tokens=200)
        tc = r.choices[0].message.tool_calls
        print("native tool calling:", "OK " + tc[0].function.arguments if tc else
              "no tool_calls returned; content=" + repr(r.choices[0].message.content)[:200])
    except Exception as e:
        print("native tool calling: FAILED", type(e).__name__, str(e)[:200])
    print("model under test:", settings.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

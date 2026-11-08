"""FastAPI backend + one-page UI.

  GET  /                 the page (static/index.html)
  GET  /api/health       {"status": "ok", ...} — no tokens spent
  POST /api/ask          {"question", "mode": "rag"|"agent"} -> AskResponse JSON
  POST /api/ask/stream   same body -> text/event-stream of agent steps, then the answer

Run:  uvicorn app.server:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .agent import AGENT_MODE, run_agent, run_agent_events
from .config import settings
from .rag import answer_rag
from .retrieval.index import get_index
from .schemas import AskRequest, AskResponse

log = logging.getLogger("tpg")
logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_index()  # load vectors + embedding model once, not on the first question
    except FileNotFoundError as e:
        log.error("%s", e)
    yield


app = FastAPI(title="Thai Province Guide", lifespan=lifespan)

# ── per-IP rate limit (20 requests / minute) ─────────────────────────────────────
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request, limit: int = 20, window: float = 60.0) -> None:
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "?"))
    now, q = time.monotonic(), _hits[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": "30"})
    q.append(now)


def upstream_errors(e: Exception) -> HTTPException:
    if isinstance(e, RateLimitError):
        return HTTPException(429, "Model is busy, try again shortly.", headers={"Retry-After": "10"})
    if isinstance(e, APITimeoutError):
        return HTTPException(504, "Model timed out.")
    if isinstance(e, (APIConnectionError, APIStatusError)):
        return HTTPException(502, "Model service error.")
    raise e


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": settings.model, "retriever": settings.retriever,
            "chunk_words": settings.chunk_words, "min_score": settings.min_score,
            "agent_mode": AGENT_MODE}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    rate_limit(request)
    fn = answer_rag if req.mode == "rag" else run_agent
    try:
        return await asyncio.to_thread(fn, req.question)
    except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
        log.warning("upstream error: %r", e)
        raise upstream_errors(e)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest, request: Request):
    rate_limit(request)

    async def gen():
        try:
            if req.mode == "rag":
                yield sse("step", {"n": 1, "tool": "retrieve", "args": {"query": req.question},
                                   "ok": True, "result": "retrieving + answering..."})
                resp = await asyncio.to_thread(answer_rag, req.question)
                yield sse("final", {"response": resp.model_dump(), "stop_reason": "answer"})
            else:
                it = run_agent_events(req.question)
                while True:
                    if await request.is_disconnected():
                        log.info("client disconnected; stopping agent")
                        return
                    ev = await asyncio.to_thread(next, it, None)
                    if ev is None:
                        break
                    yield sse(ev["type"], ev)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
            log.warning("upstream error mid-stream: %r", e)
            yield sse("error", {"message": "The model service failed; please try again."})
        except Exception as e:  # headers are already sent: report in-band
            log.exception("stream failed: %r", e)
            yield sse("error", {"message": "Something went wrong."})
        yield sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

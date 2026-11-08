"""API tests with the RAG/agent functions stubbed (no model, no index)."""

from fastapi.testclient import TestClient

from app import server
from app.schemas import AskResponse


def _fake(question, *a, **k):
    return AskResponse(question=question, mode="rag", answer="Ko Chang", abstained=False,
                       citations=[1], sources=[])


def test_health_and_ask(monkeypatch):
    monkeypatch.setattr(server, "get_index", lambda: None)
    monkeypatch.setattr(server, "answer_rag", _fake)
    with TestClient(server.app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        r = c.post("/api/ask", json={"question": "Largest island in Trat?", "mode": "rag"})
        assert r.status_code == 200 and r.json()["answer"] == "Ko Chang"
        assert c.post("/api/ask", json={"question": "x"}).status_code == 422  # too short
        assert c.post("/api/ask", json={"question": "hello", "mode": "sql"}).status_code == 422


def test_stream_emits_final_and_done(monkeypatch):
    monkeypatch.setattr(server, "get_index", lambda: None)
    monkeypatch.setattr(server, "answer_rag", _fake)
    server._hits.clear()
    with TestClient(server.app) as c:
        body = c.post("/api/ask/stream", json={"question": "Largest island?", "mode": "rag"}).text
    assert "event: final" in body and body.rstrip().endswith("data: {}")

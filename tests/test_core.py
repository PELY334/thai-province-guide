"""Offline tests: no network, no API key, no embedding model download.
A tiny bag-of-words embedder stands in for fastembed; the LLM is a scripted fake."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import agent, llm, rag, tools
from app.retrieval import index as index_mod
from app.retrieval.bm25 import BM25
from app.retrieval.chunking import chunk_document
from app.retrieval.index import Index
from app.schemas import CalcArgs, ModelAnswer, SearchArgs, Source

DOCS = [
    {"id": "lampang", "province": "Lampang", "url": "u1", "text":
     "Lampang is in northern Thailand.\n\n== Symbols ==\nThe provincial tree is Phayom. "
     "The provincial flower is the Kesorn tree.\n\n== Geography ==\nThe Wang River flows "
     "through the province.\n\n== References ==\nfoo"},
    {"id": "trat", "province": "Trat", "url": "u2", "text":
     "Trat is on the eastern coast.\n\n== Geography ==\nKo Chang is the largest island of "
     "the province."},
]


class FakeEmbedder:
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.strip(".,").encode()).hexdigest(), 16) % self.dim] += 1
        return v / max(np.linalg.norm(v), 1e-9)

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def idx(monkeypatch):
    ix = Index.build(DOCS, 12, FakeEmbedder())
    monkeypatch.setattr(index_mod, "get_index", lambda *a, **k: ix)
    # tests must not depend on the tuned defaults in .env / config
    monkeypatch.setattr(index_mod, "settings",
                        dataclasses.replace(index_mod.settings, min_score=0.0, top_k=5))
    return ix


# ── chunking / retrieval ─────────────────────────────────────────────────────────
def test_chunks_keep_sections_and_skip_references():
    chunks = chunk_document(DOCS[0], max_words=12)
    sections = {c.section for c in chunks}
    assert "Symbols" in sections and "References" not in sections
    assert all(c.id.startswith("lampang#") for c in chunks)
    assert not any("foo" in c.text for c in chunks)


def test_chunk_never_splits_inside_a_sentence():
    for c in chunk_document(DOCS[0], max_words=5):
        assert c.text.endswith(".")


def test_bm25_prefers_matching_doc():
    bm = BM25(["the wang river", "ko chang island"])
    s = bm.scores("Which island?")
    assert s[1] > s[0]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_modes_find_island(idx, mode):
    hits = idx.search("largest island Ko Chang", k=2, mode=mode)
    assert hits[0].chunk.province == "Trat"


def test_min_score_floor_drops_weak_hits(idx):
    assert idx.search("completely unrelated words zebra", k=3, min_score=0.0)
    assert idx.search("completely unrelated words zebra", k=3, min_score=0.9) == []


# ── structured output ───────────────────────────────────────────────────────────
def test_extract_json_handles_fences_and_braces_in_strings():
    txt = 'Sure!\n```json\n{"answer": "a {b} \\"c\\"", "citations": [1], "abstain": false}\n```'
    assert llm.extract_json(txt)["answer"] == 'a {b} "c"'


def test_model_answer_requires_citation_unless_abstaining():
    with pytest.raises(Exception):
        ModelAnswer(answer="x", citations=[], abstain=False)
    assert ModelAnswer(answer="", citations=[], abstain=True).abstain


def test_finalize_drops_invented_citations():
    src = [Source(n=1, province="P", section="S", url="u", text="t", cosine=0.5)]
    r = rag.finalize("q", "rag", ModelAnswer(answer="x", citations=[1, 7]), src)
    assert r.citations == [1] and not r.abstained
    r = rag.finalize("q", "rag", ModelAnswer(answer="x", citations=[7]), src)
    assert r.abstained  # only an invented citation -> refuse


def test_chat_structured_repairs_once(monkeypatch):
    replies = iter(["not json", '{"answer": "ok", "citations": [1], "abstain": false}'])
    monkeypatch.setattr(llm, "chat_text", lambda msgs, **k: next(replies))
    assert llm.chat_structured([], ModelAnswer).answer == "ok"


# ── tools ───────────────────────────────────────────────────────────────────────
def test_calculator_is_arithmetic_only():
    assert tools.safe_eval("20107 - 4506") == 15601
    for bad in ["__import__('os')", "2**1000", "abs(-1)", "x + 1"]:
        with pytest.raises(Exception):
            tools.safe_eval(bad)
    with pytest.raises(Exception):
        CalcArgs(expression="import os")


def test_run_tool_allow_list_and_validation(idx):
    ctx = tools.ToolContext()
    ok, out = tools.run_tool("delete_files", {}, ctx)
    assert not ok and "not allowed" in out
    ok, out = tools.run_tool("search_provinces", {"query": ""}, ctx)
    assert not ok and "invalid arguments" in out
    ok, out = tools.run_tool("search_provinces", '{"query": "Ko Chang island"}', ctx)
    assert ok and "[1]" in out and ctx.sources[0].n == 1
    ok, out = tools.run_tool("calculator", {"expression": "1,000 + 1"}, ctx)
    assert ok and out.endswith("= 1001")


def test_search_args_bounds():
    with pytest.raises(Exception):
        SearchArgs(query="x" * 500)


# ── agent loop with a scripted model ─────────────────────────────────────────────
def _tool_reply(name, args):
    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)),
                         model_dump=lambda: {"id": "c1", "type": "function", "function":
                                             {"name": name, "arguments": json.dumps(args)}})
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))])


def _text_reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


def test_agent_searches_then_answers(idx, monkeypatch):
    script = iter([_tool_reply("search_provinces", {"query": "Ko Chang island"}),
                   _text_reply('{"answer": "Ko Chang", "citations": [1], "abstain": false}')])
    monkeypatch.setattr(agent, "chat", lambda msgs, **k: next(script))
    monkeypatch.setattr(agent, "AGENT_MODE", "native")
    r = agent.run_agent("Largest island in Trat?")
    assert r.answer == "Ko Chang" and r.citations == [1] and len(r.steps) == 1


def test_agent_stops_at_max_steps(idx, monkeypatch):
    calls = {"n": 0}

    def fake_chat(msgs, **k):
        calls["n"] += 1
        if "tools" not in k:  # forced final turn
            return _text_reply('{"answer": "", "citations": [], "abstain": true}')
        return _tool_reply("search_provinces", {"query": f"q{calls['n']}"})

    monkeypatch.setattr(agent, "chat", fake_chat)
    monkeypatch.setattr(agent, "AGENT_MODE", "native")
    r = agent.run_agent("loop forever", max_steps=3)
    assert r.abstained and len(r.steps) == 3 and calls["n"] == 4


def test_agent_json_protocol(idx, monkeypatch):
    script = iter([_text_reply('{"tool": "calculator", "args": {"expression": "2+3"}}'),
                   _text_reply('{"tool": "search_provinces", "args": {"query": "Wang River"}}'),
                   _text_reply('{"answer": "5", "citations": [1], "abstain": false}')])
    monkeypatch.setattr(agent, "chat", lambda msgs, **k: next(script))
    monkeypatch.setattr(agent, "AGENT_MODE", "json")
    r = agent.run_agent("q")
    assert r.answer == "5" and [s["tool"] for s in r.steps] == ["calculator", "search_provinces"]

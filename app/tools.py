"""The agent's tools. Only names in TOOLS can run; arguments are validated with Pydantic
before execution; results are plain text the agent treats as untrusted data."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ValidationError

from .rag import format_sources, to_sources
from .retrieval.index import retrieve
from .schemas import CalcArgs, SearchArgs, Source


# ── calculator: a tiny arithmetic evaluator (no eval(), no names, no calls) ─────
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return ev(ast.parse(expr, mode="eval"))


# ── tool registry ───────────────────────────────────────────────────────────────
@dataclass
class ToolContext:
    """Per-question state: sources seen so far get stable global numbers [1], [2], ..."""
    sources: list[Source] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[BaseModel, ToolContext], str]

    def schema(self) -> dict:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": params}}


def _search(args: SearchArgs, ctx: ToolContext) -> str:
    hits = retrieve(args.query, k=args.k)
    if not hits:
        return "No relevant passages found (all below the relevance floor)."
    known = {(s.province, s.text): s.n for s in ctx.sources}
    new_hits = [h for h in hits if (h.chunk.province, h.chunk.text) not in known]
    start = len(ctx.sources) + 1
    ctx.sources.extend(to_sources(new_hits, start=start))
    repeated = [f"(already seen as [{known[(h.chunk.province, h.chunk.text)]}])"
                for h in hits if (h.chunk.province, h.chunk.text) in known]
    return (format_sources(new_hits, start=start) + ("\n" + " ".join(repeated) if repeated else "")
            ).strip()


def _calc(args: CalcArgs, ctx: ToolContext) -> str:
    value = safe_eval(args.expression)
    return f"{args.expression} = {round(value, 4):g}" if isinstance(value, float) else \
        f"{args.expression} = {value}"


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_provinces",
         "Search the Wikipedia articles about Thailand's 77 provinces. Returns numbered "
         "passages you can cite. Use one focused query per fact (e.g. 'Lamphun province area').",
         SearchArgs, _search),
    Tool("calculator",
         "Evaluate an arithmetic expression, e.g. '20107 - 4506' or '(3 + 4) * 2'.",
         CalcArgs, _calc),
]}


def run_tool(name: str, raw_args: dict | str, ctx: ToolContext) -> tuple[bool, str]:
    """Allow-list check -> argument validation -> run. Returns (ok, text-for-the-model).
    Errors are returned to the model as text, never raised into the agent loop."""
    tool = TOOLS.get(name)
    if tool is None:
        return False, f"Error: tool {name!r} is not allowed. Allowed: {', '.join(TOOLS)}."
    try:
        args = (tool.args_model.model_validate_json(raw_args) if isinstance(raw_args, str)
                else tool.args_model.model_validate(raw_args))
    except ValidationError as e:
        return False, f"Error: invalid arguments for {name}: {e.errors()[0]['msg']}"
    try:
        return True, tool.run(args, ctx)
    except Exception as e:  # tool failure is information for the model, not a crash
        return False, f"Error: {name} failed: {e}"

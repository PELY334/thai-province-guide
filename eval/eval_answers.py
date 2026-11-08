"""End-to-end answer evaluation against the labelled set (calls the LLM, temperature 0).

Systems compared:
  closed_book   the same model with no retrieval (baseline)
  rag_no_floor  single-shot RAG with min_score = 0 (naive default: never abstains early)
  rag           single-shot RAG with the tuned config from settings / .env
  agent         tool-calling agent (search_provinces + calculator)

Metrics (computed by a fixed string-matching rule, no LLM judge, so they are reproducible):
  accuracy         answerable questions whose answer contains a gold alias (all aliases
                   when all_required); an abstention counts as wrong
  abstain_ok       unanswerable questions where the system abstained / said it doesn't know
  false_abstain    answerable questions the system refused
  citation_prec    of the sources a system cited, the share that contain gold evidence
  overall          (correct answerable + correct abstentions) / all questions

Per-question outputs are saved to eval/results/answers_<system>.jsonl so every number in
the README can be audited.

    python eval/eval_answers.py                     # all systems
    python eval/eval_answers.py --systems rag agent
    CHUNK_WORDS=250 MIN_SCORE=0 python eval/eval_answers.py --systems rag --tag @w250
    python eval/eval_answers.py --rescore            # re-score saved answers after a label fix
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import run_agent  # noqa: E402
from app.config import EVAL_DIR, RESULTS_DIR, settings  # noqa: E402
from app.rag import answer_closed_book, answer_rag  # noqa: E402

DONT_KNOW = re.compile(r"\b(don'?t know|do not know|do(?:n'?t| not) have (?:the |any |specific |enough |information)|"
                       r"not sure|cannot|can't|unable to|no (?:reliable |specific )?information|"
                       r"not (?:able|available|aware)|i'?m sorry|real-time|up-to-date|as of my)\b", re.I)


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s)


def is_correct(answer: str, q: dict) -> bool:
    a = norm(answer)
    a_nocomma = re.sub(r"(?<=\d),(?=\d)", "", a)
    hits = [norm(x) in a or norm(x).replace(",", "") in a_nocomma for x in q["answers"]]
    return all(hits) if q["all_required"] else any(hits)


def run_system(name: str, q: dict) -> dict:
    t = time.time()
    if name == "closed_book":
        text = answer_closed_book(q["question"])
        out = {"answer": text, "abstained": bool(DONT_KNOW.search(text)), "sources": [],
               "citations": [], "steps": 0}
    else:
        if name == "agent":
            r = run_agent(q["question"])
        else:
            r = answer_rag(q["question"], min_score=0.0 if name == "rag_no_floor" else None)
        out = {"answer": r.answer, "abstained": r.abstained, "citations": r.citations,
               "sources": [s.model_dump() for s in r.sources], "steps": len(r.steps)}
    out["seconds"] = round(time.time() - t, 2)
    return out


def cited_ok(out: dict, q: dict) -> tuple[int, int]:
    cited = [s for s in out["sources"] if s["n"] in out["citations"]]
    good = sum(any(ev["text"] in s["text"] for ev in q["evidence"]) for s in cited)
    return good, len(cited)


def summarize(name: str, rows: list[dict]) -> dict:
    ans = [r for r in rows if r["q"]["evidence"]]
    una = [r for r in rows if not r["q"]["evidence"]]
    correct = sum(r["correct"] for r in ans)
    abst_ok = sum(r["out"]["abstained"] for r in una)
    good, cited = map(sum, zip(*[cited_ok(r["out"], r["q"]) for r in rows])) if rows else (0, 0)
    by_type = {}
    for t in sorted({r["q"]["type"] for r in ans}):
        sub = [r for r in ans if r["q"]["type"] == t]
        by_type[t] = f"{sum(r['correct'] for r in sub)}/{len(sub)}"
    return {
        "system": name, "n": len(rows),
        "accuracy": correct / len(ans), "correct": f"{correct}/{len(ans)}",
        "abstain_ok": abst_ok / len(una), "abstain_ok_n": f"{abst_ok}/{len(una)}",
        "false_abstain": sum(r["out"]["abstained"] for r in ans) / len(ans),
        "citation_prec": (good / cited) if cited else None,
        "overall": (correct + abst_ok) / len(rows),
        "by_type": by_type,
        "avg_seconds": sum(r["out"]["seconds"] for r in rows) / len(rows),
        "avg_tool_steps": sum(r["out"]["steps"] for r in rows) / len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["closed_book", "rag_no_floor", "rag", "agent"])
    ap.add_argument("--rescore", action="store_true",
                    help="don't call the model: re-score the saved answers_<system>.jsonl "
                    "against the current labels (use after fixing a gold label)")
    ap.add_argument("--tag", default="", help="suffix for result names, e.g. '@w250' when "
                    "running with CHUNK_WORDS=250 in the environment")
    args = ap.parse_args()
    questions = [json.loads(line) for line in (EVAL_DIR / "questions.jsonl").read_text("utf-8").splitlines()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {"model": settings.model, "embed_model": settings.embed_model,
              "retriever": settings.retriever, "chunk_words": settings.chunk_words,
              "top_k": settings.top_k, "min_score": settings.min_score}
    print("config:", config)
    summaries = []
    if args.rescore:
        old = json.loads((RESULTS_DIR / "answers_summary.json").read_text())
        by_id = {q["id"]: q for q in questions}
        for name in args.systems:
            path = RESULTS_DIR / f"answers_{name}{args.tag}.jsonl"
            rows = []
            for line in path.read_text("utf-8").splitlines():
                out = json.loads(line)
                q = by_id[out["id"]]
                if name == "closed_book":  # its abstention is detected from the text
                    out["abstained"] = bool(DONT_KNOW.search(out["answer"]))
                out["correct"] = bool(q["evidence"]) and not out["abstained"] and is_correct(out["answer"], q)
                rows.append({"q": q, "out": out, "correct": out["correct"]})
            path.write_text("".join(json.dumps(r["out"], ensure_ascii=False) + "\n" for r in rows), "utf-8")
            s = summarize(name + args.tag, rows)
            s["config"] = old.get(name + args.tag, {}).get("config", config)
            summaries.append(s)
            print(name + args.tag, s["correct"], s["abstain_ok_n"])
        args.systems = []
    for name in args.systems:
        rows = []
        with open(RESULTS_DIR / f"answers_{name}{args.tag}.jsonl", "w", encoding="utf-8") as f:
            for q in questions:
                try:
                    out = run_system(name, q)
                except Exception as e:  # keep going; an error counts as a wrong answer
                    out = {"answer": f"ERROR: {type(e).__name__}: {e}", "abstained": False,
                           "citations": [], "sources": [], "steps": 0, "seconds": 0.0}
                correct = bool(q["evidence"]) and not out["abstained"] and is_correct(out["answer"], q)
                rows.append({"q": q, "out": out, "correct": correct})
                f.write(json.dumps({"id": q["id"], "type": q["type"], "question": q["question"],
                                    "correct": correct, **out}, ensure_ascii=False) + "\n")
                mark = "ok " if correct or (not q["evidence"] and out["abstained"]) else "XX "
                print(f"[{name}] {mark}{q['id']} {out['answer'][:90]!r}")
        s = summarize(name + args.tag, rows)
        s["config"] = config
        summaries.append(s)
        print(json.dumps(s, indent=1))
    path = RESULTS_DIR / "answers_summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({s["system"]: s for s in summaries})
    path.write_text(json.dumps(old, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

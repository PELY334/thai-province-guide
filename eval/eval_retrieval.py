"""Retrieval evaluation on the labelled set (no LLM calls, fully deterministic).

A chunk is *relevant* to an evidence item if it comes from the gold article and contains
the gold evidence string (so the label doesn't depend on chunk size). Metrics at k=5:

  hit@1     top-1 chunk is relevant
  recall@5  fraction of the question's evidence items found in the top 5 (multi-hop
            questions have two)
  MRR       1 / rank of the first relevant chunk (0 if none in the top 5)

Then the min_score floor is tuned on a separate DEV set (data/eval/dev_questions.jsonl:
12 answerable + 10 unanswerable questions that are not in the test set) and the chosen
floor is reported on the TEST set. For a floor t, an answerable question "survives" if a
relevant chunk is still in its filtered top 5; an unanswerable one is correctly "rejected"
if nothing clears the floor. We pick the t with the best balanced score on dev.

    python eval/eval_retrieval.py                       # every index found in data/index
    python eval/eval_retrieval.py --sweep-config bge-m3-w120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import EVAL_DIR, INDEX_DIR, RESULTS_DIR  # noqa: E402
from app.retrieval.embed import get_embedder  # noqa: E402
from app.retrieval.index import Index  # noqa: E402

K = 5
MODES = ["bm25", "dense", "hybrid"]


def load_questions(name: str = "questions.jsonl") -> list[dict]:
    return [json.loads(line) for line in (EVAL_DIR / name).read_text("utf-8").splitlines()]


def relevant(chunk, ev) -> bool:
    return chunk.doc_id == ev["doc"] and ev["text"] in chunk.text


def score_question(hits, q) -> dict:
    found = [any(relevant(h.chunk, ev) for h in hits) for ev in q["evidence"]]
    first = next((i + 1 for i, h in enumerate(hits)
                  if any(relevant(h.chunk, ev) for ev in q["evidence"])), None)
    return {"hit1": float(bool(hits) and any(relevant(hits[0].chunk, ev) for ev in q["evidence"])),
            "recall": sum(found) / len(found), "rr": 1 / first if first else 0.0}


def evaluate(index: Index, questions: list[dict], mode: str, min_score: float = 0.0) -> dict:
    rows = [score_question(index.search(q["question"], K, mode, min_score), q)
            for q in questions if q["evidence"]]
    n = len(rows)
    return {"n": n, "hit@1": sum(r["hit1"] for r in rows) / n,
            "recall@5": sum(r["recall"] for r in rows) / n, "MRR": sum(r["rr"] for r in rows) / n}


def sweep(index: Index, questions: list[dict], mode: str) -> list[dict]:
    """Search once per question without a floor, then apply each candidate floor to the
    ranked list (identical to searching with min_score=t, because the floor only removes
    hits and never reorders them)."""
    ranked = {q["id"]: index.search(q["question"], 200, mode, 0.0) for q in questions}
    ans = [q for q in questions if q["evidence"]]
    una = [q for q in questions if not q["evidence"]]

    def top_k(qid, t):
        return [h for h in ranked[qid] if h.cosine >= t][:K]

    out = []
    for t in [0.0] + [round(0.30 + 0.01 * i, 2) for i in range(61)]:
        kept = sum(score_question(top_k(q["id"], t), q)["recall"] > 0 for q in ans)
        rejected = sum(not top_k(q["id"], t) for q in una)
        out.append({"min_score": t, "answerable_kept": kept / len(ans),
                    "unanswerable_rejected": rejected / len(una),
                    "balanced": (kept / len(ans) + rejected / len(una)) / 2})
    return out


def top_cosines(index: Index, questions: list[dict]) -> list[dict]:
    """Top-1 dense cosine per question: the raw material for choosing min_score."""
    return [{"id": q["id"], "type": q["type"],
             "top_cosine": round(float((index.vectors @ index.embedder.embed_query(q["question"])).max()), 4)}
            for q in questions]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config", help="index dir name for the min_score sweep "
                    "(default: the best config by recall@5, then MRR)")
    ap.add_argument("--sweep-mode", help="retrieval mode for the sweep (default: same rule)")
    args = ap.parse_args()
    questions = load_questions()
    configs = sorted(p for p in INDEX_DIR.iterdir() if (p / "vectors.npy").exists())
    table, indexes = [], {}
    for path in configs:
        meta = json.loads((path / "meta.json").read_text())
        index = Index.load(path, get_embedder(meta["embed_model"]))
        indexes[path.name] = index
        for mode in MODES:
            if mode == "bm25" and table and any(r["config"].endswith(f"w{meta['chunk_words']}")
                                                and r["mode"] == "bm25" for r in table):
                continue  # BM25 doesn't depend on the embedding model
            r = {"config": path.name, "embed_model": meta["embed_model"],
                 "chunk_words": meta["chunk_words"], "n_chunks": meta["n_chunks"], "mode": mode,
                 **evaluate(index, questions, mode)}
            table.append(r)
            print(f"{path.name:28} {mode:7} hit@1={r['hit@1']:.3f} recall@5={r['recall@5']:.3f} "
                  f"MRR={r['MRR']:.3f}")

    top = max((r for r in table if r["mode"] != "bm25"), key=lambda r: (r["recall@5"], r["MRR"]))
    best, mode = args.sweep_config or top["config"], args.sweep_mode or top["mode"]
    sw = sweep(indexes[best], load_questions("dev_questions.jsonl"), mode)
    best_t = max(sw, key=lambda r: (r["balanced"], r["answerable_kept"], -r["min_score"]))
    test_sw = {r["min_score"]: r for r in sweep(indexes[best], questions, mode)}
    test_at = {"no_floor": test_sw[0.0], "tuned": test_sw[best_t["min_score"]]}
    print(f"\nmin_score sweep on DEV, {best} ({mode}); every 5th + the best:")
    for r in sw:
        if r is best_t or round(r["min_score"] * 100) % 5 == 0:
            print(f"  t={r['min_score']:.2f} kept={r['answerable_kept']:.3f} "
                  f"rejected={r['unanswerable_rejected']:.3f} balanced={r['balanced']:.3f}"
                  f"{'   <- best' if r is best_t else ''}")
    for name, r in test_at.items():
        print(f"TEST {name:8} t={r['min_score']:.2f} kept={r['answerable_kept']:.3f} "
              f"rejected={r['unanswerable_rejected']:.3f} balanced={r['balanced']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"k": K, "table": table, "sweep_config": best, "sweep_mode": mode,
           "best_min_score_dev": best_t, "test_at_min_score": test_at, "dev_sweep": sw,
           "test_sweep": list(test_sw.values()),
           "top_cosines": top_cosines(indexes[best], questions)}
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULTS_DIR / 'retrieval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

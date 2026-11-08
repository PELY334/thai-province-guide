"""Choose MIN_SCORE end-to-end on the DEV set (never on the test set).

The retrieval-only sweep (eval_retrieval.py) shows that cosine alone cannot separate
answerable from unanswerable questions cleanly, and that the LLM can also abstain by
itself. So the floor is tuned on what the user sees: for each candidate floor we run the
real RAG pipeline on the 22 dev questions and score

  overall  = (correct answers + correct abstentions) / n
  skipped  = share of questions answered "I can't answer" without any LLM call

and pick the best overall; ties go to the HIGHER floor (same quality, fewer LLM calls,
and an extra guard against off-topic prompts).

    python eval/tune_min_score.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import EVAL_DIR, RESULTS_DIR  # noqa: E402
from app.rag import answer_rag  # noqa: E402
from eval_answers import is_correct  # noqa: E402

CANDIDATES = [0.0, 0.60, 0.65, 0.68, 0.70, 0.72, 0.73, 0.75]


def main() -> int:
    dev = [json.loads(line) for line in (EVAL_DIR / "dev_questions.jsonl").read_text("utf-8").splitlines()]
    rows = []
    for t in CANDIDATES:
        good = skipped = 0
        misses = []
        for q in dev:
            r = answer_rag(q["question"], min_score=t)
            skipped += r.abstained and not r.sources
            ok = (r.abstained if not q["evidence"]
                  else (not r.abstained and is_correct(r.answer, q)))
            good += ok
            if not ok:
                misses.append(q["id"])
        row = {"min_score": t, "overall": good / len(dev), "skipped_llm": skipped / len(dev),
               "misses": misses}
        rows.append(row)
        print(f"t={t:.2f} overall={row['overall']:.3f} skipped_llm={row['skipped_llm']:.3f} misses={misses}")
    best = max(rows, key=lambda r: (round(r["overall"], 6), r["min_score"]))
    print(f"chosen MIN_SCORE = {best['min_score']}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "min_score_tuning.json").write_text(json.dumps({"dev_n": len(dev), "rows": rows,
                                                                   "chosen": best}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

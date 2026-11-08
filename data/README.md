# Data

## `raw/` — the corpus (committed)

One JSON file per Thai province (77 files): the plain-text English Wikipedia article,
fetched with `python scripts/fetch_corpus.py` through the MediaWiki API (TextExtracts).
Each file records `pageid`, `revid` and a permanent `url` (`index.php?oldid=<revid>`), so
the exact revision used for the evaluation can always be checked.

Text © Wikipedia contributors, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Infoboxes and tables are
not included in TextExtracts output, so facts that appear only in an infobox (for example
some population figures) are not in the corpus. That limitation is deliberate and shows up
in the evaluation.

## `eval/` — the labelled test set

`questions.jsonl`, written by hand from the corpus. See the README section "Evaluation".

## `index/` — built artefacts (git-ignored)

Created by `python scripts/build_index.py`; rebuild any time.

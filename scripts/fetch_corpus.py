"""Fetch the corpus: the English Wikipedia article for each of Thailand's 77 provinces.

Writes one JSON file per article to data/raw/ with the page id and revision id, so the
exact text used for the evaluation is pinned and committed (Wikipedia text is CC BY-SA 4.0;
see data/README.md). Re-running skips files that already exist unless --force is given.

    python scripts/fetch_corpus.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ThaiProvinceGuide/0.1 (KMITL LLM-apps course capstone project; non-commercial)"

PROVINCES = [
    "Amnat Charoen", "Ang Thong", "Bangkok", "Bueng Kan", "Buriram", "Chachoengsao",
    "Chai Nat", "Chaiyaphum", "Chanthaburi", "Chiang Mai", "Chiang Rai", "Chonburi",
    "Chumphon", "Kalasin", "Kamphaeng Phet", "Kanchanaburi", "Khon Kaen", "Krabi",
    "Lampang", "Lamphun", "Loei", "Lopburi", "Mae Hong Son", "Maha Sarakham", "Mukdahan",
    "Nakhon Nayok", "Nakhon Pathom", "Nakhon Phanom", "Nakhon Ratchasima", "Nakhon Sawan",
    "Nakhon Si Thammarat", "Nan", "Narathiwat", "Nong Bua Lamphu", "Nong Khai",
    "Nonthaburi", "Pathum Thani", "Pattani", "Phang Nga", "Phatthalung", "Phayao",
    "Phetchabun", "Phetchaburi", "Phichit", "Phitsanulok", "Phra Nakhon Si Ayutthaya",
    "Phrae", "Phuket", "Prachinburi", "Prachuap Khiri Khan", "Ranong", "Ratchaburi",
    "Rayong", "Roi Et", "Sa Kaeo", "Sakon Nakhon", "Samut Prakan", "Samut Sakhon",
    "Samut Songkhram", "Saraburi", "Satun", "Sing Buri", "Sisaket", "Songkhla",
    "Sukhothai", "Suphan Buri", "Surat Thani", "Surin", "Tak", "Trang", "Trat",
    "Ubon Ratchathani", "Udon Thani", "Uthai Thani", "Uttaradit", "Yala", "Yasothon",
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def wiki_title(name: str) -> str:
    return name if name == "Bangkok" else f"{name} province"


def fetch(title: str) -> dict:
    params = {
        "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
        "prop": "extracts|revisions", "explaintext": "1", "rvprop": "ids|timestamp",
        "titles": title,
    }
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": USER_AGENT})
    for attempt in range(6):  # Wikipedia answers 429 when we go too fast: back off
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            print(f"  429 from Wikipedia, waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    page = data["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"no Wikipedia page for {title!r}")
    rev = page["revisions"][0]
    return {
        "title": page["title"], "pageid": page["pageid"], "revid": rev["revid"],
        "timestamp": rev["timestamp"],
        "url": f"https://en.wikipedia.org/w/index.php?oldid={rev['revid']}",
        "text": page["extract"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, name in enumerate(PROVINCES, 1):
        out = RAW_DIR / f"{slug(name)}.json"
        if out.exists() and not args.force:
            continue
        try:
            doc = fetch(wiki_title(name))
        except Exception as e:  # network errors, missing pages
            print(f"[{i:2}/77] FAILED {name}: {e}", file=sys.stderr)
            failed.append(name)
            continue
        doc = {"id": slug(name), "province": name, **doc}
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i:2}/77] {doc['title']}: {len(doc['text'].split())} words")
        time.sleep(2.0)  # be polite to the API
    print(f"done: {len(list(RAW_DIR.glob('*.json')))} files in {RAW_DIR}, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

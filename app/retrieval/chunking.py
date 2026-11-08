"""Split a Wikipedia article into retrieval chunks.

Articles come from TextExtracts as plain text with `== Section ==` headings. We keep each
chunk inside one section, pack whole paragraphs (or, for long paragraphs, whole sentences)
up to ~`max_words` words, and never cut a sentence in half. Every chunk carries its
province and section so a citation can say exactly where the evidence came from.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

HEADING = re.compile(r"^\s*(={2,})\s*(.+?)\s*\1\s*$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Sections that are lists of links, not content.
SKIP_SECTIONS = {"see also", "references", "external links", "notes", "further reading",
                 "bibliography", "sources", "citations"}


@dataclass
class Chunk:
    id: str
    doc_id: str
    province: str
    section: str
    text: str
    url: str

    def embed_text(self) -> str:
        """What gets embedded / BM25-indexed: the text plus where it lives."""
        return f"{self.province} province. {self.section}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _sections(text: str):
    section, buf = "Overview", []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            if buf:
                yield section, buf
            section, buf = m.group(2).strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        yield section, buf


def _pieces(paragraphs: list[str], max_words: int):
    """Yield paragraph-or-sentence pieces no longer than max_words (unless one sentence is)."""
    for para in paragraphs:
        if len(para.split()) <= max_words:
            yield para
        else:
            yield from SENTENCE.split(para)


def chunk_document(doc: dict, max_words: int = 120) -> list[Chunk]:
    packed: list[tuple[str, str]] = []  # (section, text)
    for section, paragraphs in _sections(doc["text"]):
        if section.lower() in SKIP_SECTIONS:
            continue
        cur: list[str] = []
        n = 0
        for piece in _pieces(paragraphs, max_words):
            w = len(piece.split())
            if cur and n + w > max_words:
                packed.append((section, " ".join(cur)))
                cur, n = [], 0
            cur.append(piece)
            n += w
        if cur:
            packed.append((section, " ".join(cur)))
    return [Chunk(id=f"{doc['id']}#{i}", doc_id=doc["id"], province=doc["province"],
                  section=section, text=text, url=doc["url"])
            for i, (section, text) in enumerate(packed)]

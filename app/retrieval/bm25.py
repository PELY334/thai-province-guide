"""A small Okapi BM25 (no extra dependency). Used as a baseline and inside hybrid search."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be by for from has have in is it its of on or that the
this to was were which with what who whom whose where when how does did do""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avgdl = float(self.lens.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        terms = [t for t in tokenize(query) if t in self.idf]
        if not terms:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lens / max(self.avgdl, 1e-9))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out

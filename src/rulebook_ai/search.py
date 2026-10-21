"""Search over rulebook chunks: BM25 keywords, local embeddings (nomic-embed-text via Ollama), and a hybrid that
fuses the two rankings with reciprocal rank fusion. Results are chunks; citations are their parent sections."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .ingest import Section, section_of
from .llm import embed

STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "on",
        "in",
        "at",
        "to",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "by",
        "for",
        "from",
        "it",
        "its",
        "this",
        "that",
        "there",
        "their",
        "his",
        "he",
        "she",
        "her",
        "they",
        "them",
        "when",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "does",
        "do",
        "can",
        "may",
        "shall",
        "will",
        "if",
        "than",
        "then",
        "any",
        "each",
    ]
)


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        if len(w) > 4 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b, self.avg = k1, b, float(self.len.mean())

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for w in set(tokens(query)):
            if w not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return s


class Index:
    def __init__(self, sections: list[Section], cache: str | Path | None = None, use_embeddings: bool = True):
        self.sections = sections
        self.by_id = {s.id: s for s in sections}
        self.bm25 = BM25([f"{s.rule_title} {s.title} {s.text}" for s in sections])
        self.vectors: np.ndarray | None = None
        if use_embeddings:
            self.vectors = self._embed_docs(cache)

    def _embed_docs(self, cache: str | Path | None) -> np.ndarray:
        texts = [f"search_document: {s.citation}. {s.text}" for s in self.sections]
        key = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        if cache and Path(cache).exists():
            data = json.loads(Path(cache).read_text())
            if data.get("key") == key:
                return np.array(data["vectors"], dtype=np.float32)
        vecs = np.array(embed(texts), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        if cache:
            Path(cache).write_text(json.dumps({"key": key, "vectors": vecs.round(5).tolist()}))
        return vecs

    def _rank(self, scores: np.ndarray) -> np.ndarray:
        return np.argsort(-scores, kind="stable")

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[Section]:
        if method == "bm25" or self.vectors is None:
            order = self._rank(self.bm25.scores(query))
        else:
            q = np.array(embed([f"search_query: {query}"])[0], dtype=np.float32)
            dense = self.vectors @ (q / np.linalg.norm(q))
            if method == "embedding":
                order = self._rank(dense)
            else:  # reciprocal rank fusion: robust to the two scores living on different scales
                fused = np.zeros(len(self.sections))
                for ranking in (self._rank(self.bm25.scores(query)), self._rank(dense)):
                    fused[ranking] += 1.0 / (60 + np.arange(1, len(ranking) + 1))
                order = self._rank(fused)
        return [self.sections[i] for i in order[:k]]

    def section_text(self, section_id: str) -> list[Section]:
        """Every chunk of a section, in order (accepts 'R7-IV' or a chunk id like 'R7-IV#2')."""
        sid = section_of(section_id)
        return [s for s in self.sections if section_of(s.id) == sid]

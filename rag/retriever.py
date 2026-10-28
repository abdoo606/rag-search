"""Chunk, index, retrieve, answer - the RAG loop without any external service.

Embeddings are TF-IDF vectors: weaker than a neural model, but inspectable,
offline and deterministic, which makes the retrieval quality testable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its of on or our should so than that the their them then
    there these they this to too up was we were what when where which who why will
    with would you your
    """.split()
)


def tokenise(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def terms(text: str) -> list[str]:
    """Content terms only - stopwords carry no retrieval signal and wreck the cosine."""
    return [t for t in tokenise(text) if t not in STOPWORDS]


@dataclass(slots=True, frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str

    @property
    def label(self) -> str:
        return f"{self.doc_id}#{self.index}"


def chunk_document(text: str, doc_id: str, *, size: int = 60, overlap: int = 15) -> list[Chunk]:
    """Word-window chunking with overlap so answers are not cut at boundaries."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > overlap >= 0")
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        window = words[start : start + size]
        chunks.append(Chunk(doc_id, len(chunks), " ".join(window)))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


@dataclass(slots=True)
class Index:
    chunks: list[Chunk] = field(default_factory=list)
    doc_freq: Counter[str] = field(default_factory=Counter)
    vectors: list[dict[str, float]] = field(default_factory=list)
    _norms: list[float] = field(default_factory=list)

    def add(self, chunks: list[Chunk]) -> "Index":
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.doc_freq.update(set(terms(chunk.text)))
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        total = len(self.chunks)
        self.vectors = []
        self._norms = []
        for chunk in self.chunks:
            tokens = terms(chunk.text)
            counts = Counter(tokens)
            vector = {
                term: (1 + math.log(freq)) * (math.log((total + 1) / (self.doc_freq[term] + 1)) + 1)
                for term, freq in counts.items()
            }
            norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
            self.vectors.append(vector)
            self._norms.append(norm)

    def search(self, query: str, *, k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks or k <= 0:
            return []
        q_tokens = terms(query)
        q_counts = Counter(q_tokens)
        q_vec = {
            term: (1 + math.log(freq)) * (math.log((len(self.chunks) + 1) / (self.doc_freq[term] + 1)) + 1)
            for term, freq in q_counts.items()
            if term in self.doc_freq
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scored = []
        for i, vector in enumerate(self.vectors):
            dot = sum(weight * vector.get(term, 0.0) for term, weight in q_vec.items())
            if dot:
                scored.append((self.chunks[i], dot / (q_norm * self._norms[i])))
        return sorted(scored, key=lambda pair: -pair[1])[:k]


@dataclass(slots=True, frozen=True)
class Answer:
    text: str
    citations: list[str]
    score: float
    confident: bool


def answer(index: Index, question: str, *, k: int = 3, min_score: float = 0.12) -> Answer:
    """Extractive answer: the best sentence from the best chunk, with sources attached.

    Saying "I don't know" below a score threshold is a feature, not a limitation.
    """
    hits = index.search(question, k=k)
    if not hits or hits[0][1] < min_score:
        return Answer("I could not find this in the provided documents.", [], 0.0, False)

    best_chunk, best_score = hits[0]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best_chunk.text) if s.strip()]
    q_tokens = set(terms(question))
    best_sentence = max(
        sentences or [best_chunk.text],
        key=lambda s: len(q_tokens & set(terms(s))),
    )
    citations = [f"{chunk.label} ({score:.2f})" for chunk, score in hits]
    return Answer(best_sentence, citations, round(best_score, 3), True)

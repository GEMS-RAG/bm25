"""Original BM25 definitions from the saved-run experiment harness.

Definitions below are extracted without modifying their behavior; see
PROVENANCE.json for source revision, paths, line spans, and hashes.
"""
from __future__ import annotations

import collections
import math
import re
from abc import ABC, abstractmethod

from .types import Evidence, QAItem, RetrievalResult

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")


SECTION_RE = re.compile(r"\b([1-9][A-Z]\.[0-9]{2})\b", re.IGNORECASE)


FIGURE_RE = re.compile(r"\b(?:Figure|Table)\s+([1-9][A-Z]-[0-9]+[A-Z]?)\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def evidence_text_from_chunk(chunk: dict, max_chars: int = 1600) -> str:
    header = (
        f"Section {chunk.get('section_id')} {chunk.get('content_type')} "
        f"{chunk.get('ordinal')} - {chunk.get('section_title')} "
        f"(p.{chunk.get('page_printed')}, {chunk.get('part')})"
    )
    body = (chunk.get("text") or "").strip()
    if len(body) > max_chars:
        body = body[: max_chars - 3].rstrip() + "..."
    return f"{header}\n{body}"


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, item: QAItem) -> RetrievalResult:
        raise NotImplementedError


class BM25Retriever(Retriever):
    def __init__(self, name: str, chunks: list[dict], top_k: int = 6, *, graph_boost: bool = False) -> None:
        self.name = name
        self.chunks = chunks
        self.top_k = top_k
        self.graph_boost = graph_boost
        self.doc_tokens = [tokenize(_chunk_search_text(chunk)) for chunk in chunks]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: collections.Counter[str] = collections.Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n_docs = max(len(self.doc_tokens), 1)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def retrieve(self, item: QAItem) -> RetrievalResult:
        query_tokens = tokenize(item.question)
        explicit_sections = {m.upper() for m in SECTION_RE.findall(item.question)}
        explicit_figures = {m.upper() for m in FIGURE_RE.findall(item.question)}
        scores: list[tuple[float, int]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            score = self._score(query_tokens, tokens, self.doc_len[idx])
            chunk = self.chunks[idx]
            section_id = str(chunk.get("section_id", "")).upper()
            if self.graph_boost and section_id in explicit_sections:
                score += 5.0
            if self.graph_boost and explicit_figures:
                refs = {str(ref).upper() for ref in chunk.get("figure_refs", []) + chunk.get("table_refs", [])}
                if refs & explicit_figures:
                    score += 2.0
            if score > 0:
                scores.append((score, idx))
        scores.sort(reverse=True)
        evidence = [
            _chunk_to_evidence(self.chunks[idx], score)
            for score, idx in scores[: self.top_k]
        ]
        return RetrievalResult(
            adapter=self.name,
            query=item.question,
            evidence=evidence,
            debug={
                "query_tokens": query_tokens,
                "explicit_sections": sorted(explicit_sections),
                "explicit_figures": sorted(explicit_figures),
            },
        )

    def _score(self, query_tokens: list[str], doc_tokens: list[str], doc_len: int) -> float:
        counts = collections.Counter(doc_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * doc_len / max(self.avgdl, 1e-9))
            score += self.idf.get(term, 0.0) * (tf * (k1 + 1)) / denom
        return score


def _chunk_search_text(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(key) or "")
        for key in ["part", "chapter", "section_id", "section_title", "content_type", "text"]
    )


def _chunk_to_evidence(chunk: dict, score: float) -> Evidence:
    chunk_id = str(chunk.get("chunk_id"))
    return Evidence(
        evidence_id=chunk_id,
        kind="chunk",
        text=evidence_text_from_chunk(chunk),
        metadata={
            "chunk_id": chunk_id,
            "section_id": chunk.get("section_id"),
            "section_title": chunk.get("section_title"),
            "content_type": chunk.get("content_type"),
            "ordinal": chunk.get("ordinal"),
            "page_printed": chunk.get("page_printed"),
            "part": chunk.get("part"),
            "figure_refs": chunk.get("figure_refs", []),
            "table_refs": chunk.get("table_refs", []),
        },
        score=float(score),
    )

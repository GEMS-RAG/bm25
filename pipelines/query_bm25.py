#!/usr/bin/env python3
"""Run the original saved-study BM25 implementation over the shared corpus."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comparison_support.bm25 import BM25Retriever
from comparison_support.data import canonicalize_chunks
from comparison_support.types import QAItem

DEFAULT_CORPUS = ROOT / "indexes" / "corpus" / "chunks.jsonl"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def load_documents(path: Path) -> list[dict[str, Any]]:
    """Recover original chunk fields from raw chunks or the exact study export."""
    chunks = []
    for row in read_jsonl(path):
        if row.get("chunk_id"):
            chunks.append(row)
            continue
        if not row.get("doc_id") or not isinstance(row.get("metadata"), dict):
            raise ValueError("expected raw MUTCD chunks or the study's canonical corpus export")
        metadata = row["metadata"]
        prefix = (
            f"{row.get('title')}\n"
            f"Part: {metadata.get('part')}\n"
            f"Chapter: {metadata.get('chapter')}\n"
            f"Page: {metadata.get('page_printed')}\n\n"
        )
        text = row.get("text")
        if not isinstance(text, str) or not text.startswith(prefix):
            raise ValueError(f"{row['doc_id']}: unrecognized study export header")
        chunks.append({**metadata, "chunk_id": row["doc_id"], "text": text[len(prefix):]})
    documents, _ = canonicalize_chunks(chunks)
    if not documents:
        raise ValueError("the BM25 corpus is empty")
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    query = subparsers.add_parser("query")
    query.add_argument("--question", required=True)
    query.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if getattr(args, "top_k", 1) < 1:
        parser.error("--top-k must be positive")
    if not args.corpus.is_file():
        print(json.dumps({"runnable": False, "method": "bm25", "error": "corpus_not_found"}))
        return 2
    documents = load_documents(args.corpus)
    if args.command == "check":
        print(json.dumps({"runnable": True, "method": "bm25", "documents": len(documents)}))
        return 0
    retriever = BM25Retriever("bm25", documents, top_k=args.top_k)
    item = QAItem(
        qa_id="query", question=args.question, question_type=None,
        expected_refusal=False, gold_answer={}, references=[],
    )
    result = retriever.retrieve(item)
    contexts = [
        {"name": evidence.evidence_id, **asdict(evidence)}
        for evidence in result.evidence
    ]
    print(json.dumps({
        "method": "bm25", "question": args.question, "top_k": args.top_k,
        "contexts": contexts, "debug": result.debug,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

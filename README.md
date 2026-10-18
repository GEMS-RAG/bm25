# BM25 for the MUTCD RAG comparison

This repository contains the dependency-free BM25 retrieval implementation used in the GEMS-RAG MUTCD comparison. The comparison runner, benchmark, canonical corpus, and shared study artifacts are in [GEMS-RAG/rag](https://github.com/GEMS-RAG/rag).

The implementation indexes each chunk's part, chapter, section identifier, section title, content type, and original text. It uses `k1=1.5` and `b=0.75`, returns stable canonical chunk identifiers, and retains the study's descending corpus-index tie break. The original `BM25Retriever` and its supporting definitions are extracted unchanged from the saved-run experiment harness. [PROVENANCE.json](PROVENANCE.json) records the source revision, definition spans, and SHA-256 hashes.

The standalone CLI reverses the shared corpus export's display header before calling the original retriever. This preserves the saved answer-run rankings and evidence text. The older bundled `rag/pipelines/query_bm25.py` instead scores the display-formatted export directly; its results differ from the saved answer-run baseline.

## Run

Python 3.11 or newer is sufficient; there are no third-party runtime dependencies. From this repository, clone the study hub as a sibling directory if it is not already present:

```bash
git clone https://github.com/GEMS-RAG/rag.git ../rag
python3 pipelines/query_bm25.py --corpus ../rag/indexes/corpus/chunks.jsonl check
python3 pipelines/query_bm25.py --corpus ../rag/indexes/corpus/chunks.jsonl query \
  --question "What shape is a STOP sign?" --top-k 10
```

The canonical study corpus has 5,705 chunks and is included in the hub checkout; BM25 does not require the prebuilt-index download. Use `--corpus` to select the exact study export or original raw MUTCD chunk JSONL. The CLI rejects exported rows whose display header does not match the documented study format. Query output is a JSON object containing ranked `contexts`, including the original evidence text, scores, and metadata.

The public method ID is `bm25`. The original saved answer comparisons retrieved six chunks; the published retrieval-scoring workflow uses ten. Set `--top-k` to the depth of the experiment being reproduced.

## Citation and attribution

Cite Robertson and Zaragoza (2009), [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019), for the retrieval method. Cite this repository for the study-specific implementation. [CITATION.cff](CITATION.cff) and [references.bib](references.bib) provide citation metadata.

Source attribution and licensing information are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository contains the study implementation of BM25; it does not vendor an external BM25 package.

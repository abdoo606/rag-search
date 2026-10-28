# 27 · rag

Retrieval-augmented answering over your own documents, with citations.

## Run

```bash
from rag.retriever import Index, answer
```

## Test

```bash
pytest projects/27-rag-search -q
```

## What it demonstrates

- Word-window chunking with overlap so answers are not cut in half
- TF-IDF vectors with stopword removal; scores are inspectable
- Extractive answers that cite `doc#chunk (score)`
- A confidence floor so the system says 'I don't know' instead of hallucinating

## Repository
- GitHub: https://github.com/{owner}/rag-search
- Python 3.11+ · tests: `pytest` · runtime dependencies: none (standard library only)

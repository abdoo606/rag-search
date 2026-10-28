from __future__ import annotations

import pytest

from rag.retriever import Index, answer, chunk_document, tokenise

DOC_A = (
    "Refunds are processed within five business days. "
    "Digital purchases are refundable only if the download never started. "
    "Contact support to start a refund request."
)
DOC_B = (
    "Shipping takes three to seven days for domestic orders. "
    "International shipping can take up to four weeks. "
    "We do not ship to PO boxes."
)


@pytest.fixture()
def index():
    index = Index()
    index.add(chunk_document(DOC_A, "refunds", size=12, overlap=4))
    index.add(chunk_document(DOC_B, "shipping", size=12, overlap=4))
    return index


def test_tokenise():
    assert tokenise("Refunds, Days!") == ["refunds", "days"]


def test_chunking_covers_every_word():
    chunks = chunk_document(" ".join(str(i) for i in range(30)), "d", size=10, overlap=3)
    covered = set()
    for chunk in chunks:
        covered.update(chunk.text.split())
    assert len(covered) == 30


def test_chunk_overlap_shares_words():
    chunks = chunk_document("a b c d e f", "d", size=4, overlap=2)
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_chunk_validation():
    with pytest.raises(ValueError):
        chunk_document("a b", "d", size=5, overlap=5)


def test_search_finds_the_right_document(index):
    chunk, score = index.search("refund request", k=1)[0]
    assert chunk.doc_id == "refunds" and score > 0


def test_search_returns_k_results(index):
    assert len(index.search("shipping days", k=2)) == 2


def test_search_with_no_overlap_returns_nothing(index):
    assert index.search("astrophysics telescope") == []


def test_answer_is_extractive_and_cited(index):
    result = answer(index, "when are refunds processed")
    assert result.confident
    assert "five business days" in result.text
    assert result.citations[0].startswith("refunds#")


def test_answer_refuses_when_unsure(index):
    result = answer(index, "what is the airspeed of a swallow")
    assert not result.confident
    assert result.citations == []
    assert "could not find" in result.text


def test_empty_index_is_safe():
    assert Index().search("anything") == []
    assert not answer(Index(), "anything").confident


def test_invalid_k(index):
    assert index.search("refund", k=0) == []

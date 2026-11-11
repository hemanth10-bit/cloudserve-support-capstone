"""TC-FR03: corpus chunking and retrieval (FR-03, Build Spec A4)."""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from src.config import get_settings
from src.corpus import build_chunks, load_documents
from src.retrieval import Retriever

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def docs():
    return load_documents(DATA / "documentation.json")


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    settings = dataclasses.replace(get_settings(), chroma_path=tmp_path_factory.mktemp("chroma"))
    return Retriever(settings)


# ------------------------------------------------------------------ chunking
def test_every_article_is_chunked_by_section(docs):
    chunks = build_chunks(docs)
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, set()).add(c.section)
    assert set(by_doc) == {d["doc_id"] for d in docs}
    for sections in by_doc.values():
        assert {"symptoms", "common_causes", "resolution", "notes"} <= sections


def test_chunk_ids_are_unique(docs):
    ids = [c.chunk_id for c in build_chunks(docs)]
    assert len(ids) == len(set(ids))


def test_resolution_steps_are_never_split(docs):
    for d in docs:
        res = next(c for c in build_chunks([d]) if c.section == "resolution")
        numbered = re.findall(r"^\d+\.", d["content"].split("## Resolution")[1].split("## Notes")[0], re.M)
        assert len(re.findall(r"^\d+\.", res.text, re.M)) == len(numbered) > 0


def test_chunk_text_comes_from_the_article(docs):
    for d in docs:
        for c in build_chunks([d]):
            assert c.text in d["content"] or c.section == "overview"


# ----------------------------------------------------------------- retrieval
def test_known_question_finds_the_right_article(retriever):
    # Ines's example: the customer's words never appear in the article title.
    res = retriever.search("my deployment keeps dying and the container is restarting")
    assert res.error is None
    assert res.doc_ids[0] == "DOC-DEPLOY-001"


def test_known_questions_per_topic(retriever):
    cases = {
        "I keep getting invalid credentials even though my password is right": "DOC-AUTH-001",
        "how do I roll back a bad release": "DOC-DEPLOY-002",
        "why is my invoice higher than last month": "DOC-BILL-001",
        "I need to export all of our account data": "DOC-DATA-001",
    }
    for q, want in cases.items():
        res = retriever.search(q, top_k=5)
        assert want in res.doc_ids[:3], (q, res.doc_ids)


def test_returned_sources_resolve_to_the_real_corpus(retriever, docs):
    ids = {d["doc_id"] for d in docs}
    res = retriever.search("webhook deliveries are failing with signature errors")
    assert res.passages
    for p in res.passages:
        assert p.doc_id in ids
        chunk = retriever.get_passage(p.chunk_id)
        assert chunk is not None and chunk.text == p.text
        assert retriever.get_document(p.doc_id)["title"] == p.title


def test_scores_are_ranked_and_in_range(retriever):
    res = retriever.search("database connection pool exhausted")
    scores = [p.score for p in res.passages]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert res.top_score == scores[0]


def test_search_is_deterministic(retriever):
    a = retriever.search("rate limit errors on the API")
    b = retriever.search("rate limit errors on the API")
    assert [(p.chunk_id, p.score) for p in a.passages] == [(p.chunk_id, p.score) for p in b.passages]


def test_threshold_can_return_nothing(retriever):
    res = retriever.search("rate limit errors on the API", min_score=0.99)
    assert res.passages == [] and res.doc_ranking == [] and res.error is None


def test_unrelated_text_returns_little_or_nothing(retriever):
    res = retriever.search("zxcv qwer asdf")
    assert all(p.score >= retriever.settings.retrieval_min_score for p in res.passages)


def test_empty_and_blank_queries_do_not_raise(retriever):
    for q in ("", "   ", None):
        res = retriever.search(q)
        assert res.passages == [] and res.error == "empty_query"


def test_unavailable_index_degrades_instead_of_raising(retriever):
    broken = Retriever.__new__(Retriever)
    broken.settings, broken._collection, broken._build_error = retriever.settings, None, "boom"
    broken.chunks_by_id = {}
    res = broken.search("anything")
    assert res.passages == [] and res.error.startswith("index_unavailable")


def test_index_is_reused_when_corpus_unchanged(tmp_path):
    settings = dataclasses.replace(get_settings(), chroma_path=tmp_path)
    first = Retriever(settings)
    second = Retriever(settings)
    assert first._collection.count() == second._collection.count() == len(first.chunks)
    assert json.dumps(first.search("export data").doc_ids) == json.dumps(second.search("export data").doc_ids)

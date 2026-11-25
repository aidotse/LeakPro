"""Elasticsearch verifier against an in-process fake client (no server needed)."""

import sys
import types
from typing import Any, Dict, List

import pytest

from leakpro.attacks.utils.extraction.elasticsearch_verifier import (
    ElasticsearchVerifier,
    chunk_document,
    connect,
    index_corpus,
)
from leakpro.attacks.utils.extraction.records import Windowing
from leakpro.attacks.utils.extraction.verification import InMemoryVerifier


class FakeES:
    """Indexes chunks like `index_corpus`; `match_phrase` is a plain (looser) substring test."""

    def __init__(self, docs: List[tuple], chunk_words: int = 100, stride: int = 50) -> None:
        self.chunks = [
            {"_id": f"{doc_id}_{start}", "_source": {"doc_id": doc_id, "word_start": start, "text": chunk}}
            for doc_id, text in docs for start, chunk in chunk_document(text, chunk_words, stride)
        ]
        self.searches = 0
        self.msearches = 0

    def count(self, index: str) -> Dict[str, int]:
        return {"count": len(self.chunks)}

    def _hits(self, query: Dict[str, Any], size: int) -> Dict[str, Any]:
        phrase = query["match_phrase"]["text"]["query"]
        assert query["match_phrase"]["text"]["slop"] == 0
        return {"hits": {"hits": [c for c in self.chunks if phrase in c["_source"]["text"]][:size]}}

    def search(self, index: str, query: Dict[str, Any], size: int) -> Dict[str, Any]:
        self.searches += 1
        return self._hits(query, size)

    def msearch(self, searches: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.msearches += 1
        bodies = searches[1::2]
        return {"responses": [self._hits(b["query"], b["size"]) for b in bodies]}


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_cover_the_tail_and_short_documents() -> None:
    assert chunk_document("A  b\nC", 100, 50) == [(0, "a b c")]
    starts = [s for s, _ in chunk_document(_words("w", 130), 100, 50)]
    assert starts == [0, 30]
    starts = [s for s, _ in chunk_document(_words("w", 150), 100, 50)]
    assert starts == [0, 50]


def test_lookup_reports_document_word_offset() -> None:
    doc = _words("w", 150)
    verifier = ElasticsearchVerifier(FakeES([("d1", doc)]), "idx")
    window = " ".join(doc.split()[120:150])
    assert verifier.lookup(window) == ("d1", 120)
    assert verifier.lookup("not there") is None


def test_hits_are_rechecked_as_substrings() -> None:
    verifier = ElasticsearchVerifier(FakeES([("d1", "xalpha beta gamma")]), "idx")
    assert verifier.lookup("alpha beta") == ("d1", 0)  # inside the word "xalpha", as `window in chunk`
    assert verifier.lookup("beta gamma") == ("d1", 1)
    assert verifier.lookup("beta delta") is None


def test_matches_in_memory_verifier_in_both_modes() -> None:
    docs = [("a", "Intro. " + _words("x", 80) + " outro"), ("b", _words("y", 40))]
    text = " ".join(["junk"] * 7 + _words("x", 80).split()[10:60] + ["tail"])
    windowing = Windowing(size=30, stride=5, min_span=30)
    es = FakeES(docs)
    es_verifier, mem_verifier = ElasticsearchVerifier(es, "idx"), InMemoryVerifier(docs)
    for mode in ("first", "all"):
        assert es_verifier.verify(text, windowing, mode) == mem_verifier.verify(text, windowing, mode)
    assert es.msearches >= 1


def test_corpus_id_depends_on_index_size_and_version() -> None:
    small, large = FakeES([("a", "one two")]), FakeES([("a", "one two"), ("b", "three")])
    assert ElasticsearchVerifier(small, "idx").corpus_id != ElasticsearchVerifier(large, "idx").corpus_id
    assert (ElasticsearchVerifier(small, "idx", corpus_version="v1").corpus_id
            != ElasticsearchVerifier(small, "idx", corpus_version="v2").corpus_id)


def test_msearch_errors_are_raised() -> None:
    class Broken(FakeES):
        def msearch(self, searches: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {"responses": [{"error": "boom"}]}

    verifier = ElasticsearchVerifier(Broken([("a", "x")]), "idx")
    with pytest.raises(RuntimeError, match="boom"):
        verifier.lookup_many(["x y"])


def test_legacy_offset_field_token_start_is_read() -> None:
    es = FakeES([("d1", _words("w", 150))])
    for chunk in es.chunks:  # an older index that names the offset `token_start`
        chunk["_source"]["token_start"] = chunk["_source"].pop("word_start")
    window = " ".join(_words("w", 150).split()[120:150])
    assert ElasticsearchVerifier(es, "idx").lookup(window) == ("d1", 120)


def test_connect_reads_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    elasticsearch = pytest.importorskip("elasticsearch")
    calls = []
    monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda host, **kwargs: calls.append((host, kwargs)))
    monkeypatch.setenv("ES_HOST", "https://es:9200")
    monkeypatch.setenv("ES_USER", "u")
    monkeypatch.setenv("ES_PASSWORD", "p")
    monkeypatch.setenv("ES_CA_CERT", "~/ca.crt")
    connect()
    host, kwargs = calls[0]
    assert host == "https://es:9200"
    assert kwargs["basic_auth"] == ("u", "p")
    assert kwargs["ca_certs"].endswith("/ca.crt") and not kwargs["ca_certs"].startswith("~")
    connect(ca_certs="/explicit.crt")  # an explicit setting wins
    assert calls[1][1]["ca_certs"] == "/explicit.crt"


class _Indices:
    """Stand-in for `client.indices`: records what index_corpus does."""

    def __init__(self) -> None:
        self.created: Dict[str, Any] = {}
        self.settings: List[Dict[str, Any]] = []
        self.refreshed = False

    def exists(self, index: str) -> bool:
        return index in self.created

    def delete(self, index: str) -> None:
        self.created.pop(index)

    def create(self, index: str, settings: Dict[str, Any], mappings: Dict[str, Any]) -> None:
        self.created[index] = {"settings": settings, "mappings": mappings}

    def put_settings(self, index: str, settings: Dict[str, Any]) -> None:
        self.settings.append(settings)

    def refresh(self, index: str) -> None:
        self.refreshed = True

    def get_mapping(self, index: str) -> Dict[str, Any]:
        return {index: {"mappings": self.created[index]["mappings"]}}


def test_index_corpus_records_chunk_settings_and_indexes_every_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    indexed: List[Dict[str, Any]] = []

    def bulk(client: Any, actions: Any, chunk_size: int) -> tuple:
        indexed.extend(actions)
        return len(indexed), []

    # index_corpus imports `elasticsearch.helpers` lazily; a stand-in module keeps this test free of the package
    fake = types.ModuleType("elasticsearch")
    fake.helpers = types.SimpleNamespace(bulk=bulk)
    monkeypatch.setitem(sys.modules, "elasticsearch", fake)
    client = types.SimpleNamespace(indices=_Indices())
    count = index_corpus(client, "idx", iter([("d1", _words("w", 130)), ("d2", "Short Doc")]), chunk_words=60, stride=20)

    assert count == len(indexed) == 6  # d1: chunks at 0, 20, 40, 60 and the tail at 70; d2: one chunk
    assert [a["_source"]["word_start"] for a in indexed] == [0, 20, 40, 60, 70, 0]
    assert indexed[-1]["_source"] == {"doc_id": "d2", "word_start": 0, "text": "short doc"}
    mappings = client.indices.created["idx"]["mappings"]
    assert mappings["_meta"] == {"chunk_words": 60, "stride": 20}
    assert mappings["properties"]["text"]["index_options"] == "positions"
    assert client.indices.refreshed and client.indices.settings[-1] == {"index": {"refresh_interval": "1s"}}

    es = FakeES([("d1", "x")])
    es.indices = client.indices
    assert ElasticsearchVerifier(es, "idx").max_window_words() == 41  # 60 - 20 + 1


def test_max_window_words_defaults_for_an_index_without_meta() -> None:
    assert ElasticsearchVerifier(FakeES([("d1", "x")]), "idx").max_window_words() == 51

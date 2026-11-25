#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Verify candidates against a corpus indexed in Elasticsearch.

For corpora too large to hold in memory, such as a 20 GB RedPajama sample.
The corpus is indexed once as overlapping word chunks (:func:`index_corpus`); each window is then
looked up with a ``match_phrase`` query (``slop=0``) and every hit is re-checked with a literal
substring test on the normalized chunk, so Elasticsearch's analyzer can never make a match looser
than :class:`~leakpro.attacks.utils.extraction.verification.InMemoryVerifier`'s.

Chunks are ``chunk_words`` long and start every ``stride`` words. Any window of at most
``chunk_words - stride + 1`` words therefore lies inside one chunk; the defaults (100 / 50) cover
windows up to 51 words. The last chunk of a document is
aligned to its end and documents shorter than one chunk are indexed whole, so no tail is lost.

The ``elasticsearch`` package is imported only when a client is created from settings
(:meth:`ElasticsearchVerifier.connect`); tests pass any object with ``search``/``msearch``/``count``.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from leakpro.attacks.utils.extraction.records import normalize
from leakpro.attacks.utils.extraction.verification import Verifier, word_offset
from leakpro.utils.import_helper import Self

INDEX_SETTINGS: Dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "verbatim_analyzer": {"type": "custom", "tokenizer": "whitespace", "filter": ["lowercase"]},
            },
        },
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "word_start": {"type": "integer"},
            "text": {"type": "text", "analyzer": "verbatim_analyzer", "index_options": "positions"},
        },
    },
}


def connect(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password_env: str = "ES_PASSWORD",  # noqa: S107 (the name of an environment variable, not a password)
    ca_certs: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Create an ``elasticsearch.Elasticsearch`` client.

    The password is read from the environment variable ``password_env`` so it never appears in a
    config file. ``host``, ``user`` and ``ca_certs`` fall back to ``ES_HOST`` / ``ES_USER`` /
    ``ES_CA_CERT``.
    """
    try:
        from elasticsearch import Elasticsearch  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("The Elasticsearch verifier needs `pip install elasticsearch`") from e
    host = host or os.environ.get("ES_HOST", "https://localhost:9200")
    user = user or os.environ.get("ES_USER")
    ca_certs = ca_certs or os.environ.get("ES_CA_CERT")
    password = os.environ.get(password_env)
    kwargs: Dict[str, Any] = {}
    if user is not None:
        kwargs["basic_auth"] = (user, password)
    if ca_certs is not None:
        kwargs["ca_certs"] = os.path.expanduser(ca_certs)
    return Elasticsearch(host, **kwargs)


def chunk_document(text: str, chunk_words: int = 100, stride: int = 50) -> List[Tuple[int, str]]:
    """``(word_start, normalized_chunk)`` for the overlapping chunks of one document."""
    words = normalize(text).split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [(0, " ".join(words))]
    starts = list(range(0, len(words) - chunk_words + 1, stride))
    if starts[-1] != len(words) - chunk_words:
        starts.append(len(words) - chunk_words)
    return [(s, " ".join(words[s:s + chunk_words])) for s in starts]


def iter_jsonl_documents(path: str, text_field: str = "text", id_field: str = "doc_id") -> Iterator[Tuple[str, str]]:
    """``(doc_id, text)`` per line of a JSON-lines corpus; a missing id falls back to the line number."""
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                yield str(record.get(id_field, line_no)), record[text_field]


def index_corpus(
    client: Any,  # noqa: ANN401
    index: str,
    documents: Iterator[Tuple[str, str]],
    chunk_words: int = 100,
    stride: int = 50,
    recreate: bool = True,
    bulk_size: int = 1000,
) -> int:
    """Index ``(doc_id, text)`` documents as overlapping chunks; returns the number of chunks."""
    from elasticsearch import helpers  # noqa: PLC0415

    if recreate and client.indices.exists(index=index):
        client.indices.delete(index=index)
    if not client.indices.exists(index=index):
        client.indices.create(index=index, **INDEX_SETTINGS)
    client.indices.put_settings(index=index, settings={"index": {"refresh_interval": "-1"}})

    def _actions() -> Iterator[Dict[str, Any]]:
        for doc_id, text in documents:
            for start, chunk in chunk_document(text, chunk_words, stride):
                yield {"_index": index, "_id": f"{doc_id}_{start}",
                       "_source": {"doc_id": doc_id, "word_start": start, "text": chunk}}

    count, _ = helpers.bulk(client, _actions(), chunk_size=bulk_size)
    client.indices.put_settings(index=index, settings={"index": {"refresh_interval": "1s"}})
    client.indices.refresh(index=index)
    return count


class ElasticsearchVerifier(Verifier):
    """Look windows up in an index built by :func:`index_corpus`.

    Args:
    ----
        client: An ``elasticsearch.Elasticsearch`` client (or anything with the same ``search``,
            ``msearch`` and ``count`` methods).
        index: Index name.
        hits_per_query: Hits re-checked per window.
        corpus_version: Free-form label of the indexed corpus. It goes into every cache key
            together with the index name and its chunk count; change it when you re-index different
            data with the same number of chunks.
        batch_size: Windows per ``msearch`` request in ``all`` mode.

    """

    def __init__(
        self: Self,
        client: Any,  # noqa: ANN401
        index: str,
        hits_per_query: int = 5,
        corpus_version: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.client = client
        self.index = index
        self.hits_per_query = hits_per_query
        self.batch_size = batch_size
        count = client.count(index=index)["count"]
        self._corpus_id = f"elasticsearch:{index}:{count}:{corpus_version or ''}"
        self._cache: Dict[str, Optional[Tuple[str, int]]] = {}

    @property
    def corpus_id(self: Self) -> str:
        """Index name, chunk count and ``corpus_version``."""
        return self._corpus_id

    def _query(self: Self, window: str) -> Dict[str, Any]:
        return {"match_phrase": {"text": {"query": window, "slop": 0}}}

    @staticmethod
    def _check(window: str, hits: Sequence[Dict[str, Any]]) -> Optional[Tuple[str, int]]:
        for hit in hits:
            source = hit["_source"]
            offset = word_offset(normalize(source["text"]), window)
            if offset is not None:
                # Older indexes name the chunk offset `token_start` (also a word offset).
                start = source.get("word_start", source.get("token_start", 0))
                return str(source.get("doc_id", hit.get("_id"))), int(start) + offset
        return None

    def lookup(self: Self, window: str) -> Optional[Tuple[str, int]]:
        """First indexed chunk containing ``window`` (substring test), with the document word offset."""
        if window not in self._cache:
            res = self.client.search(index=self.index, query=self._query(window), size=self.hits_per_query)
            self._cache[window] = self._check(window, res["hits"]["hits"])
        return self._cache[window]

    def lookup_many(self: Self, windows: Sequence[str]) -> List[Optional[Tuple[str, int]]]:
        """Look up uncached windows with ``msearch``, ``batch_size`` at a time."""
        todo = list(dict.fromkeys(w for w in windows if w not in self._cache))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            searches: List[Dict[str, Any]] = []
            for window in batch:
                searches.append({"index": self.index})
                searches.append({"query": self._query(window), "size": self.hits_per_query})
            responses = self.client.msearch(searches=searches)["responses"]
            for window, res in zip(batch, responses):
                if "error" in res:
                    raise RuntimeError(f"Elasticsearch query failed: {res['error']}")
                self._cache[window] = self._check(window, res["hits"]["hits"])
        return [self._cache[w] for w in windows]

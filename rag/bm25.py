"""Okapi BM25 over the lexemes produced by PostgreSQL's Italian text search configuration,
and its fusion with dense retrieval."""

import math
import time
from collections import Counter, defaultdict
from dataclasses import replace

import psycopg
from psycopg import sql

from rag.db import table_identifier
from rag.retrieval import RetrievedChunk, dense_search

# One row per (chunk, lexeme): the number of positions is the term frequency
CHUNK_LEXEMES = """
SELECT c.id, u.lexeme, cardinality(u.positions)
FROM {table} AS c, unnest(c.content_tsv) AS u
"""
QUESTION_LEXEMES = "SELECT u.lexeme FROM unnest(to_tsvector('italian', %s)) AS u"
FETCH_CHUNKS = """
SELECT id, doc_id, chunk_index, page_start, page_end, label_start, label_end, content
FROM {table}
WHERE id = ANY(%s)
"""


class BM25Index:
    def __init__(self, postings: dict[str, list[tuple[int, int]]], lengths: dict[int, int],
                 k1: float = 1.2, b: float = 0.75):
        self.postings = postings  # lexeme -> [(row id, term frequency)]
        self.lengths = lengths    # row id -> number of lexemes in the chunk
        self.avg_length = sum(lengths.values()) / len(lengths)
        self.k1, self.b = k1, b
        n = len(lengths)
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1) for t, p in postings.items()}

    @classmethod
    def from_table(cls, conn: psycopg.Connection, table: str) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: Counter = Counter()
        for row_id, lexeme, tf in conn.execute(sql.SQL(CHUNK_LEXEMES).format(table=table_identifier(table))):
            postings[lexeme].append((row_id, tf))
            lengths[row_id] += tf
        return cls(dict(postings), dict(lengths))

    def search(self, terms: list[str], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            for row_id, tf in self.postings.get(term, []):
                length_norm = 1 - self.b + self.b * self.lengths[row_id] / self.avg_length
                scores[row_id] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


_indexes: dict[str, BM25Index] = {}  # built once per process and table


def bm25_search(
    conn: psycopg.Connection, table: str, question: str, k: int
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    if table not in _indexes:
        _indexes[table] = BM25Index.from_table(conn, table)
    t0 = time.perf_counter()
    terms = [row[0] for row in conn.execute(QUESTION_LEXEMES, (question,))]
    ranked = _indexes[table].search(terms, k)
    rows = conn.execute(sql.SQL(FETCH_CHUNKS).format(table=table_identifier(table)),
                        ([row_id for row_id, _ in ranked],)).fetchall()
    by_id = {row[0]: row[1:] for row in rows}
    chunks = [RetrievedChunk(*by_id[row_id], score) for row_id, score in ranked]
    return chunks, {"embed": 0.0, "search": time.perf_counter() - t0}


def rrf_fuse(rankings: list[list[RetrievedChunk]], k: int, rrf_k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: sum of 1 / (rrf_k + rank) over the rankings where a chunk appears."""
    scores: dict[tuple[str, int], float] = defaultdict(float)
    chunks: dict[tuple[str, int], RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = (chunk.doc_id, chunk.chunk_index)
            scores[key] += 1 / (rrf_k + rank)
            chunks.setdefault(key, chunk)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [replace(chunks[key], score=scores[key]) for key in best]


def hybrid_bm25_search(
    conn: psycopg.Connection, table: str, question: str, k: int, candidates: int = 20,
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    dense, dense_times = dense_search(conn, table, question, candidates)
    lexical, lexical_times = bm25_search(conn, table, question, candidates)
    return rrf_fuse([dense, lexical], k), {
        "embed": dense_times["embed"], "search": dense_times["search"] + lexical_times["search"]}
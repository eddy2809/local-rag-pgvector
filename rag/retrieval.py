"""Dense retrieval over a pgvector table."""

import time
from dataclasses import dataclass

import psycopg
from psycopg import sql

from rag.db import table_identifier
from rag.embeddings import embed_query

DENSE_SEARCH = """
SELECT doc_id, chunk_index, page_start, page_end, label_start, label_end, content,
       1 - (embedding <=> %(q)s) AS similarity
FROM {table}
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_index: int
    page_start: int
    page_end: int
    label_start: str
    label_end: str
    content: str
    score: float  # cosine similarity (dense), ts_rank (lexical) or RRF score (hybrid)

    @property
    def pages(self) -> str:
        """Printed page range used in citations, e.g. "70" or "70-71"."""
        if self.label_start == self.label_end:
            return self.label_start
        return f"{self.label_start}-{self.label_end}"


def dense_search(
    conn: psycopg.Connection, table: str, question: str, k: int
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """Return the k chunks closest to the question and the time spent in each step."""
    t0 = time.perf_counter()
    q = embed_query(question)
    t1 = time.perf_counter()
    rows = conn.execute(
        sql.SQL(DENSE_SEARCH).format(table=table_identifier(table)), {"q": q, "k": k}
    ).fetchall()
    t2 = time.perf_counter()
    return [RetrievedChunk(*row) for row in rows], {"embed": t1 - t0, "search": t2 - t1}



# plainto_tsquery joins all words with AND (&): for natural-language questions almost no chunk
# would match, so the ANDs are turned into ORs (|) and ranking rewards chunks with more matches.
LEXICAL_SEARCH = """
SELECT doc_id, chunk_index, page_start, page_end, label_start, label_end, content,
       ts_rank(content_tsv, query) AS score
FROM {table},
     to_tsquery('italian', replace(plainto_tsquery('italian', %(q)s)::text, ' & ', ' | ')) AS query
WHERE content_tsv @@ query
ORDER BY score DESC
LIMIT %(k)s
"""

# Reciprocal Rank Fusion of the dense and the lexical ranking, in a single query
HYBRID_SEARCH = """
WITH dense AS (
    SELECT id, row_number() OVER (ORDER BY distance) AS rank
    FROM (SELECT id, embedding <=> %(emb)s AS distance
          FROM {table}
          ORDER BY embedding <=> %(emb)s
          LIMIT %(n)s) AS d
),
lexical AS (
    SELECT id, row_number() OVER (ORDER BY lex_score DESC) AS rank
    FROM (SELECT id, ts_rank(content_tsv, query) AS lex_score
          FROM {table},
               to_tsquery('italian', replace(plainto_tsquery('italian', %(q)s)::text, ' & ', ' | ')) AS query
          WHERE content_tsv @@ query
          ORDER BY lex_score DESC
          LIMIT %(n)s) AS l
),
fused AS (
    SELECT id, sum(1.0 / (%(rrf_k)s + rank)) AS score
    FROM (SELECT * FROM dense UNION ALL SELECT * FROM lexical) AS ranked
    GROUP BY id
)
SELECT c.doc_id, c.chunk_index, c.page_start, c.page_end, c.label_start, c.label_end, c.content, f.score
FROM fused AS f
JOIN {table} AS c USING (id)
ORDER BY f.score DESC
LIMIT %(k)s
"""


def lexical_search(
    conn: psycopg.Connection, table: str, question: str, k: int
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """Full-text search only (Italian stemming, OR of the question terms)."""
    t0 = time.perf_counter()
    rows = conn.execute(
        sql.SQL(LEXICAL_SEARCH).format(table=table_identifier(table)), {"q": question, "k": k}
    ).fetchall()
    return [RetrievedChunk(*row) for row in rows], {"embed": 0.0, "search": time.perf_counter() - t0}


def hybrid_search(
    conn: psycopg.Connection, table: str, question: str, k: int,
    candidates: int = 20, rrf_k: int = 60,
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """Dense and lexical search fused with Reciprocal Rank Fusion."""
    t0 = time.perf_counter()
    q = embed_query(question)
    t1 = time.perf_counter()
    rows = conn.execute(
        sql.SQL(HYBRID_SEARCH).format(table=table_identifier(table)),
        {"emb": q, "q": question, "n": candidates, "rrf_k": rrf_k, "k": k},
    ).fetchall()
    t2 = time.perf_counter()
    return [RetrievedChunk(*row) for row in rows], {"embed": t1 - t0, "search": t2 - t1}
"""Dense retrieval over a pgvector table."""

import time
from dataclasses import dataclass

import psycopg
from psycopg import sql

from rag.db import table_identifier
from rag.embeddings import embed_query

DENSE_SEARCH = """
SELECT chunk_index, page_start, page_end, label_start, label_end, content,
       1 - (embedding <=> %(q)s) AS similarity
FROM {table}
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""


@dataclass
class RetrievedChunk:
    chunk_index: int
    page_start: int
    page_end: int
    label_start: str
    label_end: str
    content: str
    similarity: float

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
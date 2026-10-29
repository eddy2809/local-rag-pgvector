"""Embed chunks with bge-m3 and store them in a PostgreSQL/pgvector table."""

import argparse
import json
import re
import time
from pathlib import Path

from psycopg import sql

from rag.config import EMBED_DIM
from rag.db import get_connection
from rag.embeddings import embed_query, embed_texts

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id          bigserial PRIMARY KEY,
    doc_id      text    NOT NULL,
    chunk_index integer NOT NULL,
    page_start  integer NOT NULL,
    page_end    integer NOT NULL,
    label_start text,
    label_end   text,
    n_tokens    integer NOT NULL,
    content     text    NOT NULL,
    embedding   vector({dim}) NOT NULL,
    UNIQUE (doc_id, chunk_index)
)
"""

# vector_cosine_ops must match the <=> operator used at query time, otherwise the index is ignored
CREATE_INDEX = "CREATE INDEX IF NOT EXISTS {index} ON {table} USING hnsw (embedding vector_cosine_ops)"

INSERT = """
INSERT INTO {table}
    (doc_id, chunk_index, page_start, page_end, label_start, label_end, n_tokens, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

SEARCH = """
SELECT chunk_index, label_start, label_end, 1 - (embedding <=> %s) AS similarity
FROM {table}
ORDER BY embedding <=> %s
LIMIT 3
"""

TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_file", type=Path)
    parser.add_argument("--table", required=True, help="target table, e.g. chunks_300")
    args = parser.parse_args()
    if not TABLE_NAME.match(args.table):
        raise SystemExit(f"Invalid table name: {args.table!r}")

    table = sql.Identifier(args.table)
    index = sql.Identifier(f"{args.table}_embedding_hnsw")
    chunks = load_chunks(args.chunks_file)

    embed_texts(["warm-up"])  # load the model before timing
    t0 = time.perf_counter()
    vectors, ollama_tokens = embed_texts([c["text"] for c in chunks])
    embed_seconds = time.perf_counter() - t0

    expected_tokens = sum(c["n_tokens"] for c in chunks) + 2 * len(chunks)  # + <s> and </s>
    print(f"Embedded {len(chunks)} chunks in {embed_seconds:.2f} s "
          f"({len(chunks) / embed_seconds:.0f} chunks/s)")
    print(f"Tokens: ours + special = {expected_tokens}, Ollama = {ollama_tokens}")

    doc_ids = sorted({c["doc_id"] for c in chunks})
    rows = [
        (c["doc_id"], c["chunk_index"], c["page_start"], c["page_end"],
         c["label_start"], c["label_end"], c["n_tokens"], c["text"], vector)
        for c, vector in zip(chunks, vectors)
    ]

    t0 = time.perf_counter()
    with get_connection() as conn:  # one transaction: committed only if everything succeeds
        conn.execute(sql.SQL(CREATE_TABLE).format(table=table, dim=sql.Literal(EMBED_DIM)))
        conn.execute(sql.SQL("DELETE FROM {table} WHERE doc_id = ANY(%s)").format(table=table),
                     (doc_ids,))
        with conn.cursor() as cur:
            cur.executemany(sql.SQL(INSERT).format(table=table), rows)
        conn.execute(sql.SQL(CREATE_INDEX).format(index=index, table=table))
        total = conn.execute(sql.SQL("SELECT count(*) FROM {table}").format(table=table)).fetchone()[0]
    print(f"Stored and indexed in {time.perf_counter() - t0:.2f} s | rows in {args.table}: {total}")

    # Quick sanity check of the retrieval
    question = "Perché non è stato possibile eseguire il modello linguistico direttamente sul visore?"
    q = embed_query(question)
    with get_connection() as conn:
        results = conn.execute(sql.SQL(SEARCH).format(table=table), (q, q)).fetchall()
    print(f"\nTest question: {question}")
    for chunk_index, label_start, label_end, similarity in results:
        print(f"  {similarity:.3f}  chunk {chunk_index:>3}  pp. {label_start}-{label_end}")


if __name__ == "__main__":
    main()
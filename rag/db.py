import psycopg
from pgvector.psycopg import register_vector

from rag.config import PG_DSN


def get_connection(autocommit: bool = False) -> psycopg.Connection:
    """Open a connection with the pgvector extension enabled and its types registered."""
    conn = psycopg.connect(PG_DSN, autocommit=autocommit)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if not autocommit:
        conn.commit()
    register_vector(conn)  # must run after CREATE EXTENSION: it looks up the type OID
    return conn
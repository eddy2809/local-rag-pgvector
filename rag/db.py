import psycopg
import re

from pgvector.psycopg import register_vector
from rag.config import PG_DSN
from psycopg import sql

TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")

def get_connection(autocommit: bool = False) -> psycopg.Connection:
    """Open a connection with the pgvector extension enabled and its types registered."""
    conn = psycopg.connect(PG_DSN, autocommit=autocommit)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if not autocommit:
        conn.commit()
    register_vector(conn)  # must run after CREATE EXTENSION: it looks up the type OID
    return conn


def table_identifier(name: str) -> sql.Identifier:
    """Validate a table name and return it as a safely quoted SQL identifier."""
    if not TABLE_NAME.match(name):
        raise ValueError(f"Invalid table name: {name!r}")
    return sql.Identifier(name)
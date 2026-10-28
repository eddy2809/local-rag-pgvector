import ollama

from rag.config import EMBED_DIM, EMBED_MODEL, LLM_MODEL
from rag.db import get_connection


def with_tag(name: str) -> str:
    """Ollama lists untagged models with ':latest' (e.g. 'bge-m3' -> 'bge-m3:latest')."""
    return name if ":" in name else f"{name}:latest"


def main() -> None:
    with get_connection() as conn:
        version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]
    print(f"[ok] PostgreSQL reachable, pgvector {version}")

    installed = {m.model for m in ollama.list().models}
    for name in (LLM_MODEL, EMBED_MODEL):
        status = "ok" if with_tag(name) in installed else "MISSING"
        print(f"[{status}] model {name}")

    emb = ollama.embed(model=EMBED_MODEL, input="test").embeddings[0]
    assert len(emb) == EMBED_DIM, f"embedding size {len(emb)}, expected {EMBED_DIM}"
    print(f"[ok] embedding size {len(emb)}")


if __name__ == "__main__":
    main()
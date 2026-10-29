"""Embedding helpers built on Ollama's bge-m3."""

import numpy as np
import ollama

from rag.config import EMBED_DIM, EMBED_MODEL

KEEP_ALIVE = "30m"  # keep the model in VRAM between calls


def embed_texts(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, int]:
    """Return an (n, dim) float32 array of embeddings and the number of tokens Ollama processed."""
    vectors: list[list[float]] = []
    tokens = 0
    for i in range(0, len(texts), batch_size):
        response = ollama.embed(
            model=EMBED_MODEL, input=texts[i:i + batch_size], keep_alive=KEEP_ALIVE
        )
        vectors.extend(response.embeddings)
        tokens += response.prompt_eval_count or 0

    array = np.asarray(vectors, dtype=np.float32)
    assert array.shape[1] == EMBED_DIM, f"embedding size {array.shape[1]}, expected {EMBED_DIM}"
    return array, tokens


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0][0]
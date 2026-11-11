"""Second-stage re-ranking with a multilingual cross-encoder (bge-reranker-v2-m3)."""

import time
from dataclasses import replace

from rag.config import RERANKER_MODEL
from rag.retrieval import RetrievedChunk, dense_search, hybrid_search

RERANK_CANDIDATES = 30  # chunks retrieved by the first stage and re-scored by the cross-encoder
MAX_LENGTH = 512        # question + chunk tokens: 300-token chunks fit without truncation

_model = None


def get_model():
    """Load the cross-encoder once. Imports are lazy: torch is only needed for this experiment."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        _model = CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH, device=device)
    return _model


def rerank(question: str, chunks: list[RetrievedChunk], k: int) -> tuple[list[RetrievedChunk], float]:
    """Score every (question, chunk) pair jointly and keep the k best chunks."""
    t0 = time.perf_counter()
    scores = get_model().predict(
        [(question, c.content) for c in chunks], batch_size=32, show_progress_bar=False
    )
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [replace(c, score=float(s)) for c, s in ranked], time.perf_counter() - t0


def _with_rerank(first_stage):
    """Wrap a first-stage search: retrieve RERANK_CANDIDATES chunks, then re-rank them."""
    def search(conn, table, question, k):
        candidates, timings = first_stage(conn, table, question, RERANK_CANDIDATES)
        reranked, seconds = rerank(question, candidates, k)
        return reranked, {**timings, "rerank": seconds}
    return search


dense_rerank_search = _with_rerank(dense_search)
hybrid_rerank_search = _with_rerank(
    lambda conn, table, question, k: hybrid_search(conn, table, question, k, candidates=k)
)
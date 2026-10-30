"""Evaluate retrieval quality (hit rate@k, MRR) and latency on the question set."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.bm25 import bm25_search, hybrid_bm25_search
from rag.db import get_connection
from rag.embeddings import embed_texts
from rag.rerank import dense_rerank_search, hybrid_rerank_search
from rag.retrieval import dense_search, hybrid_search, lexical_search

QUESTIONS_FILE = Path("eval/questions.jsonl")
RESULTS_DIR = Path("results")
K_VALUES = (1, 2, 3, 5)

SEARCH_METHODS = {
    "dense": dense_search,
    "lexical": lexical_search,
    "hybrid": hybrid_search,
    "bm25": bm25_search,
    "hybrid_bm25": hybrid_bm25_search,
    "dense_rerank": dense_rerank_search,
    "hybrid_rerank": hybrid_rerank_search,
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_relevant_rank(contents: list[str], evidence: str) -> int | None:
    """1-based rank of the first chunk containing the evidence, or None if it is not retrieved."""
    target = normalize(evidence)
    for rank, content in enumerate(contents, start=1):
        if target in normalize(content):
            return rank
    return None


def summarize(records: list[dict], depth: int) -> dict:
    n = len(records)
    summary = {"n": n}
    for k in K_VALUES:
        summary[f"hit@{k}"] = sum(r["rank"] is not None and r["rank"] <= k for r in records) / n
    summary[f"mrr@{depth}"] = sum(1 / r["rank"] for r in records if r["rank"]) / n
    return summary


def percentiles(values: list[float]) -> tuple[float, float]:
    return float(np.median(values)), float(np.percentile(values, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="chunks_300")
    parser.add_argument("--method", default="dense", choices=SEARCH_METHODS)
    parser.add_argument("--depth", type=int, default=10, help="retrieval depth, also used for MRR")
    args = parser.parse_args()

    search = SEARCH_METHODS[args.method]
    questions = [q for q in load_questions() if q["evidence"]]  # answerable questions only

    embed_texts(["warm-up"])
    records = []
    with get_connection() as conn:
        search(conn, args.table, "warm-up", args.depth)  # warm up database, indexes and models
        for q in questions:
            chunks, timings = search(conn, args.table, q["question"], args.depth)
            records.append({
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "rank": first_relevant_rank([c.content for c in chunks], q["evidence"]),
                "retrieved": [c.chunk_index for c in chunks],
                "embed_ms": timings["embed"] * 1000,
                "search_ms": timings["search"] * 1000,
                "rerank_ms": timings.get("rerank", 0.0) * 1000,
            })

    overall = summarize(records, args.depth)
    by_type = {t: summarize([r for r in records if r["type"] == t], args.depth)
               for t in sorted({r["type"] for r in records})}
    latency = {}
    for step in ("embed", "search", "rerank"):
        median, p95 = percentiles([r[f"{step}_ms"] for r in records])
        latency[f"{step}_ms_median"], latency[f"{step}_ms_p95"] = median, p95

    mrr_key = f"mrr@{args.depth}"
    header = f"{'group':<12} {'n':>3}  " + "  ".join(f"hit@{k}" for k in K_VALUES) + f"  {mrr_key}"
    print(f"\nRetrieval evaluation | table={args.table} method={args.method} depth={args.depth}\n")
    print(header)
    for name, s in [("all", overall)] + list(by_type.items()):
        hits = "  ".join(f"{s[f'hit@{k}']:5.2f}" for k in K_VALUES)
        print(f"{name:<12} {s['n']:>3}  {hits}  {s[mrr_key]:6.3f}")

    print(f"\nLatency (median / p95): embedding {latency['embed_ms_median']:.1f} / {latency['embed_ms_p95']:.1f} ms, "
          f"search {latency['search_ms_median']:.2f} / {latency['search_ms_p95']:.2f} ms, "
          f"rerank {latency['rerank_ms_median']:.1f} / {latency['rerank_ms_p95']:.1f} ms")

    misses = [r for r in records if r["rank"] is None or r["rank"] > 5]
    print(f"\nNot in the top 5 ({len(misses)}):")
    for r in misses:
        where = f"rank {r['rank']}" if r["rank"] else f"not in top {args.depth}"
        print(f"  {r['id']} [{r['type']}] {where:<14} {r['question']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "" if args.depth == 10 else f"_depth{args.depth}"
    out_file = RESULTS_DIR / f"retrieval_{args.table}_{args.method}{suffix}.json"
    out_file.write_text(json.dumps({
        "table": args.table, "method": args.method, "depth": args.depth,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall": overall, "by_type": by_type, "latency": latency, "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
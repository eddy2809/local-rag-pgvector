"""Evaluate retrieval quality (hit rate@k, MRR@10) and latency on the question set."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.db import get_connection
from rag.embeddings import embed_texts
from rag.retrieval import dense_search

QUESTIONS_FILE = Path("eval/questions.jsonl")
RESULTS_DIR = Path("results")
K_VALUES = (1, 2, 3, 5)
MAX_K = 10  # retrieval depth, also used for MRR

SEARCH_METHODS = {"dense": dense_search}


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


def summarize(records: list[dict]) -> dict:
    n = len(records)
    summary = {"n": n}
    for k in K_VALUES:
        summary[f"hit@{k}"] = sum(r["rank"] is not None and r["rank"] <= k for r in records) / n
    summary[f"mrr@{MAX_K}"] = sum(1 / r["rank"] for r in records if r["rank"]) / n
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="chunks_300")
    parser.add_argument("--method", default="dense", choices=SEARCH_METHODS)
    args = parser.parse_args()

    search = SEARCH_METHODS[args.method]
    questions = [q for q in load_questions() if q["evidence"]]  # answerable questions only

    embed_texts(["warm-up"])
    records = []
    with get_connection() as conn:
        search(conn, args.table, "warm-up", MAX_K)  # warm up the database path as well
        for q in questions:
            chunks, timings = search(conn, args.table, q["question"], MAX_K)
            records.append({
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "rank": first_relevant_rank([c.content for c in chunks], q["evidence"]),
                "retrieved": [c.chunk_index for c in chunks],
                "embed_ms": timings["embed"] * 1000,
                "search_ms": timings["search"] * 1000,
            })

    overall = summarize(records)
    by_type = {t: summarize([r for r in records if r["type"] == t])
               for t in sorted({r["type"] for r in records})}
    embed_ms = [r["embed_ms"] for r in records]
    search_ms = [r["search_ms"] for r in records]
    latency = {
        "embed_ms_median": float(np.median(embed_ms)), "embed_ms_p95": float(np.percentile(embed_ms, 95)),
        "search_ms_median": float(np.median(search_ms)), "search_ms_p95": float(np.percentile(search_ms, 95)),
    }

    header = f"{'group':<12} {'n':>3}  " + "  ".join(f"hit@{k}" for k in K_VALUES) + f"  mrr@{MAX_K}"
    print(f"\nRetrieval evaluation | table={args.table} method={args.method}\n")
    print(header)
    for name, s in [("all", overall)] + list(by_type.items()):
        hits = "  ".join(f"{s[f'hit@{k}']:5.2f}" for k in K_VALUES)
        print(f"{name:<12} {s['n']:>3}  {hits}  {s[f'mrr@{MAX_K}']:6.3f}")

    print(f"\nLatency: embedding median {latency['embed_ms_median']:.1f} ms (p95 {latency['embed_ms_p95']:.1f}), "
          f"search median {latency['search_ms_median']:.2f} ms (p95 {latency['search_ms_p95']:.2f})")

    misses = [r for r in records if r["rank"] is None or r["rank"] > 5]
    print(f"\nNot in the top 5 ({len(misses)}):")
    for r in misses:
        where = f"rank {r['rank']}" if r["rank"] else "not in top 10"
        print(f"  {r['id']} [{r['type']}] {where:<13} {r['question']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"retrieval_{args.table}_{args.method}.json"
    out_file.write_text(json.dumps({
        "table": args.table, "method": args.method, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall": overall, "by_type": by_type, "latency": latency, "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
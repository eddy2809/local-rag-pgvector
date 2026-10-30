"""Run the full RAG pipeline on the question set: automatic checks on abstentions and citations,
plus a CSV export for manual grading of correctness and faithfulness."""

import argparse
import csv
import json
from datetime import datetime

import numpy as np

from rag.db import get_connection
from rag.embeddings import embed_texts
from eval.evaluate_retrieval import RESULTS_DIR, first_relevant_rank, load_questions, normalize
from rag.generation import generate_answer, warm_up
from rag.retrieval import dense_search

CSV_COLUMNS = ["id", "type", "question", "expected_answer", "answer", "cited_sources",
               "evidence_rank", "abstained", "correctness", "faithful", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="chunks_300")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    embed_texts(["warm-up"])
    warm_up()

    records = []
    with get_connection() as conn:
        for q in load_questions():
            chunks, timings = dense_search(conn, args.table, q["question"], args.k)
            answer = generate_answer(q["question"], chunks)

            evidence = q["evidence"]
            cited = [chunks[i - 1] for i in answer.cited_ids]
            records.append({
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "answerable": evidence is not None,
                "answer": answer.text,
                "abstained": answer.is_abstention,
                "cited_ids": answer.cited_ids,
                "cited_sources": ", ".join(f"[{i}] pp. {chunks[i - 1].pages}" for i in answer.cited_ids),
                "evidence_rank": first_relevant_rank([c.content for c in chunks], evidence) if evidence else None,
                "cites_evidence": bool(evidence) and any(normalize(evidence) in normalize(c.content) for c in cited),
                "total_seconds": timings["embed"] + timings["search"] + answer.seconds,
                "output_tokens": answer.output_tokens,
            })
            print(f"  {q['id']} done ({answer.seconds:.2f} s)")

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]
    abstained = [r for r in answerable if r["abstained"]]
    evidence_retrieved = [r for r in answered if r["evidence_rank"]]
    totals = [r["total_seconds"] for r in records]

    print(f"\nAnswer evaluation | table={args.table} k={args.k}\n")
    print(f"Unanswerable questions, correct abstentions:  {sum(r['abstained'] for r in unanswerable)}/{len(unanswerable)}")
    print(f"Answerable questions, abstentions:            {len(abstained)}/{len(answerable)}")
    print(f"  with the evidence among the sources (model error):  "
          f"{sum(1 for r in abstained if r['evidence_rank'])}")
    print(f"  without the evidence (retrieval error, correct behaviour):  "
          f"{sum(1 for r in abstained if not r['evidence_rank'])}")
    print(f"Answers with at least one citation:           {sum(1 for r in answered if r['cited_ids'])}/{len(answered)}")
    print(f"Answers citing the evidence chunk (when retrieved): "
          f"{sum(r['cites_evidence'] for r in evidence_retrieved)}/{len(evidence_retrieved)}")
    print(f"\nEnd-to-end latency: median {np.median(totals):.2f} s, p95 {np.percentile(totals, 95):.2f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"answers_{args.table}_k{args.k}"
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps({
        "table": args.table, "k": args.k, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Semicolon separator and UTF-8 BOM so that Excel with Italian locale opens it correctly
    csv_file = RESULTS_DIR / f"{stem}.csv"
    with csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({**r, "correctness": "", "faithful": "", "notes": ""})
    print(f"\nSaved {RESULTS_DIR / (stem + '.json')} and {csv_file} (fill in correctness/faithful)")


if __name__ == "__main__":
    main()
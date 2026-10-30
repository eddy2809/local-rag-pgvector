"""Find which chunks contain given phrases and at which rank they were retrieved for a question."""

import json
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    results_file, chunks_file, question_id, *phrases = sys.argv[1:]
    results = json.loads(Path(results_file).read_text(encoding="utf-8"))
    record = next(r for r in results["records"] if r["id"] == question_id)
    rank_of = {idx: rank for rank, idx in enumerate(record["retrieved"], start=1)}
    chunks = load_jsonl(chunks_file)

    print(f"{question_id}: {record['question']}\n")
    for phrase in phrases:
        target = normalize(phrase)
        print(f'"{phrase}"')
        matches = [c for c in chunks if target in normalize(c["text"])]
        if not matches:
            print("  not found in any chunk")
        for c in matches:
            text = normalize(c["text"])
            pos = text.find(target)
            rank = rank_of.get(c["chunk_index"])
            where = f"rank {rank}" if rank else "not in top 10"
            print(f"  chunk {c['chunk_index']:>3} (pp. {c['label_start']}-{c['label_end']})  {where}")
            print(f"     ...{text[max(0, pos - 150):pos + len(target) + 150]}...")
        print()


if __name__ == "__main__":
    main()
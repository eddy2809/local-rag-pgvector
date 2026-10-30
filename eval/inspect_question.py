"""Show the chunks retrieved for one question, to analyse retrieval errors."""

import json
import re
import sys
import textwrap
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_list(evidence) -> list[str]:
    if evidence is None:
        return []
    return evidence if isinstance(evidence, list) else [evidence]


def main() -> None:
    results_file, chunks_file, question_id = sys.argv[1:4]
    results = json.loads(Path(results_file).read_text(encoding="utf-8"))
    chunks = {c["chunk_index"]: c for c in load_jsonl(chunks_file)}
    question = next(q for q in load_jsonl("eval/questions.jsonl") if q["id"] == question_id)
    record = next(r for r in results["records"] if r["id"] == question_id)

    evidences = [normalize(e) for e in as_list(question["evidence"])]
    gold = sorted(i for i, c in chunks.items() if any(e in normalize(c["text"]) for e in evidences))

    print(f"{question_id}: {question['question']}")
    print(f"Expected answer: {question['answer']}")
    print(f"Evidence: {question['evidence']}  ->  gold chunks {gold}, first hit at rank {record['rank']}\n")

    for rank, idx in enumerate(record["retrieved"][:5], start=1):
        c = chunks[idx]
        mark = "  <-- contains evidence" if idx in gold else ""
        print(f"{rank}. chunk {idx} (pp. {c['label_start']}-{c['label_end']}){mark}")
        print(textwrap.indent(textwrap.fill(normalize(c["text"]), 100), "   "))
        print()


if __name__ == "__main__":
    main()
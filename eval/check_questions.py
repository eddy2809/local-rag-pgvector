"""Check that every evidence snippet appears in the extracted pages and in at least one chunk."""

import json
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    questions = load_jsonl(Path("eval/questions.jsonl"))
    pages = {p["page_number"]: normalize(p["text"]) for p in load_jsonl(Path(sys.argv[1]))}
    chunks = [normalize(c["text"]) for c in load_jsonl(Path(sys.argv[2]))]

    problems = 0
    for q in questions:
        if q["evidence"] is None:
            continue
        evidence = normalize(q["evidence"])
        in_pages = [n for n, text in pages.items() if evidence in text]
        in_chunks = sum(evidence in c for c in chunks)
        ok = in_pages and set(in_pages) <= set(q["pages"]) and in_chunks > 0
        if not ok:
            problems += 1
        status = "ok " if ok else "BAD"
        print(f"{status} {q['id']}  pages found {in_pages} (expected {q['pages']}), chunks {in_chunks}")

    answerable = sum(q["evidence"] is not None for q in questions)
    print(f"\n{len(questions)} questions, {answerable} answerable, {problems} problems")


if __name__ == "__main__":
    main()
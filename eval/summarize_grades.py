"""Summarize the manual grades of the answers: correctness, faithfulness and their link to retrieval."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from eval.evaluate_retrieval import RESULTS_DIR

CORRECTNESS = ("correct", "partial", "wrong")
FAITHFUL = ("yes", "no")


def read_rows(path: Path) -> list[dict]:
    """Read the graded CSV, whether Excel saved it as UTF-8 or as Windows-1252."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open(encoding=encoding, newline="") as f:
                return list(csv.DictReader(f, delimiter=";"))
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Cannot decode {path}")


def read_grades(path: Path) -> dict[str, dict]:
    grades = {}
    for row in read_rows(path):
        correctness = row["correctness"].strip().lower()
        faithful = row["faithful"].strip().lower()
        if correctness not in CORRECTNESS or faithful not in FAITHFUL:
            raise SystemExit(f"Invalid grade for {row['id']}: correctness={correctness!r}, faithful={faithful!r}")
        grades[row["id"]] = {"correctness": correctness, "faithful": faithful,
                             "notes": (row.get("notes") or "").strip()}
    return grades


def print_row(name: str, records: list[dict]) -> dict:
    counts = Counter(r["correctness"] for r in records)
    faithful = sum(r["faithful"] == "yes" for r in records)
    print(f"{name:<22} {len(records):>3}  " + "  ".join(f"{counts[c]:>7}" for c in CORRECTNESS)
          + f"  {faithful:>4}/{len(records):<3}")
    return {"n": len(records), **{c: counts[c] for c in CORRECTNESS}, "faithful": faithful}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="answers_chunks_300_k5", help="results file name without extension")
    args = parser.parse_args()

    records = json.loads((RESULTS_DIR / f"{args.stem}.json").read_text(encoding="utf-8"))["records"]
    grades = read_grades(RESULTS_DIR / f"{args.stem}.csv")
    missing = [r["id"] for r in records if r["id"] not in grades]
    if missing:
        raise SystemExit(f"Missing grades for: {missing}")
    for r in records:
        r.update(grades[r["id"]])

    header = f"{'group':<22} {'n':>3}  " + "  ".join(f"{c:>7}" for c in CORRECTNESS) + "  faithful"
    summary = {}

    print("\nCorrectness and faithfulness by question type\n")
    print(header)
    summary["all"] = print_row("all", records)
    for t in sorted({r["type"] for r in records}):
        summary[t] = print_row(t, [r for r in records if r["type"] == t])

    answerable = [r for r in records if r["answerable"]]
    print("\nAnswerable questions, by retrieval outcome\n")
    print(header)
    summary["evidence_in_sources"] = print_row(
        "evidence in sources", [r for r in answerable if r["evidence_rank"]])
    summary["evidence_missing"] = print_row(
        "evidence missing", [r for r in answerable if not r["evidence_rank"]])

    print("\nAnswers that are not fully correct or not faithful\n")
    for r in records:
        if r["correctness"] != "correct" or r["faithful"] == "no":
            retrieval = f"evidence rank {r['evidence_rank']}" if r["evidence_rank"] else "evidence not retrieved"
            if not r["answerable"]:
                retrieval = "unanswerable"
            print(f"  {r['id']} [{r['type']}] {r['correctness']}, faithful={r['faithful']}, {retrieval}")
            if r["notes"]:
                print(f"       note: {r['notes']}")

    out_file = RESULTS_DIR / f"{args.stem}_grades_summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
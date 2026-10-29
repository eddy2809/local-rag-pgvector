"""Ask a question to the local RAG and print the answer with its sources and timings."""

import argparse

from rag.db import get_connection
from rag.embeddings import embed_texts
from rag.generation import generate_answer, warm_up
from rag.retrieval import dense_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--table", default="chunks_300")
    parser.add_argument("--k", type=int, default=5, help="number of chunks given to the LLM")
    args = parser.parse_args()

    embed_texts(["warm-up"])
    warm_up()

    with get_connection() as conn:
        chunks, timings = dense_search(conn, args.table, args.question, args.k)
    answer = generate_answer(args.question, chunks)

    print(f"\nQuestion: {args.question}\n")
    print(f"Answer:\n{answer.text}\n")

    print("Retrieved sources (* = cited in the answer):")
    for i, c in enumerate(chunks, start=1):
        mark = "*" if i in answer.cited_ids else " "
        print(f"  [{i}]{mark} pp. {c.pages:<7} sim {c.similarity:.3f}  {c.content[:70]!r}")

    print("\nTimings:")
    print(f"  question embedding  {timings['embed'] * 1000:7.1f} ms")
    print(f"  vector search       {timings['search'] * 1000:7.1f} ms")
    print(f"  generation          {answer.seconds:7.2f} s "
          f"({answer.prompt_tokens} prompt tokens, {answer.output_tokens} output tokens, "
          f"{answer.tokens_per_second:.0f} tok/s)")


if __name__ == "__main__":
    main()
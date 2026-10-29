"""Split extracted pages into overlapping, token-sized chunks."""

import argparse
import bisect
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenizers import Tokenizer

from rag.config import TOKENIZER_NAME

WORD_START = "\u2581"  # SentencePiece marks the first piece of every word with "▁"


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    page_start: int   # physical PDF pages covered by the chunk
    page_end: int
    label_start: str  # printed page labels, used in citations
    label_end: str
    n_tokens: int
    text: str


def load_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.no_truncation()  # tokenize the whole document, not just the first 8192 tokens
    tokenizer.no_padding()
    return tokenizer


def load_pages(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def back_to_word_start(i: int, tokens: list[str], lower_bound: int) -> int:
    """Move a token index back until it points at the first piece of a word."""
    while i > lower_bound and not tokens[i].startswith(WORD_START):
        i -= 1
    return i


def chunk_document(pages: list[dict], tokenizer: Tokenizer, size: int, overlap: int) -> list[Chunk]:
    # Join the pages into one text so that chunks can cross page breaks,
    # remembering the character offset where each page starts.
    page_offsets, pos = [], 0
    for p in pages:
        page_offsets.append(pos)
        pos += len(p["text"]) + 2  # length of the "\n\n" separator
    full_text = "\n\n".join(p["text"] for p in pages)

    encoding = tokenizer.encode(full_text, add_special_tokens=False)
    tokens, offsets = encoding.tokens, encoding.offsets
    n = len(tokens)

    def page_at(char_pos: int) -> dict:
        return pages[bisect.bisect_right(page_offsets, char_pos) - 1]

    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            end = back_to_word_start(end, tokens, start + size // 2)  # never cut a word in half

        char_start, char_end = offsets[start][0], offsets[end - 1][1]
        while full_text[char_start].isspace():
            char_start += 1
        first, last = page_at(char_start), page_at(char_end - 1)

        chunks.append(Chunk(
            doc_id=pages[0]["doc_id"],
            chunk_index=len(chunks),
            page_start=first["page_number"],
            page_end=last["page_number"],
            label_start=first["page_label"],
            label_end=last["page_label"],
            n_tokens=end - start,
            text=full_text[char_start:char_end].strip(),
        ))

        if end == n:
            break
        start = back_to_word_start(end - overlap, tokens, start + 1)

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages_file", type=Path)
    parser.add_argument("--size", type=int, default=300, help="max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="tokens shared by consecutive chunks")
    args = parser.parse_args()

    pages = load_pages(args.pages_file)
    chunks = chunk_document(pages, load_tokenizer(), args.size, args.overlap)

    doc_name = args.pages_file.name.split(".")[0]
    out_file = args.pages_file.parent / f"{doc_name}.chunks{args.size}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    lengths = [c.n_tokens for c in chunks]
    spanning = sum(1 for c in chunks if c.page_start != c.page_end)
    print(f"Chunks: {len(chunks)} | tokens min {min(lengths)}, "
          f"median {statistics.median(lengths):.0f}, max {max(lengths)}")
    print(f"Chunks spanning two pages: {spanning}")

    a, b = chunks[10], chunks[11]
    print(f"\nChunk 10 (pp. {a.label_start}-{a.label_end}) ends with:   ...{a.text[-150:]!r}")
    print(f"Chunk 11 (pp. {b.label_start}-{b.label_end}) starts with: {b.text[:150]!r}...")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
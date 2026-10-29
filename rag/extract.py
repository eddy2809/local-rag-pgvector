"""Extract and clean text from PDF files, page by page."""

import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

MIN_PAGE_CHARS = 50    # pages with less text than this are treated as blank
MIN_REPEATS = 2        # an edge block found on at least this many pages is a running header/footer
MAX_EDGE_CHARS = 80    # only short blocks can be headers/footers
MIN_TOC_LEADERS = 5    # dot leaders (". . . . .") needed to classify a page as table of contents
END_MATTER = re.compile(
    r"^(bibliografia|riferimenti bibliografici|references|bibliography)$", re.IGNORECASE
)

LOWERCASE_START = r"[a-zàèéìòù]"
QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})


@dataclass
class Page:
    doc_id: str
    page_number: int  # 1-based physical index in the PDF
    page_label: str   # printed label (e.g. "iv", "12"), empty if the PDF defines none
    text: str


def _compose(letter: str, mark: str) -> str:
    letter = "i" if letter == "ı" else letter  # LaTeX puts accents on a dotless i
    return unicodedata.normalize("NFC", letter + mark)


def fix_latex_accents(text: str) -> str:
    """Rebuild accented letters that LaTeX PDFs store as a separate accent glyph.

    "realt`a" -> "realtà", "perch ́e" -> "perché", "Mu ̃noz" -> "Muñoz".
    Runs after NFKC, which turns spacing accents (´ ¨ ˜) into a space plus a combining mark.
    """
    text = re.sub(r"`([A-Za-zı])", lambda m: _compose(m[1], "\u0300"), text)
    text = re.sub(r" ?([\u0301\u0303\u0308])([A-Za-zı])", lambda m: _compose(m[2], m[1]), text)
    return text


def clean_block(text: str) -> str:
    """Normalize a text block extracted from the PDF into a single clean paragraph."""
    text = unicodedata.normalize("NFKC", text)              # ligatures: "ﬁ" -> "fi"
    text = fix_latex_accents(text)
    text = text.translate(QUOTES)                           # typographic quotes -> plain
    text = re.sub(rf"(\w)-\n(?={LOWERCASE_START})", r"\1", text)  # re-join hyphenated words
    text = re.sub(r"\s*\n\s*", " ", text)                   # line breaks inside a paragraph
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def mask_digits(text: str) -> str:
    return re.sub(r"\d+", "#", text)


def is_table_of_contents(text: str) -> bool:
    return len(re.findall(r"(?:\. ){4,}\.", text)) >= MIN_TOC_LEADERS


def read_blocks(pdf_path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (page_number, page_label, cleaned text blocks) for every page."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            blocks = [clean_block(b[4]) for b in page.get_text("blocks", sort=True) if b[6] == 0]
            pages.append((page.number + 1, page.get_label(), [b for b in blocks if b]))
    return pages


def count_edge_blocks(raw_pages: list[tuple[int, str, list[str]]]) -> Counter:
    """Count on how many pages each first/last block appears (digits masked)."""
    counts: Counter = Counter()
    for _, _, blocks in raw_pages:
        if blocks:
            counts.update({mask_digits(blocks[0]), mask_digits(blocks[-1])})
    return counts


def join_split_words(pages: list[Page]) -> None:
    """Re-join words hyphenated across a page break, keeping the word on the first page."""
    for prev, nxt in zip(pages, pages[1:]):
        if (
            nxt.page_number == prev.page_number + 1
            and re.search(r"\w-$", prev.text)
            and re.match(LOWERCASE_START, nxt.text)
        ):
            head, _, rest = nxt.text.partition(" ")
            prev.text = prev.text[:-1] + head
            nxt.text = rest


def extract_pages(pdf_path: Path) -> tuple[list[Page], Counter, Counter]:
    """Return the cleaned pages, the removed header/footer blocks and the skipped pages by reason."""
    raw_pages = read_blocks(pdf_path)
    edges = count_edge_blocks(raw_pages)
    removed: Counter = Counter()
    skipped: Counter = Counter()
    pages: list[Page] = []

    for number, label, blocks in raw_pages:
        kept = list(blocks)
        for idx in (0, -1):
            if (
                kept
                and len(kept[idx]) <= MAX_EDGE_CHARS
                and edges[mask_digits(kept[idx])] >= MIN_REPEATS
            ):
                removed[mask_digits(kept.pop(idx))] += 1

        if kept and END_MATTER.match(kept[0]):
            skipped["end matter"] += len(raw_pages) - number + 1
            break

        text = "\n\n".join(kept)
        if is_table_of_contents(text):
            skipped["table of contents"] += 1
        elif len(text) < MIN_PAGE_CHARS:
            skipped["blank"] += 1
        else:
            pages.append(Page(pdf_path.stem, number, label, text))

    join_split_words(pages)
    return pages, removed, skipped


def main() -> None:
    pdf_path = Path(sys.argv[1])
    pages, removed, skipped = extract_pages(pdf_path)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{pdf_path.stem}.pages.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    print(f"Kept pages: {len(pages)} | skipped: {dict(skipped)}")

    print("\nRemoved header/footer blocks (digits -> #):")
    for text, n in removed.most_common(15):
        print(f"  {n:4d}x  {text}")

    full_text = "\n".join(p.text for p in pages)
    accent_leftovers = len(re.findall(r"[`\u0300-\u036f]", full_text))
    hyphen_endings = sum(1 for p in pages if re.search(r"\w-$", p.text))
    print(f"\nLeftover accent artifacts: {accent_leftovers}")
    print(f"Pages still ending with a hyphen: {hyphen_endings}")

    print("\nFirst and last kept pages:")
    for p in pages[:4] + pages[-2:]:
        print(f"  p.{p.page_number:<3} label={p.page_label or '-':<4} {p.text[:100]!r}")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
"""Structure-based chunking of page markdown.

Pages are split into blocks at markdown structure (headings, blank lines, table boundaries), then
consecutive blocks are packed into chunks up to a token budget. Blocks that are larger than the
budget on their own are split at sentence boundaries, then at word boundaries as a last resort.
Every chunk keeps the page it came from, so page-level retrieval can max-pool chunk scores.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

LengthFn = Callable[[str], int]

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_ROW = re.compile(r"^\s*\|")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    page_id: int
    index: int  # position of the chunk within its page
    text: str


def split_blocks(markdown: str) -> list[str]:
    """Split markdown into structural blocks; table rows stay together, headings start a block."""
    blocks: list[str] = []
    current: list[str] = []
    in_table = False

    def flush() -> None:
        if current and any(line.strip() for line in current):
            blocks.append("\n".join(current).strip())
        current.clear()

    for line in markdown.splitlines():
        is_table = bool(_TABLE_ROW.match(line))
        if not line.strip():
            if not in_table:
                flush()
            continue
        if _HEADING.match(line) or is_table != in_table:
            flush()
        current.append(line)
        in_table = is_table
    flush()
    return blocks


def _split_oversized(block: str, max_tokens: int, length: LengthFn) -> list[str]:
    pieces: list[str] = []
    for unit in _SENTENCE_END.split(block) if "|" not in block else block.splitlines():
        if length(unit) <= max_tokens:
            pieces.append(unit)
            continue
        words = unit.split()
        part: list[str] = []
        for word in words:
            if part and length(" ".join([*part, word])) > max_tokens:
                pieces.append(" ".join(part))
                part = []
            part.append(word)
        if part:
            pieces.append(" ".join(part))
    return _pack(pieces, max_tokens, length, sep=" ")


def _pack(units: list[str], max_tokens: int, length: LengthFn, sep: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{sep}{unit}" if current else unit
        if current and length(candidate) > max_tokens:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_page(page_id: int, markdown: str, max_tokens: int, length: LengthFn) -> list[Chunk]:
    """Chunk one page. Empty pages yield no chunks (they are retrieved only by other arms)."""
    units: list[str] = []
    for block in split_blocks(markdown):
        if length(block) > max_tokens:
            units.extend(_split_oversized(block, max_tokens, length))
        else:
            units.append(block)
    texts = _pack(units, max_tokens, length, sep="\n\n")
    return [Chunk(page_id=page_id, index=i, text=t) for i, t in enumerate(texts)]

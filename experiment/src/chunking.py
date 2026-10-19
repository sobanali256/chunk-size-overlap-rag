"""
chunking.py - fixed-size token windows, identical to the pilot's chunk_doc.

Definitions (EXPERIMENT_PROTOCOL.md section 3):
  overlap_tokens = round(chunk_size * overlap_pct / 100)     (Python round, as in the pilot)
  stride         = chunk_size - overlap_tokens
  A chunk is the token interval [start, min(start + chunk_size, n)) with start = 0, stride, 2*stride, ...
  The last chunk is the first one that reaches the end of the document, so it may be shorter.
Token positions are the model tokenizer's tokens of the whole document, without special tokens.
"""
from __future__ import annotations

from dataclasses import dataclass


def overlap_tokens(size: int, pct: float) -> int:
    ov = round(size * pct / 100)
    if not 0 <= ov < size:
        raise ValueError(f"overlap {pct}% of {size} gives {ov} tokens; must be in [0, size)")
    return ov


def token_windows(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Token intervals [t0, t1). Same loop as the pilot's chunk_doc."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    stride = size - overlap
    out, start = [], 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            break
        start += stride
    return out


@dataclass(frozen=True)
class Chunk:
    paper_id: str
    idx: int          # position within the paper
    t0: int           # first token (inclusive)
    t1: int           # last token (exclusive)
    c0: int           # char start = start of token t0
    c1: int           # char end = end of token t1 - 1

    @property
    def n_tokens(self) -> int:
        return self.t1 - self.t0


def chunk_paper(paper_id: str, offsets: list[tuple[int, int]], size: int, overlap: int) -> list[Chunk]:
    return [Chunk(paper_id, i, t0, t1, offsets[t0][0], offsets[t1 - 1][1])
            for i, (t0, t1) in enumerate(token_windows(len(offsets), size, overlap))]

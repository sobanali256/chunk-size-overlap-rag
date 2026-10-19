"""
metrics.py - per-question retrieval metrics on source-document token positions.

Every quantity is defined on token positions of the paper's tokenization (the tokenizer that
also sets chunk boundaries), never on characters, so repeated text from overlapping chunks is
counted once. See EXPERIMENT_PROTOCOL.md section 5 for the frozen definitions.

  E_j    token positions of gold evidence span j: tokens whose character range intersects the
         span's character range. E = union of E_j.
  chunk  token interval [t0, t1).
  R      the top-K retrieved chunks; U(R) = union of their token positions.

  hit          1 if U(R) ∩ E is non-empty
  complete     1 if for every span j there is a chunk c in R with E_j ⊆ c
  ev_recall    |U(R) ∩ E| / |E|
  completable  1 if for every span j there is a chunk c in the paper's index with E_j ⊆ c
  random_hit   probability that K chunks drawn uniformly without replacement from the paper's
               n chunks include at least one chunk that overlaps E: 1 - C(n-r, k)/C(n, k),
               k = min(K, n), r = number of chunks overlapping E. Exact, no sampling.
"""
from __future__ import annotations

import bisect
import math

from .chunking import Chunk


def span_token_set(offsets: list[tuple[int, int]], s: int, e: int) -> range:
    """Token indices whose char range [a, b) intersects [s, e). Contiguous, so a range."""
    if e <= s:
        raise ValueError("empty evidence span")
    lo = bisect.bisect_right([o[1] for o in offsets], s)   # first token ending after s
    hi = bisect.bisect_left([o[0] for o in offsets], e)    # first token starting at or after e
    if lo >= hi:
        raise ValueError(f"no token covers evidence span [{s}, {e})")
    return range(lo, hi)


def _contains(c: Chunk, r: range) -> bool:
    return c.t0 <= r.start and r.stop <= c.t1


def _overlaps(c: Chunk, r: range) -> bool:
    return c.t0 < r.stop and r.start < c.t1


def _union_size(intervals) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0)


def _intersection_size(a_intervals, b_intervals) -> int:
    """|union(a) ∩ union(b)| for half-open integer intervals."""
    a = _merge(a_intervals)
    b = _merge(b_intervals)
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def completable(chunks: list[Chunk], ev: list[range]) -> bool:
    return all(any(_contains(c, r) for c in chunks) for r in ev)


def score_question(ranked: list[Chunk], ev: list[range], k: int) -> dict:
    """ranked: all chunks of the question's paper, best first. ev: token ranges of the gold spans."""
    if not ev:
        raise ValueError("question has no evidence spans")
    if not ranked:
        raise ValueError("paper has no chunks")
    top = ranked[:k]
    ev_iv = [(r.start, r.stop) for r in ev]
    top_iv = [(c.t0, c.t1) for c in top]
    rel = [any(_overlaps(c, r) for r in ev) for c in ranked]

    n, n_rel, kk = len(ranked), sum(rel), min(k, len(ranked))
    first = next((i for i, x in enumerate(rel, 1) if x), None)
    ev_unique = _union_size(ev_iv)
    ev_in_top = _intersection_size(top_iv, ev_iv)
    retrieved = sum(c.n_tokens for c in top)
    retrieved_unique = _union_size(top_iv)
    return {
        "hit": float(any(rel[:k])),
        "complete": float(all(any(_contains(c, r) for c in top) for r in ev)),
        "ev_recall": ev_in_top / ev_unique,
        "completable": float(completable(ranked, ev)),
        "random_hit": 1 - math.comb(n - n_rel, kk) / math.comb(n, kk),
        "first_rel_rank": first,
        "rr": 1.0 / first if first and first <= k else 0.0,
        "k": k,
        "k_effective": kk,
        "n_candidates": n,
        "n_relevant_chunks": n_rel,
        "evidence_tokens": ev_unique,
        "retrieved_tokens": retrieved,               # tokens sent, counting overlap twice
        "retrieved_unique_tokens": retrieved_unique, # distinct source positions
        "evidence_precision_unique": ev_in_top / retrieved_unique if retrieved_unique else 0.0,
        "redundancy": 1 - retrieved_unique / retrieved if retrieved else 0.0,
    }

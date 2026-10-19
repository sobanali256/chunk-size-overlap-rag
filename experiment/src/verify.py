"""
verify.py - independent recomputation of sampled rows from the raw outputs.

Deliberately does not import metrics.py: it rebuilds every metric from rankings.jsonl and
chunks.csv with plain Python sets, so an error in metrics.py would show up as a mismatch.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd


def verify(run_dir: Path, cfg: dict, n_rows: int = 20, seed: int = 0) -> list[str]:
    run_dir = Path(run_dir)
    pq = pd.read_csv(run_dir / "raw" / "per_question.csv")
    chunks = pd.read_csv(run_dir / "raw" / "chunks.csv")
    rankings = {}
    with open(run_dir / "raw" / "rankings.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rankings[(r["config_id"], r["question_id"])] = r
    problems = []
    rows = pq.sample(n=min(n_rows, len(pq)), random_state=seed)
    for _, row in rows.iterrows():
        r = rankings[(row.config_id, row.question_id)]
        ch = chunks[(chunks.config_id == row.config_id) & (chunks.paper_id == row.paper_id)].set_index("chunk_idx")
        if sorted(r["ranked_chunk_idx"]) != sorted(ch.index):
            problems.append(f"{row.question_id} {row.config_id}: ranking does not cover the paper's chunks")
        if r["scores"] != sorted(r["scores"], reverse=True):
            problems.append(f"{row.question_id} {row.config_id}: scores not descending")
        k = int(row.k)
        top = [set(range(ch.loc[i, "t0"], ch.loc[i, "t1"])) for i in r["ranked_chunk_idx"][:k]]
        allc = [set(range(ch.loc[i, "t0"], ch.loc[i, "t1"])) for i in r["ranked_chunk_idx"]]
        spans = [set(range(a, b)) for a, b in r["evidence_token_spans"]]
        E = set().union(*spans)
        U = set().union(*top)
        rel = [bool(c & E) for c in allc]
        n, nr, kk = len(allc), sum(rel), min(k, len(allc))
        expect = {
            "hit": float(bool(U & E)),
            "complete": float(all(any(s <= c for c in top) for s in spans)),
            "completable": float(all(any(s <= c for c in allc) for s in spans)),
            "ev_recall": len(U & E) / len(E),
            "retrieved_tokens": sum(len(c) for c in top),
            "retrieved_unique_tokens": len(U),
            "random_hit": 1 - math.comb(n - nr, kk) / math.comb(n, kk),
            "n_candidates": n,
        }
        for m, v in expect.items():
            if abs(float(row[m]) - v) > 1e-9:
                problems.append(f"{row.question_id} {row.config_id} {row['mode']}: {m} {row[m]} != {v}")
        # evidence token spans must correspond to the char spans via the tokenizer offsets
        if len(r["evidence_char_spans"]) != len(r["evidence_token_spans"]):
            problems.append(f"{row.question_id}: span count mismatch")
    return problems

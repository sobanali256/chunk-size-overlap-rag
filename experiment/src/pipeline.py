"""
pipeline.py - load a sample, chunk, embed, retrieve and score every question in every configuration.

Writes raw outputs only (no aggregation): see EXPERIMENT_PROTOCOL.md section 8 for the layout.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import chunk_paper, overlap_tokens
from .embeddings import cached_encode
from .metrics import score_question, span_token_set
from .retrieval import k_for, rank_paper


@dataclass
class EvalQuestion:
    paper_id: str
    question_id: str
    question: str
    spans: list[tuple[int, int]]    # char ranges in the paper text
    type: str


def config_id(size: int, pct: float) -> str:
    return f"s{size}_o{int(pct)}"


def run(cfg: dict, papers: dict[str, str], questions: list[EvalQuestion], emb, tokenizer,
        out: Path, cache_dir: Path | None, sample_name: str, extra_meta: dict | None = None) -> dict:
    out = Path(out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    paper_ids = sorted(papers)
    if not questions:
        raise ValueError("no questions")
    missing = {q.paper_id for q in questions} - set(papers)
    if missing:
        raise ValueError(f"questions refer to missing papers: {sorted(missing)[:5]}")

    offsets = {pid: tokenizer.offsets(papers[pid]) for pid in paper_ids}
    ev_tokens = {q.question_id: [span_token_set(offsets[q.paper_id], s, e) for s, e in q.spans]
                 for q in questions}
    for q in questions:   # spans must be exact substrings located in the text
        for s, e in q.spans:
            if not (0 <= s < e <= len(papers[q.paper_id])):
                raise ValueError(f"{q.question_id}: bad span {(s, e)}")

    qp, dp = cfg["embedding"]["query_prefix"], cfg["embedding"]["doc_prefix"]
    Q, q_ntok, q_cached = cached_encode(emb, [q.question for q in questions], qp, cache_dir, "queries")
    modes = cfg["modes"]

    per_q_rows, eng_rows, chunk_rows = [], [], []
    ranking_f = open(raw / "rankings.jsonl", "w", encoding="utf-8")
    for size in cfg["chunking"]["sizes"]:
        for pct in cfg["chunking"]["overlap_pct"]:
            cid = config_id(size, pct)
            ov = overlap_tokens(size, pct)
            chunks = {pid: chunk_paper(pid, offsets[pid], size, ov) for pid in paper_ids}
            flat = [c for pid in paper_ids for c in chunks[pid]]
            texts = [papers[c.paper_id][c.c0:c.c1] for c in flat]
            t0 = time.time()
            C, in_tok, cached = cached_encode(emb, texts, dp, cache_dir, f"chunks_{cid}")
            embed_s = time.time() - t0
            row_of = {}
            for i, c in enumerate(flat):
                row_of[(c.paper_id, c.idx)] = i
                chunk_rows.append({"config_id": cid, "paper_id": c.paper_id, "chunk_idx": c.idx,
                                   "t0": c.t0, "t1": c.t1, "c0": c.c0, "c1": c.c1,
                                   "n_tokens": c.n_tokens, "model_input_tokens": in_tok[i]})
            lens = np.array([c.n_tokens for c in flat])
            total_doc_tokens = sum(len(offsets[p]) for p in paper_ids)
            eng_rows.append({
                "config_id": cid, "chunk_size": size, "overlap_pct": pct, "overlap_tokens": ov,
                "stride": size - ov, "n_chunks": len(flat),
                "chunk_len_mean": float(lens.mean()), "chunk_len_median": float(np.median(lens)),
                "chunk_len_p5": float(np.percentile(lens, 5)), "chunk_len_min": int(lens.min()),
                "chunks_shorter_than_size": int((lens < size).sum()),
                "max_model_input_tokens": int(max(in_tok)),
                "doc_tokens": total_doc_tokens, "indexed_tokens": int(lens.sum()),
                "duplicate_token_fraction": 1 - total_doc_tokens / lens.sum(),
                "index_bytes_float32": int(C.shape[0] * C.shape[1] * 4),
                "embed_seconds": None if cached else round(embed_s, 1), "cache_hit": cached})

            row_idx = {pid: np.array([row_of[(pid, c.idx)] for c in chunks[pid]]) for pid in paper_ids}
            for qi, q in enumerate(questions):
                pc = chunks[q.paper_id]
                order, scores = rank_paper(Q[qi], C[row_idx[q.paper_id]])
                ranked = [pc[i] for i in order]
                ranking_f.write(json.dumps({
                    "config_id": cid, "paper_id": q.paper_id, "question_id": q.question_id,
                    "ranked_chunk_idx": [int(i) for i in order],
                    "scores": [round(float(scores[i]), 6) for i in order],
                    "evidence_char_spans": q.spans,
                    "evidence_token_spans": [[r.start, r.stop] for r in ev_tokens[q.question_id]]}) + "\n")
                for mode_name, mode in modes.items():
                    k = k_for(mode, size)
                    m = score_question(ranked, ev_tokens[q.question_id], k)
                    per_q_rows.append({"sample": sample_name, "mode": mode_name, "config_id": cid,
                                       "chunk_size": size, "overlap_pct": pct, "paper_id": q.paper_id,
                                       "question_id": q.question_id, "type": q.type,
                                       "n_spans": len(q.spans),
                                       "top_chunk_idx": " ".join(str(c.idx) for c in ranked[:k]),
                                       **m})
            print(f"{cid}: {len(flat)} chunks, embed {'cached' if cached else f'{embed_s:.0f}s'}",
                  flush=True)
    ranking_f.close()

    _write_csv(raw / "per_question.csv", per_q_rows)
    _write_csv(raw / "chunks.csv", chunk_rows)
    _write_csv(raw / "engineering.csv", eng_rows)
    with open(raw / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.__dict__, ensure_ascii=False) + "\n")
    meta = {"sample": sample_name, "n_papers": len(paper_ids), "n_questions": len(questions),
            "config": cfg, "versions": getattr(emb, "versions", {}), "python": sys.version,
            "platform": platform.platform(), "argv": sys.argv, "query_cache_hit": q_cached,
            "max_query_input_tokens": int(max(q_ntok)), "wall_seconds": round(time.time() - t_start, 1),
            **(extra_meta or {})}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

"""
verify_all_fields.py - independent recomputation of every reported field of a run.

Added after the fresh_v1 run (see EXPERIMENT_PROTOCOL.md section 10, operational log). It changes
no result and is not one of the frozen files. The committed src/verify.py checks a sample of rows
and only part of the per-question fields; this script checks every row and every field, and the
aggregates and statistics built from them.

Like src/verify.py, it does not import src/metrics.py, src/chunking.py, src/statistics.py or
src/reporting.py: everything is rebuilt with plain Python sets and pandas. The only project code it
uses is src/data.build_doc (the pilot's document builder), and the rebuilt documents are checked
against the SHA-256 recorded in the frozen paper manifest.

Checks:
  A  questions.jsonl and rankings.jsonl agree on evidence character spans; documents match the
     manifest hashes; spans lie inside the document
  B  evidence token spans = tokens whose character range intersects the span (needs the tokenizer)
  C  chunks.csv: token windows rebuilt from the chunking rule; character offsets from the tokenizer
  D  rankings: a permutation of the paper's chunks, with scores descending
  E  per_question.csv: every column of every row, both modes (including sample, which must equal the
     run's sample in run_metadata.json)
  F  aggregated/per_paper.csv: per-paper means and question counts; per_paper_canonical_ids.csv, if
     present (written by export_canonical_ids.py), matches it with canonical paper IDs
  G  aggregated/summary.csv and raw/engineering.csv: means over papers, token quantiles, index measures
  H  aggregated/failure_categories.csv: category counts
  I  statistics/primary_comparisons.csv: means, differences, paper counts and Holm exactly; p-values
     and bootstrap CIs (also the summary CIs) by an independent Monte Carlo re-estimate with
     different seeds, within a tolerance

Usage (from experiment/):
  HF_HUB_OFFLINE=1 python verify_all_fields.py --run results/fresh_v1
  python verify_all_fields.py --run results/fresh_v1 --no-tokenizer   # skips B and C's char offsets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import build_doc  # noqa: E402

TOL = 1e-9
PER_Q_FIELDS = ["hit", "complete", "ev_recall", "completable", "random_hit", "first_rel_rank", "rr",
                "k", "k_effective", "n_candidates", "n_relevant_chunks", "evidence_tokens",
                "retrieved_tokens", "retrieved_unique_tokens", "evidence_precision_unique", "redundancy"]
METRICS = ["complete", "hit", "ev_recall", "completable", "random_hit", "rr",
           "evidence_precision_unique", "redundancy", "retrieved_tokens", "retrieved_unique_tokens"]
CI_METRICS = ["complete", "hit", "ev_recall"]


class Report:
    def __init__(self):
        self.checked, self.problems, self.warnings = {}, [], []

    def eq(self, section, what, got, want, tol=TOL):
        self.checked[section] = self.checked.get(section, 0) + 1
        if isinstance(want, float) or isinstance(got, float):
            ok = (pd.isna(got) and pd.isna(want)) or abs(float(got) - float(want)) <= tol
        else:
            ok = got == want
        if not ok:
            self.problems.append(f"[{section}] {what}: reported {got!r}, recomputed {want!r}")


def k_of(mode: dict, size: int) -> int:
    return mode["budget"] // size if mode["type"] == "token_budget" else int(mode["k"])


def windows(n: int, size: int, overlap: int):
    stride, out, start = size - overlap, [], 0
    while True:
        out.append((start, min(start + size, n)))
        if start + size >= n:
            return out
        start += stride


def mc_perm_p(d: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    t_obs, count = abs(d.mean()), 0
    for _ in range(n_perm // 10_000):
        s = rng.integers(0, 2, size=(10_000, len(d))) * 2 - 1
        count += int((np.abs((s * d).mean(axis=1)) >= t_obs - 1e-12).sum())
    return (1 + count) / (1 + n_perm)


def boot_ci(x: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-tokenizer", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    here = Path(__file__).resolve().parent
    cfg = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    R = Report()

    pq = pd.read_csv(run / "raw" / "per_question.csv", dtype={"paper_id": str, "sample": str})
    ch = pd.read_csv(run / "raw" / "chunks.csv", dtype={"paper_id": str})
    eng = pd.read_csv(run / "raw" / "engineering.csv")
    qs = {}
    with open(run / "raw" / "questions.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qs[q["question_id"]] = q
    rk = {}
    with open(run / "raw" / "rankings.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rk[(r["config_id"], r["question_id"])] = r

    # ---------------------------------------------------------------- A: inputs
    data = json.loads((here / cfg["dataset"]["file"]).read_text(encoding="utf-8"))
    manifest = pd.read_csv(here / "manifests" / "paper_manifest.csv", dtype={"paper_id": str}).set_index("paper_id")
    papers = sorted({q["paper_id"] for q in qs.values()})
    docs = {pid: build_doc(data[pid])[0] for pid in papers}
    for pid, text in docs.items():
        R.eq("A", f"{pid} doc sha256", manifest.loc[pid, "doc_sha256"],
             hashlib.sha256(text.encode("utf-8")).hexdigest())
    for (cid, qid), r in rk.items():
        R.eq("A", f"{qid} {cid} char spans", [list(s) for s in qs[qid]["spans"]], r["evidence_char_spans"])
        for s, e in r["evidence_char_spans"]:
            R.eq("A", f"{qid} span in bounds", True, 0 <= s < e <= len(docs[qs[qid]["paper_id"]]))

    # ------------------------------------------------------- B, C: tokens and chunks
    n_tok = {}
    offsets = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"], revision=cfg["tokenizer"]["revision"])
        offsets = {pid: [tuple(o) for o in tok(docs[pid], add_special_tokens=False, return_offsets_mapping=True,
                                                truncation=False, verbose=False)["offset_mapping"]]
                   for pid in papers}
        n_tok = {pid: len(o) for pid, o in offsets.items()}
        for (cid, qid), r in rk.items():
            off = offsets[qs[qid]["paper_id"]]
            for (s, e), (t0, t1) in zip(r["evidence_char_spans"], r["evidence_token_spans"]):
                idx = [i for i, (a, b) in enumerate(off) if b > s and a < e]
                R.eq("B", f"{qid} {cid} token span", [t0, t1], [idx[0], idx[-1] + 1])
    else:
        n_tok = ch.groupby("paper_id").t1.max().to_dict()

    sizes, pcts = cfg["chunking"]["sizes"], cfg["chunking"]["overlap_pct"]
    cid_of = {f"s{s}_o{int(p)}": (s, p) for s in sizes for p in pcts}
    for (cid, pid), g in ch.groupby(["config_id", "paper_id"]):
        size, pct = cid_of[cid]
        want = windows(n_tok[pid], size, round(size * pct / 100))
        g = g.sort_values("chunk_idx")
        R.eq("C", f"{cid} {pid} chunk_idx", g.chunk_idx.tolist(), list(range(len(want))))
        R.eq("C", f"{cid} {pid} windows", list(zip(g.t0, g.t1)), want)
        R.eq("C", f"{cid} {pid} n_tokens", g.n_tokens.tolist(), [b - a for a, b in want])
        if offsets is not None:
            off = offsets[pid]
            R.eq("C", f"{cid} {pid} char offsets", list(zip(g.c0, g.c1)),
                 [(off[a][0], off[b - 1][1]) for a, b in want])
    R.eq("C", "configurations in chunks.csv", sorted(ch.config_id.unique()), sorted(cid_of))

    # ------------------------------------------------------------- D, E: per question
    iv = {(c, p, i): (a, b) for c, p, i, a, b in ch[["config_id", "paper_id", "chunk_idx", "t0", "t1"]]
          .itertuples(index=False)}
    mode_cfg = cfg["modes"]
    for (cid, qid), r in rk.items():
        R.eq("D", f"{qid} {cid} ranking is a permutation", sorted(r["ranked_chunk_idx"]),
             list(range(len(r["ranked_chunk_idx"]))))
        R.eq("D", f"{qid} {cid} scores descending", r["scores"], sorted(r["scores"], reverse=True))
    per_paper_n = ch.groupby(["config_id", "paper_id"]).size().to_dict()
    expected_rows = {(m, c, q) for m in mode_cfg for c in cid_of for q in qs}
    R.eq("E", "row set", set(zip(pq["mode"], pq.config_id, pq.question_id)) == expected_rows, True)
    R.eq("E", "row count", len(pq), len(expected_rows))
    for row in pq.itertuples(index=False):
        r = rk[(row.config_id, row.question_id)]
        size, pct = cid_of[row.config_id]
        R.eq("E", f"{row.question_id} {row.config_id} sample/size/overlap/paper/type/n_spans",
             (row.sample, row.chunk_size, row.overlap_pct, row.paper_id, row.type, row.n_spans),
             (meta["sample"], size, pct, qs[row.question_id]["paper_id"], qs[row.question_id]["type"],
              len(r["evidence_char_spans"])))
        R.eq("D", f"{row.question_id} {row.config_id} ranking covers paper",
             len(r["ranked_chunk_idx"]), per_paper_n[(row.config_id, row.paper_id)])
        k = k_of(mode_cfg[row.mode], size)
        allc = [set(range(*iv[(row.config_id, row.paper_id, i)])) for i in r["ranked_chunk_idx"]]
        top = allc[:k]
        spans = [set(range(a, b)) for a, b in r["evidence_token_spans"]]
        E = set().union(*spans)
        U = set().union(*top)
        rel = [bool(c & E) for c in allc]
        n, nr, kk = len(allc), sum(rel), min(k, len(allc))
        first = next((i + 1 for i, x in enumerate(rel) if x), None)
        sent = sum(len(c) for c in top)
        want = {
            "hit": float(bool(U & E)),
            "complete": float(all(any(s <= c for c in top) for s in spans)),
            "completable": float(all(any(s <= c for c in allc) for s in spans)),
            "ev_recall": len(U & E) / len(E),
            "random_hit": 1 - math.comb(n - nr, kk) / math.comb(n, kk),
            "first_rel_rank": first,
            "rr": 1 / first if first and first <= k else 0.0,
            "k": k, "k_effective": kk, "n_candidates": n, "n_relevant_chunks": nr,
            "evidence_tokens": len(E), "retrieved_tokens": sent, "retrieved_unique_tokens": len(U),
            "evidence_precision_unique": len(U & E) / len(U) if U else 0.0,
            "redundancy": 1 - len(U) / sent if sent else 0.0,
        }
        for f in PER_Q_FIELDS:
            got = getattr(row, f)
            if want[f] is None:
                R.eq("E", f"{row.question_id} {row.config_id} {row.mode} {f}", pd.isna(got), True)
            else:
                R.eq("E", f"{row.question_id} {row.config_id} {row.mode} {f}", float(got), float(want[f]))
        R.eq("E", f"{row.question_id} {row.config_id} {row.mode} top_chunk_idx",
             str(row.top_chunk_idx).split(), [str(i) for i in r["ranked_chunk_idx"][:k]])

    # ---------------------------------------------------------------- F: per paper
    pp = pd.read_csv(run / "aggregated" / "per_paper.csv", dtype={"paper_id": str})
    mine = pq.groupby(["mode", "config_id", "paper_id"]).agg(**{m: (m, "mean") for m in METRICS},
                                                            n_questions=("question_id", "size"))
    R.eq("F", "row count", len(pp), len(mine))
    # reporting.py reads per_question.csv without a string dtype, so IDs such as 1604.00400 are
    # written as 1604.004. Map them back (the mapping is checked to be one-to-one) and warn.
    canon = {float(pid): pid for pid in papers}
    R.eq("F", "paper ids distinct as numbers", len(canon), len(papers))
    mangled = sorted({pid for pid in pp.paper_id if pid not in docs})
    if mangled:
        R.warnings.append(f"[F] per_paper.csv: {len(mangled)} paper ids lost trailing zeros "
                          f"(e.g. {mangled[0]} for {canon[float(mangled[0])]}); values are unaffected")
    pp["paper_id"] = [canon[float(pid)] for pid in pp.paper_id]
    fixed_path = run / "aggregated" / "per_paper_canonical_ids.csv"
    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path, dtype={"paper_id": str})
        R.eq("F", "per_paper_canonical_ids.csv ids are canonical", set(fixed.paper_id) <= set(docs), True)
        R.eq("F", "per_paper_canonical_ids.csv equals per_paper.csv with canonical ids", fixed.equals(pp), True)
    for row in pp.itertuples(index=False):
        w = mine.loc[(row.mode, row.config_id, row.paper_id)]
        for m in METRICS + ["n_questions"]:
            R.eq("F", f"{row.mode} {row.config_id} {row.paper_id} {m}", float(getattr(row, m)), float(w[m]))

    # ----------------------------------------------------- G: summary and engineering
    sm = pd.read_csv(run / "aggregated" / "summary.csv")
    st = cfg["statistics"]
    nb = st["bootstrap"]["n_resamples"]
    ci_tol = 0.01
    for row in sm.itertuples(index=False):
        g = mine.loc[row.mode].loc[row.config_id]
        qg = pq[(pq["mode"] == row.mode) & (pq.config_id == row.config_id)]
        key = f"{row.mode} {row.config_id}"
        R.eq("G", f"{key} n_papers", int(row.n_papers), len(g))
        R.eq("G", f"{key} n_questions", int(row.n_questions), len(qg))
        R.eq("G", f"{key} k", int(row.k), k_of(mode_cfg[row.mode], row.chunk_size))
        for m in METRICS:
            R.eq("G", f"{key} {m}", float(getattr(row, m)), float(g[m].mean()))
        t = qg.retrieved_tokens
        for col, v in [("tokens_q_min", t.min()), ("tokens_q_p5", t.quantile(.05)), ("tokens_q_median", t.median()),
                       ("tokens_q_p95", t.quantile(.95)), ("tokens_q_max", t.max()),
                       ("questions_with_fewer_chunks_than_k", int((qg.n_candidates < qg.k).sum()))]:
            R.eq("G", f"{key} {col}", float(getattr(row, col)), float(v))
        for i, m in enumerate(CI_METRICS):
            lo, hi = boot_ci(g[m].to_numpy(), nb, seed=900_000 + i)
            R.eq("I", f"{key} {m}_ci_low (MC tolerance {ci_tol})", float(getattr(row, f"{m}_ci_low")), lo, ci_tol)
            R.eq("I", f"{key} {m}_ci_high (MC tolerance {ci_tol})", float(getattr(row, f"{m}_ci_high")), hi, ci_tol)
    total_doc = sum(n_tok[p] for p in papers)
    for row in eng.itertuples(index=False):
        size, pct = cid_of[row.config_id]
        ov = round(size * pct / 100)
        g = ch[ch.config_id == row.config_id]
        lens = g.n_tokens
        for col, v in [("overlap_tokens", ov), ("stride", size - ov), ("n_chunks", len(g)),
                       ("chunk_len_mean", lens.mean()), ("chunk_len_median", lens.median()),
                       ("chunk_len_p5", lens.quantile(.05)), ("chunk_len_min", lens.min()),
                       ("chunks_shorter_than_size", int((lens < size).sum())),
                       ("doc_tokens", total_doc), ("indexed_tokens", int(lens.sum())),
                       ("duplicate_token_fraction", 1 - total_doc / lens.sum()),
                       ("index_bytes_float32", len(g) * cfg["embedding"]["dimension"] * 4)]:
            R.eq("G", f"{row.config_id} {col}", float(getattr(row, col)), float(v))

    # ---------------------------------------------------------- H: failure categories
    fc = pd.read_csv(run / "aggregated" / "failure_categories.csv")
    cat = np.select([pq.complete == 1, pq.completable == 0, pq.hit == 0], [
        "complete", "unfit_no_chunk_can_contain", "ranked_out_no_overlap_in_top_k"],
        default="partial_overlap_but_incomplete")
    counts = pd.Series(cat).groupby([pq["mode"], pq.config_id]).value_counts()
    for row in fc.itertuples(index=False):
        for c in ["complete", "unfit_no_chunk_can_contain", "ranked_out_no_overlap_in_top_k",
                  "partial_overlap_but_incomplete"]:
            R.eq("H", f"{row.mode} {row.config_id} {c}", int(getattr(row, c)),
                 int(counts.get((row.mode, row.config_id, c), 0)))

    # ---------------------------------------------------------------- I: statistics
    comp = pd.read_csv(run / "statistics" / "primary_comparisons.csv")
    pm = mine.loc[st["primary_mode"]][st["primary_metric"]]
    R.eq("I", "comparison pairs", [[a, b] for a, b in zip(comp.a, comp.b)], st["comparisons"])
    n_perm = st["n_permutations"]
    for i, row in enumerate(comp.itertuples(index=False)):
        a, b = pm.loc[row.a], pm.loc[row.b]
        R.eq("I", f"{row.a}-{row.b} same papers", sorted(a.index) == sorted(b.index), True)
        d = (a - b.loc[a.index]).to_numpy()
        key = f"{row.a} - {row.b}"
        for col, v in [("mean_a", a.mean()), ("mean_b", b.mean()), ("diff", d.mean()), ("n_papers", len(d)),
                       ("papers_a_better", int((d > 0).sum())), ("papers_b_better", int((d < 0).sum())),
                       ("papers_equal", int((d == 0).sum()))]:
            R.eq("I", f"{key} {col}", float(getattr(row, col)), float(v))
        p = mc_perm_p(d, n_perm, seed=800_000 + i)
        se = math.sqrt(max(p * (1 - p), 1 / n_perm) / n_perm)
        R.eq("I", f"{key} p_raw (MC tolerance 5 SE)", float(row.p_raw), p, 5 * se + 1e-5)
        lo, hi = boot_ci(d, nb, seed=700_000 + i)
        R.eq("I", f"{key} ci_low (MC tolerance {ci_tol})", float(row.ci_low), lo, ci_tol)
        R.eq("I", f"{key} ci_high (MC tolerance {ci_tol})", float(row.ci_high), hi, ci_tol)
    # Holm, recomputed exactly from the reported raw p-values
    p = comp.p_raw.to_numpy()
    order, m, run_max, adj = np.argsort(p, kind="stable"), len(p), 0.0, np.empty(len(p))
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * p[i])
        adj[i] = min(1.0, run_max)
    for row, v in zip(comp.itertuples(index=False), adj):
        R.eq("I", f"{row.a} - {row.b} p_holm", float(row.p_holm), float(v))

    # ------------------------------------------------------------------- report
    for s in "ABCDEFGHI":
        print(f"{s}: {R.checked.get(s, 0):>7} checks")
    print(f"total: {sum(R.checked.values())} checks, {len(R.problems)} mismatches")
    for p in R.problems[:50]:
        print("  " + p)
    for w in R.warnings:
        print("warning: " + w)
    sys.exit(1 if R.problems else 0)


if __name__ == "__main__":
    main()

"""
reporting.py - paper-level aggregation, the pre-specified statistics, tables and figures.

Reads only <run>/raw/*. Writes <run>/aggregated, <run>/statistics, <run>/tables, <run>/figures.
Every number in a table is computed here from the raw per-question file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import bootstrap_mean_ci, compare, holm, paper_means

METRICS = ["complete", "hit", "ev_recall", "completable", "random_hit", "rr",
           "evidence_precision_unique", "redundancy", "retrieved_tokens", "retrieved_unique_tokens"]
CI_METRICS = ["complete", "hit", "ev_recall"]


def analyze(run_dir: Path, cfg: dict) -> dict:
    run_dir = Path(run_dir)
    pq = pd.read_csv(run_dir / "raw" / "per_question.csv")
    eng = pd.read_csv(run_dir / "raw" / "engineering.csv")
    st = cfg["statistics"]
    bs = st["bootstrap"]
    for d in ("aggregated", "statistics", "tables", "figures"):
        (run_dir / d).mkdir(exist_ok=True)
    _integrity(pq, cfg)

    # ---------------------------------------------------------------- aggregation
    pp = (pq.groupby(["mode", "config_id", "chunk_size", "overlap_pct", "paper_id"])[METRICS]
          .mean().reset_index())
    nq = pq.groupby(["mode", "config_id", "paper_id"]).size().rename("n_questions").reset_index()
    pp = pp.merge(nq, on=["mode", "config_id", "paper_id"], validate="one_to_one")
    pp.to_csv(run_dir / "aggregated" / "per_paper.csv", index=False)

    rows = []
    for (mode, cid), g in pp.groupby(["mode", "config_id"], sort=False):
        qg = pq[(pq["mode"] == mode) & (pq.config_id == cid)]
        r = {"mode": mode, "config_id": cid, "chunk_size": int(g.chunk_size.iloc[0]),
             "overlap_pct": g.overlap_pct.iloc[0], "k": int(qg.k.iloc[0]),
             "n_papers": len(g), "n_questions": len(qg)}
        for m in METRICS:
            r[m] = float(g[m].mean())
        for m in CI_METRICS:
            r[f"{m}_ci_low"], r[f"{m}_ci_high"] = bootstrap_mean_ci(g[m].to_numpy(), bs["n_resamples"], bs["seed"])
        t = qg.retrieved_tokens
        r.update({"tokens_q_min": int(t.min()), "tokens_q_p5": float(t.quantile(.05)),
                  "tokens_q_median": float(t.median()), "tokens_q_p95": float(t.quantile(.95)),
                  "tokens_q_max": int(t.max()),
                  "questions_with_fewer_chunks_than_k": int((qg.n_candidates < qg.k).sum())})
        rows.append(r)
    summary = pd.DataFrame(rows).merge(eng, on=["config_id", "chunk_size", "overlap_pct"])
    summary.to_csv(run_dir / "aggregated" / "summary.csv", index=False)

    # ------------------------------------------------------ pre-specified tests
    prim = pq[pq["mode"] == st["primary_mode"]]
    pm = paper_means(prim, st["primary_metric"])
    comps = [compare(pm, a, b, bs["n_resamples"], st["n_permutations"], st["permutation_seed"])
             for a, b in st["comparisons"]]
    for c, p in zip(comps, holm([c["p_raw"] for c in comps])):
        c["p_holm"] = p
        c["metric"] = st["primary_metric"]
        c["mode"] = st["primary_mode"]
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(run_dir / "statistics" / "primary_comparisons.csv", index=False)

    # ------------------------------------- pre-defined failure categories (exploratory)
    pq["failure"] = np.select(
        [pq.complete == 1, pq.completable == 0, pq.hit == 0],
        ["complete", "unfit_no_chunk_can_contain", "ranked_out_no_overlap_in_top_k"],
        "partial_overlap_but_incomplete")
    fc = (pq.groupby(["mode", "config_id", "failure"]).size().unstack(fill_value=0).reset_index())
    fc.to_csv(run_dir / "aggregated" / "failure_categories.csv", index=False)

    _tables(run_dir, summary, comp_df, cfg)
    _figures(run_dir, summary, comp_df, cfg)
    return {"n_papers": int(summary.n_papers.iloc[0]), "n_questions": int(summary.n_questions.iloc[0])}


def _integrity(pq: pd.DataFrame, cfg: dict):
    """Fail loudly on missing configs, missing questions or duplicates."""
    n_cfg = len(cfg["chunking"]["sizes"]) * len(cfg["chunking"]["overlap_pct"])
    for mode, g in pq.groupby("mode"):
        if g.config_id.nunique() != n_cfg:
            raise ValueError(f"{mode}: {g.config_id.nunique()} configs, expected {n_cfg}")
        counts = g.groupby("config_id").question_id.agg(["size", "nunique"])
        if (counts["size"] != counts["nunique"]).any():
            raise ValueError(f"{mode}: duplicate question rows")
        if counts["nunique"].nunique() != 1:
            raise ValueError(f"{mode}: configs have different question sets")
    if set(pq["mode"]) != set(cfg["modes"]):
        raise ValueError("modes in results differ from config")


def _fmt_ci(r, m):
    return f"{r[m]:.3f} ({r[m + '_ci_low']:.3f}–{r[m + '_ci_high']:.3f})"


def _tables(run_dir, summary, comp, cfg):
    for mode in cfg["modes"]:
        s = summary[summary["mode"] == mode].sort_values(["chunk_size", "overlap_pct"])
        n_p, n_q = int(s.n_papers.iloc[0]), int(s.n_questions.iloc[0])
        lines = [f"Mode `{mode}` ({cfg['modes'][mode]}). {n_p} papers, {n_q} questions. "
                 "Values are means of per-paper means; 95% CIs: percentile bootstrap over papers.", "",
                 "| Size | Overlap | K | Complete@K (95% CI) | Hit@K (95% CI) | Evidence recall@K (95% CI) | "
                 "Completable | Random Hit@K | Retrieved tokens: median [min–max] | Chunks | Dup. index tokens |",
                 "|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|"]
        for _, r in s.iterrows():
            lines.append(f"| {r.chunk_size} | {r.overlap_pct:.0f}% | {r.k} | {_fmt_ci(r, 'complete')} | "
                         f"{_fmt_ci(r, 'hit')} | {_fmt_ci(r, 'ev_recall')} | {r.completable:.3f} | "
                         f"{r.random_hit:.3f} | {r.tokens_q_median:.0f} [{r.tokens_q_min}–{r.tokens_q_max}] | "
                         f"{r.n_chunks} | {r.duplicate_token_fraction:.3f} |")
        (run_dir / "tables" / f"{mode}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    st = cfg["statistics"]
    lines = [f"Pre-specified primary comparisons: {st['primary_metric']}, mode `{st['primary_mode']}` "
             f"({cfg['modes'][st['primary_mode']]}), paper-level.",
             "Diff = mean over papers of (A − B). CI: paired percentile bootstrap over papers. "
             "p: two-sided paired sign-flip permutation test. Holm across all 8 tests.", "",
             "| A | B | Mean A | Mean B | Diff (95% CI) | p (raw) | p (Holm) | Papers A>B / A<B / = | n papers | Test |",
             "|---|---|---:|---:|---|---:|---:|---|---:|---|"]
    for _, c in comp.iterrows():
        lines.append(f"| {c.a} | {c.b} | {c.mean_a:.3f} | {c.mean_b:.3f} | {c['diff']:+.3f} "
                     f"({c.ci_low:+.3f} to {c.ci_high:+.3f}) | {c.p_raw:.4f} | {c.p_holm:.4f} | "
                     f"{c.papers_a_better} / {c.papers_b_better} / {c.papers_equal} | {c.n_papers} | {c.test} |")
    lines += ["", "A p-value above 0.05 is not evidence that two configurations are equivalent."]
    (run_dir / "tables" / "primary_comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figures(run_dir, summary, comp, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {0: "#4a4a4a", 10: "#2a78d6", 20: "#eb6834"}
    fig, axes = plt.subplots(1, len(cfg["modes"]), figsize=(5 * len(cfg["modes"]), 4), sharey=True)
    axes = np.atleast_1d(axes)
    sizes = cfg["chunking"]["sizes"]
    for ax, mode in zip(axes, cfg["modes"]):
        s = summary[summary["mode"] == mode]
        for j, pct in enumerate(cfg["chunking"]["overlap_pct"]):
            g = s[s.overlap_pct == pct].set_index("chunk_size").loc[sizes]
            x = np.arange(len(sizes)) + (j - 1) * 0.12
            ax.errorbar(x, g.complete, yerr=[g.complete - g.complete_ci_low, g.complete_ci_high - g.complete],
                        fmt="o", color=colors.get(int(pct), "k"), capsize=3, label=f"{pct:.0f}% overlap")
            ax.plot(x, g.completable, "_", color=colors.get(int(pct), "k"), ms=12, alpha=0.6)
        ax.set_xticks(np.arange(len(sizes)), [str(z) for z in sizes])
        ax.set_xlabel("Chunk size (tokens)")
        m = cfg["modes"][mode]
        desc = (f"nominal {m['budget']:,}-token budget, K = " + " / ".join(str(m['budget'] // z) for z in sizes)
                if m["type"] == "token_budget" else f"fixed K = {m['k']}")
        ax.set_title(f"{mode.capitalize()}: {desc}", fontsize=9, loc="left")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Complete@K (mean of per-paper means)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "Points: mean with 95% paper-bootstrap CI. Dashes: Completable ceiling.", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(run_dir / "figures" / "complete_by_config.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    y = np.arange(len(comp))[::-1]
    ax.errorbar(comp["diff"], y, xerr=[comp["diff"] - comp.ci_low, comp.ci_high - comp["diff"]],
                fmt="o", color="#2a78d6", capsize=3)
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(y, [f"{a} − {b}" for a, b in zip(comp.a, comp.b)], fontsize=8)
    ax.set_xlabel("Difference in Complete@K (per-paper mean, 95% CI)")
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "primary_comparisons.png", dpi=160)
    plt.close(fig)

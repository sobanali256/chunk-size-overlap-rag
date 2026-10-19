"""
statistics.py - paper-level aggregation, paired bootstrap CIs, paired permutation tests, Holm.

The paper is the unit of inference (EXPERIMENT_PROTOCOL.md section 6):
  per-question metric -> mean within each paper -> mean across papers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paper_means(per_q: pd.DataFrame, metric: str, keys=("config_id",)) -> pd.DataFrame:
    """One row per (config, paper): mean of the metric over that paper's questions."""
    return (per_q.groupby([*keys, "paper_id"])[metric]
            .agg(value="mean", n_questions="size").reset_index())


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int, level: float = 0.95):
    """Percentile CI for the mean of per-paper values, resampling papers with replacement."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_permutation_test(d: np.ndarray, n_perm: int, seed: int, exact_max_n: int = 16):
    """Two-sided sign-flip test of mean(d) = 0 for paired per-paper differences d.

    Under H0 each paper's difference is equally likely to have either sign. Exact enumeration of
    all 2^n sign vectors when n <= exact_max_n; otherwise Monte Carlo with n_perm random sign
    vectors and p = (1 + #{|T*| >= |T|}) / (1 + n_perm), which is never zero.
    Returns (p_value, method).
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        raise ValueError("no paired papers")
    t_obs = abs(d.mean())
    tol = 1e-12
    if n <= exact_max_n:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        t = np.abs((signs * d).mean(axis=1))
        return float((t >= t_obs - tol).mean()), f"exact ({2 ** n} sign vectors)"
    rng = np.random.default_rng(seed)
    count, done, batch = 0, 0, 10_000
    while done < n_perm:
        b = min(batch, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        count += int((np.abs((signs * d).mean(axis=1)) >= t_obs - tol).sum())
        done += b
    return (1 + count) / (1 + n_perm), f"Monte Carlo ({n_perm} sign vectors)"


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare(pm: pd.DataFrame, a: str, b: str, n_boot: int, n_perm: int, seed: int) -> dict:
    """Paired comparison of config a minus config b on per-paper values."""
    wa = pm[pm.config_id == a].set_index("paper_id").value
    wb = pm[pm.config_id == b].set_index("paper_id").value
    common = wa.index.intersection(wb.index).sort_values()
    if len(common) != len(wa) or len(common) != len(wb):
        raise ValueError(f"{a} and {b} are not evaluated on the same papers")
    d = (wa.loc[common] - wb.loc[common]).to_numpy()
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    p, method = paired_permutation_test(d, n_perm, seed)
    return {"a": a, "b": b, "mean_a": float(wa.loc[common].mean()), "mean_b": float(wb.loc[common].mean()),
            "diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "p_raw": p, "test": method,
            "n_papers": len(common), "papers_a_better": int((d > 0).sum()),
            "papers_b_better": int((d < 0).sum()), "papers_equal": int((d == 0).sum())}

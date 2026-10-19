"""
export_canonical_ids.py - post-run export of aggregated/per_paper.csv with canonical paper IDs.

Added after the fresh_v1 run (see EXPERIMENT_PROTOCOL.md section 10, operational log). src/reporting.py
(frozen, unchanged) reads per_question.csv without a string dtype, so paper IDs such as 1604.00400 are
written to per_paper.csv as 1604.004. This script writes aggregated/per_paper_canonical_ids.csv: the same
rows and values, with each paper_id mapped back to its canonical form from raw/per_question.csv. The mapping
is checked to be one-to-one, and every other column is checked to be unchanged. per_paper.csv is not modified.

Usage (from experiment/):
  python export_canonical_ids.py --run results/fresh_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    agg = Path(args.run) / "aggregated"

    papers = pd.read_csv(Path(args.run) / "raw" / "per_question.csv", dtype={"paper_id": str}).paper_id.unique()
    canon = {float(pid): pid for pid in papers}
    assert len(canon) == len(papers), "paper ids are not distinct as numbers"

    raw = pd.read_csv(agg / "per_paper.csv", dtype={"paper_id": str})
    fixed = raw.copy()
    fixed["paper_id"] = [canon[float(pid)] for pid in raw.paper_id]
    changed = sorted({(a, b) for a, b in zip(raw.paper_id, fixed.paper_id) if a != b})

    out = agg / "per_paper_canonical_ids.csv"
    fixed.to_csv(out, index=False)
    back = pd.read_csv(out, dtype={"paper_id": str})
    assert back.paper_id.tolist() == fixed.paper_id.tolist()
    assert set(back.paper_id) == set(papers)
    other = [c for c in raw.columns if c != "paper_id"]
    pd.testing.assert_frame_equal(back[other], raw[other])

    print(f"wrote {out}: {len(back)} rows, {len(changed)} paper ids restored")
    for a, b in changed:
        print(f"  {a} -> {b}")


if __name__ == "__main__":
    main()

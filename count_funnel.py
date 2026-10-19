#!/usr/bin/env python3
"""
count_funnel.py - count how QASPER questions were filtered down to eval.jsonl.

Replays prepare_qasper.py's selection exactly (same shuffle, same filters, same cap)
and reports why each question was dropped. Run it from the same folder as
prepare_qasper.py, with the same arguments you used to build the corpus:

  python count_funnel.py --qasper data/qasper_raw/qasper-dev-v0.3.json --papers 40 --max-per-paper 5 --seed 0

The last line checks the result against eval.jsonl, so you know the replay matches.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare_qasper import build_doc  # the exact document builder used for the corpus


def why_dropped(q, text):
    """None if prepare_qasper.pick_answer would keep this question, else the reason."""
    answers = [a["answer"] for a in q["answers"]]
    answerable = [a for a in answers if not a["unanswerable"]]
    if not answerable:
        return "unanswerable"
    with_hl = [a for a in answerable if any(h.strip() for h in a["highlighted_evidence"])]
    if not with_hl:
        return "no highlighted evidence"
    text_only = [a for a in with_hl
                 if not any("FLOAT SELECTED" in h for h in a["highlighted_evidence"] if h.strip())]
    if not text_only:
        return "table or figure only"
    for a in text_only:
        hl = [h.strip() for h in a["highlighted_evidence"] if h.strip()]
        if all(h in text for h in hl):
            return None
    return "evidence not found verbatim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasper", type=Path, required=True)
    ap.add_argument("--papers", type=int, default=40)
    ap.add_argument("--max-per-paper", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=Path, default=Path("eval.jsonl"))
    args = ap.parse_args()

    data = json.loads(args.qasper.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    paper_ids = sorted(data)
    rng.shuffle(paper_ids)

    selected, skipped_papers = 0, 0
    reasons, total, kept, capped = Counter(), 0, 0, 0
    for pid in paper_ids:
        if selected == args.papers:
            break
        text, _ = build_doc(data[pid])
        usable = 0
        paper_reasons = Counter()
        for q in data[pid]["qas"]:
            r = why_dropped(q, text)
            if r is None:
                usable += 1
            else:
                paper_reasons[r] += 1
        if usable == 0:
            skipped_papers += 1  # prepare_qasper.py skips papers with no usable question
            continue
        selected += 1
        total += len(data[pid]["qas"])
        reasons.update(paper_reasons)
        kept += min(usable, args.max_per_paper)
        capped += max(0, usable - args.max_per_paper)

    print(f"Papers drawn: {selected + skipped_papers} "
          f"({skipped_papers} skipped for having no usable question, {selected} kept)")
    print(f"Questions in the {selected} kept papers: {total}")
    for r in ("unanswerable", "no highlighted evidence", "table or figure only", "evidence not found verbatim"):
        print(f"  dropped, {r}: {reasons[r]}")
    print(f"  dropped, over the {args.max_per_paper}-per-paper cap: {capped}")
    print(f"Kept: {kept}")
    if args.eval.exists():
        n = sum(1 for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"eval.jsonl has {n} questions -> {'MATCH' if n == kept else 'MISMATCH: check the arguments'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prepare_qasper.py - build corpus/ and eval.jsonl for chunk_experiment.py from QASPER.

QASPER (Dasigi et al., 2021): questions about NLP papers, written by people who had
read only the title and abstract, answered by other annotators who highlighted the
supporting sentences in the full text. Those highlighted sentences become the
evidence spans; every one must appear verbatim in the rebuilt document.

Each paper becomes one Markdown file: title, abstract and sections with headings.
Questions are kept only if an annotator marked them answerable and every
highlighted sentence is found verbatim in the text. Questions whose evidence is a
figure or table ("FLOAT SELECTED") are dropped, because tables are not in the text.

Question type is derived from the evidence, not from the answer:
  multi-part      highlighted sentences come from two or more different paragraphs
  multi-sentence  one paragraph, more than one sentence
  factoid         a single sentence
("boundary" is not a fixed label: chunk_experiment.py tags it per chunk size.)

Papers are split 70/30 into "dev" and "heldout" by paper, so a held-out question
never shares a document with a dev question.

Usage
-----
  curl -L -o data/qasper_raw/qasper.tgz \
      https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
  tar -xzf data/qasper_raw/qasper.tgz -C data/qasper_raw
  python prepare_qasper.py --qasper data/qasper_raw/qasper-dev-v0.3.json \
      --papers 30 --max-per-paper 5 --corpus corpus --eval eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SENT_END = re.compile(r"[.!?](?=\s+[A-Z(])")


def build_doc(paper: dict) -> tuple[str, list[tuple[int, int]]]:
    """Markdown text plus the character range of every paragraph."""
    parts, para_ranges, pos = [], [], 0

    def add(text, is_para):
        nonlocal pos
        if parts:
            parts.append("\n\n")
            pos += 2
        if is_para:
            para_ranges.append((pos, pos + len(text)))
        parts.append(text)
        pos += len(text)

    add(f"# {paper['title'].strip()}", False)
    add("## Abstract", False)
    add(paper["abstract"].strip(), True)
    for sec in paper["full_text"]:
        name = (sec.get("section_name") or "").strip()
        if name:
            add(f"## {name}", False)
        for p in sec["paragraphs"]:
            if p.strip():
                add(p.strip(), True)
    return "".join(parts) + "\n", para_ranges


def pick_answer(q: dict, text: str):
    """First answerable annotation whose highlighted evidence is all verbatim in the text."""
    for a in q["answers"]:
        ans = a["answer"]
        if ans["unanswerable"]:
            continue
        hl = [h.strip() for h in ans["highlighted_evidence"] if h.strip()]
        if not hl or any("FLOAT SELECTED" in h for h in hl):
            continue
        if all(h in text for h in hl):
            return ans, list(dict.fromkeys(hl))  # drop duplicate highlights, keep order
    return None, None


def answer_text(ans: dict) -> str:
    if ans["extractive_spans"]:
        return " | ".join(ans["extractive_spans"])
    if ans["free_form_answer"]:
        return ans["free_form_answer"]
    if ans["yes_no"] is not None:
        return "yes" if ans["yes_no"] else "no"
    return ""


def classify(evidence: list[str], text: str, para_ranges) -> str:
    def para_of(ev):
        s = text.find(ev)
        return next((i for i, (a, b) in enumerate(para_ranges) if a <= s < b), -1)

    if len({para_of(ev) for ev in evidence}) > 1:
        return "multi-part"
    n_sent = sum(1 + len(SENT_END.findall(ev)) for ev in evidence)
    return "multi-sentence" if n_sent > 1 else "factoid"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qasper", type=Path, required=True, help="qasper-dev-v0.3.json")
    ap.add_argument("--papers", type=int, default=30)
    ap.add_argument("--max-per-paper", type=int, default=5, help="cap so a few papers don't dominate")
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--eval", type=Path, default=Path("eval.jsonl"))
    args = ap.parse_args()

    data = json.loads(args.qasper.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    paper_ids = sorted(data)
    rng.shuffle(paper_ids)

    selected = []  # (paper_id, text, [question dicts])
    for pid in paper_ids:
        if len(selected) == args.papers:
            break
        paper = data[pid]
        text, para_ranges = build_doc(paper)
        qs = []
        for q in paper["qas"]:
            ans, evidence = pick_answer(q, text)
            if ans is None:
                continue
            qs.append({"question": q["question"].strip(), "evidence": evidence,
                       "type": classify(evidence, text, para_ranges), "answer": answer_text(ans)})
        if qs:
            selected.append((pid, text, qs[: args.max_per_paper]))

    n_heldout = round(len(selected) * args.heldout_frac)
    heldout = {pid for pid, _, _ in rng.sample(selected, n_heldout)}

    args.corpus.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, text, qs in selected:
        doc = f"{pid}.md"
        (args.corpus / doc).write_text(text, encoding="utf-8")
        for i, q in enumerate(qs, 1):
            rows.append({"id": f"{pid}-q{i}", "question": q["question"], "doc": doc,
                         "evidence": q["evidence"], "type": q["type"], "answer": q["answer"],
                         "split": "heldout" if pid in heldout else "dev"})
    with open(args.eval, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    words = [len(t.split()) for _, t, _ in selected]
    print(f"{len(selected)} papers ({sum(words):,} words, {min(words)}-{max(words)} per paper), "
          f"{len(rows)} questions -> {args.corpus}/, {args.eval}")
    print("by type: ", dict(Counter(r["type"] for r in rows)))
    print("by split:", dict(Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()

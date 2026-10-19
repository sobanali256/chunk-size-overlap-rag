"""
data.py - QASPER loading, document building, question eligibility and paper-level sampling.

The document builder and the answer-selection rule are copied unchanged from the pilot's
prepare_qasper.py (build_doc, pick_answer) so that the fresh sample is processed exactly like
the pilot. tests/test_data.py checks that this module regenerates the pilot corpus and
eval.jsonl byte for byte.

Eligibility rules (EXPERIMENT_PROTOCOL.md section 4), applied per question in this order:
  unanswerable              every annotation is marked unanswerable
  no_highlighted_evidence   no answerable annotation highlights any text
  table_or_figure_only      every answerable annotation with highlights cites a table/figure
                            ("FLOAT SELECTED"); mixed text+table annotations are also skipped,
                            as in the pilot
  evidence_not_verbatim     no remaining annotation has all highlights verbatim in the document
  ambiguous_evidence        [new rule] a selected highlight occurs more than once in the document,
                            so its location cannot be determined
  duplicate_question        [new rule] same question text (case-insensitive) as an earlier
                            eligible question about the same paper
  over_cap                  eligible, but beyond the first `max_per_paper` eligible questions
                            of the paper (pilot rule, qas order)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

SENT_END = re.compile(r"[.!?](?=\s+[A-Z(])")


# ------------------------------------------------------- pilot code, copied unchanged

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


# ------------------------------------------------------------------- eligibility

def drop_reason(q: dict, text: str) -> str | None:
    """Pilot exclusion reason (None = pick_answer would keep it). Mirrors count_funnel.why_dropped."""
    answers = [a["answer"] for a in q["answers"]]
    answerable = [a for a in answers if not a["unanswerable"]]
    if not answerable:
        return "unanswerable"
    with_hl = [a for a in answerable if any(h.strip() for h in a["highlighted_evidence"])]
    if not with_hl:
        return "no_highlighted_evidence"
    text_only = [a for a in with_hl
                 if not any("FLOAT SELECTED" in h for h in a["highlighted_evidence"] if h.strip())]
    if not text_only:
        return "table_or_figure_only"
    if pick_answer(q, text)[0] is None:
        return "evidence_not_verbatim"
    return None


@dataclass
class Question:
    paper_id: str
    qasper_question_id: str
    index_in_paper: int          # 0-based position in the paper's qas list
    question: str
    status: str                  # "eligible" or an exclusion reason
    evidence: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)   # char ranges in the document
    type: str = ""
    answer: str = ""


def paper_questions(paper_id: str, paper: dict, text: str, para_ranges, max_per_paper: int) -> list[Question]:
    out, seen, n_ok = [], set(), 0
    for i, q in enumerate(paper["qas"]):
        qq = Question(paper_id, q.get("question_id", ""), i, q["question"].strip(), "eligible")
        reason = drop_reason(q, text)
        if reason is None:
            ans, ev = pick_answer(q, text)
            if any(text.count(e) > 1 for e in ev):
                reason = "ambiguous_evidence"
            elif qq.question.lower() in seen:
                reason = "duplicate_question"
            else:
                seen.add(qq.question.lower())
                qq.evidence = ev
                qq.spans = [(text.find(e), text.find(e) + len(e)) for e in ev]
                qq.type = classify(ev, text, para_ranges)
                qq.answer = answer_text(ans)
                n_ok += 1
                if n_ok > max_per_paper:
                    reason = "over_cap"
        if reason:
            qq.status = reason
        out.append(qq)
    return out


# ---------------------------------------------------------------------- sampling

def load_qasper(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pilot_drawn_papers(data: dict, n_papers: int, seed: int) -> list[str]:
    """Replay prepare_qasper.py's draw: every paper it looked at, kept or skipped."""
    ids = sorted(data)
    random.Random(seed).shuffle(ids)
    drawn, kept = [], 0
    for pid in ids:
        if kept == n_papers:
            break
        drawn.append(pid)
        text, _ = build_doc(data[pid])
        if any(pick_answer(q, text)[0] is not None for q in data[pid]["qas"]):
            kept += 1
    return drawn


def draw_sample(eligible_ids: list[str], n: int, seed: int) -> list[str]:
    """Sort, shuffle with a fixed seed, take the first n. Nested: the first n of a larger draw."""
    ids = sorted(eligible_ids)
    random.Random(seed).shuffle(ids)
    if n > len(ids):
        raise ValueError(f"requested {n} papers but only {len(ids)} eligible unused papers exist")
    return ids[:n]


def norm_title(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

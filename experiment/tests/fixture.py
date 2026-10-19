"""Synthetic fixture for unit and integration tests. Never part of any real result."""
from __future__ import annotations

import re

from src.pipeline import EvalQuestion


class FixtureTokenizer:
    """Whitespace/punctuation tokenizer with character offsets (one token per word or symbol)."""

    def offsets(self, text):
        return [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]

    def n_input_tokens(self, texts):
        return [len(self.offsets(t)) + 2 for t in texts]


TOPICS = ["apples grow on trees in the orchard", "the river floods every spring season",
          "solar panels convert light into power", "violins have four strings and a bow",
          "glaciers carve deep valleys slowly", "bread rises because yeast makes gas"]


def fixture_sample(n_papers: int = 6):
    papers, qs = {}, []
    for p in range(n_papers):
        sents = [f"Sentence {i} of paper {p} talks about {TOPICS[(p + i) % len(TOPICS)]} ." for i in range(30)]
        text = " ".join(sents) + "\n"
        papers[f"fx{p}"] = text
        for j, i in enumerate((3, 11, 20)):
            ev = [sents[i]] if j < 2 else [sents[i], sents[i + 5]]
            spans = [(text.find(e), text.find(e) + len(e)) for e in ev]
            qs.append(EvalQuestion(f"fx{p}", f"fx{p}:q{j}", f"what about {TOPICS[(p + i) % len(TOPICS)]} sentence {i}",
                                   spans, "multi-part" if j == 2 else "factoid"))
    return papers, qs

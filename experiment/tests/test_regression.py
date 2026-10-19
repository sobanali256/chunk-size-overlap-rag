"""
Regression tests against the audited pilot. They need the QASPER dev file and the Nomic
tokenizer (both local after setup); they never embed or retrieve anything from the fresh sample.
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # the original project folder
EXP = Path(__file__).resolve().parents[1]
QASPER = ROOT / "data" / "qasper_raw" / "qasper-dev-v0.3.json"

from src import data as D                           # noqa: E402
from src.chunking import chunk_paper, overlap_tokens  # noqa: E402
from src.metrics import completable, span_token_set  # noqa: E402


def _load_original(name):
    spec = importlib.util.spec_from_file_location(f"orig_{name}", ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(QASPER.exists(), "QASPER dev file not present")
class TestDataMatchesPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = D.load_qasper(QASPER)
        cls.orig = _load_original("prepare_qasper")
        cls.pilot = [json.loads(x) for x in (ROOT / "eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    def test_copied_functions_identical_on_every_dev_paper(self):
        for pid, p in self.data.items():
            text, paras = D.build_doc(p)
            self.assertEqual((text, paras), self.orig.build_doc(p))
            for q in p["qas"]:
                self.assertEqual(D.pick_answer(q, text), self.orig.pick_answer(q, text))

    def test_pilot_corpus_regenerates(self):
        for pid in {r["doc"][:-3] for r in self.pilot}:
            text, _ = D.build_doc(self.data[pid])
            self.assertEqual(text, (ROOT / "corpus" / f"{pid}.md").read_text(encoding="utf-8"))

    def test_pilot_questions_regenerate(self):
        """Eligible questions under the new rules == pilot eval.jsonl (new rules exclude nothing there)."""
        for pid in sorted({r["doc"][:-3] for r in self.pilot}):
            text, paras = D.build_doc(self.data[pid])
            qs = [q for q in D.paper_questions(pid, self.data[pid], text, paras, 5) if q.status == "eligible"]
            ref = [r for r in self.pilot if r["doc"] == f"{pid}.md"]
            self.assertEqual([(q.question, q.evidence, q.type, q.answer) for q in qs],
                             [(r["question"], r["evidence"], r["type"], r["answer"]) for r in ref])

    def test_pilot_draw_replay(self):
        drawn = D.pilot_drawn_papers(self.data, 40, 0)
        self.assertEqual(len(drawn), 42)
        self.assertTrue({r["doc"][:-3] for r in self.pilot} <= set(drawn))

    def test_sampling_is_nested_and_deterministic(self):
        ids = [f"p{i}" for i in range(50)]
        a, b = D.draw_sample(ids, 10, 3), D.draw_sample(list(reversed(ids)), 20, 3)
        self.assertEqual(a, b[:10])
        with self.assertRaises(ValueError):
            D.draw_sample(ids, 51, 3)


@unittest.skipUnless(QASPER.exists(), "QASPER dev file not present")
class TestTokenGeometryMatchesPilot(unittest.TestCase):
    """Token-position Completable and boundary flags must equal the pilot's char-based values."""

    @classmethod
    def setUpClass(cls):
        import yaml
        from src.embeddings import Tokenizer
        cfg = yaml.safe_load((EXP / "configs" / "clean_rerun.yaml").read_text(encoding="utf-8"))
        cls.tok = Tokenizer(cfg["tokenizer"]["id"], cfg["tokenizer"]["revision"])
        cls.pq = pd.read_csv(ROOT / "results" / "budget2048" / "per_question.csv")
        cls.pilot = [json.loads(x) for x in (ROOT / "eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        cls.docs = {r["doc"]: (ROOT / "corpus" / r["doc"]).read_text(encoding="utf-8") for r in cls.pilot}

    def test_total_tokens(self):
        self.assertEqual(sum(len(self.tok.offsets(t)) for t in self.docs.values()), 205160)

    def test_completable_and_chunk_counts(self):
        offs = {d: self.tok.offsets(t) for d, t in self.docs.items()}
        n_ref = pd.read_csv(ROOT / "results" / "budget2048" / "results.csv")
        for size in (64, 128, 256, 512, 1024):
            for pct in (0, 10, 20):
                ov = overlap_tokens(size, pct)
                chunks = {d: chunk_paper(d, offs[d], size, ov) for d in self.docs}
                ref_n = n_ref[(n_ref.chunk_size == size) & (n_ref.overlap_pct == pct)].n_chunks.item()
                self.assertEqual(sum(map(len, chunks.values())), ref_n)
                ref = self.pq[(self.pq.chunk_size == size) & (self.pq.overlap_pct == pct)].set_index("id")
                for r in self.pilot:
                    t = self.docs[r["doc"]]
                    ev = [span_token_set(offs[r["doc"]], t.find(e), t.find(e) + len(e)) for e in r["evidence"]]
                    self.assertEqual(float(completable(chunks[r["doc"]], ev)), ref.loc[r["id"], "completable"],
                                     f"{r['id']} {size}/{pct}")


if __name__ == "__main__":
    unittest.main()

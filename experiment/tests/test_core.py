"""Unit tests on small hand-built examples. Run: python -m unittest discover -s tests -t ."""
import math
import unittest

import numpy as np

from src.chunking import Chunk, chunk_paper, overlap_tokens, token_windows
from src.metrics import completable, score_question, span_token_set
from src.retrieval import k_for, rank_paper
from src.statistics import bootstrap_mean_ci, holm, paired_permutation_test


def offs(n, w=2):
    """n tokens of width w separated by one space: token i covers chars [3i, 3i+2)."""
    return [(i * (w + 1), i * (w + 1) + w) for i in range(n)]


def ck(t0, t1, idx=0):
    o = offs(1000)
    return Chunk("p", idx, t0, t1, o[t0][0], o[t1 - 1][1])


class TestChunking(unittest.TestCase):
    def test_overlap_definition(self):
        self.assertEqual([overlap_tokens(128, p) for p in (0, 10, 20)], [0, 13, 26])
        self.assertEqual([overlap_tokens(256, p) for p in (0, 10, 20)], [0, 26, 51])
        self.assertEqual([overlap_tokens(512, p) for p in (0, 10, 20)], [0, 51, 102])
        with self.assertRaises(ValueError):
            overlap_tokens(10, 100)

    def test_windows_no_overlap(self):
        self.assertEqual(token_windows(10, 5, 0), [(0, 5), (5, 10)])

    def test_windows_overlap_matches_article_example(self):
        # article: A..J, size 5, overlap 2 -> ABCDE, DEFGH, GHIJ
        self.assertEqual(token_windows(10, 5, 2), [(0, 5), (3, 8), (6, 10)])

    def test_final_partial_chunk_and_short_doc(self):
        self.assertEqual(token_windows(12, 5, 0), [(0, 5), (5, 10), (10, 12)])
        self.assertEqual(token_windows(3, 5, 1), [(0, 3)])
        self.assertEqual(token_windows(0, 5, 0), [])
        self.assertEqual(token_windows(5, 5, 2), [(0, 5)])   # exact fit: no redundant tail chunk

    def test_every_token_covered_and_stride(self):
        for n in (1, 7, 128, 1000, 1001):
            for size, ov in ((128, 13), (256, 51), (8, 0), (16, 3)):
                w = token_windows(n, size, ov)
                cov = set()
                for a, b in w:
                    cov.update(range(a, b))
                    self.assertLessEqual(b - a, size)
                self.assertEqual(cov, set(range(n)))
                self.assertTrue(all(b[0] - a[0] == size - ov for a, b in zip(w, w[1:])))

    def test_overlap_guarantee_o_plus_1(self):
        """Any span of at most O+1 tokens lies inside some chunk (article claim)."""
        n, size, ov = 300, 20, 4
        w = token_windows(n, size, ov)
        for s in range(n - ov):
            self.assertTrue(any(a <= s and s + ov + 1 <= b for a, b in w))

    def test_chunk_char_range(self):
        c = chunk_paper("p", offs(10), 4, 0)
        self.assertEqual((c[1].t0, c[1].t1, c[1].c0, c[1].c1), (4, 8, 12, 23))


class TestSpanTokens(unittest.TestCase):
    def test_exact_and_partial_tokens(self):
        o = offs(10)                              # token 2 = [6, 8), token 3 = [9, 11)
        self.assertEqual(span_token_set(o, 6, 11), range(2, 4))
        self.assertEqual(span_token_set(o, 7, 10), range(2, 4))   # starts and ends mid-token
        with self.assertRaises(ValueError):
            span_token_set(o, 8, 9)               # whitespace only: no token
        with self.assertRaises(ValueError):
            span_token_set(o, 5, 5)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # paper of 20 tokens, chunk size 5, no overlap -> chunks [0,5) [5,10) [10,15) [15,20)
        self.chunks = [ck(0, 5, 0), ck(5, 10, 1), ck(10, 15, 2), ck(15, 20, 3)]

    def rank(self, *order):
        return [self.chunks[i] for i in order]

    def test_single_span_complete(self):
        m = score_question(self.rank(1, 0, 2, 3), [range(6, 9)], k=1)
        self.assertEqual((m["hit"], m["complete"], m["ev_recall"], m["completable"]), (1, 1, 1, 1))
        self.assertEqual(m["first_rel_rank"], 1)

    def test_span_split_across_two_retrieved_chunks_is_hit_not_complete(self):
        ev = [range(8, 12)]                       # tokens 8-11 straddle chunks 1 and 2
        m = score_question(self.rank(1, 2, 0, 3), ev, k=2)
        self.assertEqual((m["hit"], m["complete"], m["ev_recall"]), (1.0, 0.0, 1.0))
        self.assertEqual(m["completable"], 0.0)   # no single chunk can hold it

    def test_partial_hit(self):
        m = score_question(self.rank(1, 0, 2, 3), [range(8, 12)], k=1)
        self.assertEqual((m["hit"], m["complete"]), (1.0, 0.0))
        self.assertAlmostEqual(m["ev_recall"], 2 / 4)

    def test_multi_span_needs_all_spans_in_possibly_different_chunks(self):
        ev = [range(1, 3), range(16, 18)]
        self.assertEqual(score_question(self.rank(0, 3, 1, 2), ev, k=2)["complete"], 1.0)
        m = score_question(self.rank(0, 1, 3, 2), ev, k=2)
        self.assertEqual((m["hit"], m["complete"]), (1.0, 0.0))
        self.assertAlmostEqual(m["ev_recall"], 0.5)

    def test_miss(self):
        m = score_question(self.rank(3, 2, 0, 1), [range(6, 8)], k=2)
        self.assertEqual((m["hit"], m["complete"], m["ev_recall"], m["rr"]), (0, 0, 0, 0))
        self.assertEqual(m["first_rel_rank"], 4)

    def test_overlapping_chunks_do_not_inflate_recall_or_precision(self):
        ch = [ck(0, 6, 0), ck(4, 10, 1), ck(8, 14, 2)]   # size 6, overlap 2
        ev = [range(4, 6)]                               # inside both chunk 0 and chunk 1
        m = score_question([ch[0], ch[1], ch[2]], ev, k=2)
        self.assertEqual(m["ev_recall"], 1.0)
        self.assertEqual(m["retrieved_tokens"], 12)
        self.assertEqual(m["retrieved_unique_tokens"], 10)
        self.assertAlmostEqual(m["evidence_precision_unique"], 2 / 10)
        self.assertAlmostEqual(m["redundancy"], 1 - 10 / 12)

    def test_overlapping_gold_spans_counted_once(self):
        ev = [range(2, 6), range(4, 8)]           # union = 6 tokens
        m = score_question(self.rank(0, 1, 2, 3), ev, k=1)
        self.assertEqual(m["evidence_tokens"], 6)
        self.assertAlmostEqual(m["ev_recall"], 3 / 6)

    def test_fewer_chunks_than_k(self):
        ch = [ck(0, 5, 0), ck(5, 7, 1)]           # short paper, short final chunk
        m = score_question([ch[1], ch[0]], [range(1, 3)], k=16)
        self.assertEqual((m["k_effective"], m["n_candidates"], m["retrieved_tokens"]), (2, 2, 7))
        self.assertEqual((m["hit"], m["complete"], m["random_hit"]), (1.0, 1.0, 1.0))

    def test_random_hit_exact(self):
        # 4 chunks, 1 relevant, k=2: 1 - C(3,2)/C(4,2) = 0.5
        m = score_question(self.rank(3, 2, 1, 0), [range(6, 8)], k=2)
        self.assertAlmostEqual(m["random_hit"], 0.5)
        # Monte Carlo agreement
        rng = np.random.default_rng(0)
        sims = [(1 in rng.choice(4, 2, replace=False)) for _ in range(20000)]
        self.assertAlmostEqual(np.mean(sims), 0.5, delta=0.02)
        # 2 relevant (evidence straddles chunks 1 and 2), k=1: 2/4
        self.assertAlmostEqual(score_question(self.rank(0, 1, 2, 3), [range(8, 12)], k=1)["random_hit"], 0.5)

    def test_no_evidence_raises(self):
        with self.assertRaises(ValueError):
            score_question(self.chunks, [], k=2)

    def test_completable_independent_of_ranking(self):
        ev = [range(6, 9)]
        self.assertTrue(completable(self.chunks, ev))
        self.assertTrue(all(score_question(self.rank(*o), ev, 1)["completable"] == 1
                            for o in ((0, 1, 2, 3), (3, 2, 1, 0))))


class TestRetrieval(unittest.TestCase):
    def test_k_for(self):
        b = {"type": "token_budget", "budget": 2048}
        self.assertEqual([k_for(b, s) for s in (128, 256, 512)], [16, 8, 4])
        self.assertEqual(k_for({"type": "fixed_k", "k": 5}, 512), 5)
        with self.assertRaises(ValueError):
            k_for(b, 3000)

    def test_rank_stable_ties(self):
        C = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
        order, _ = rank_paper(np.array([1.0, 0.0]), C)
        self.assertEqual(order.tolist(), [0, 2, 1])


class TestStatistics(unittest.TestCase):
    def test_exact_permutation_small(self):
        # d = [1, 1, 1]: only all-+ and all-- reach |mean| = 1 -> p = 2/8
        p, method = paired_permutation_test(np.array([1.0, 1.0, 1.0]), 1000, 0)
        self.assertAlmostEqual(p, 0.25)
        self.assertIn("exact", method)
        p, _ = paired_permutation_test(np.array([1.0, -1.0]), 1000, 0)
        self.assertAlmostEqual(p, 1.0)

    def test_permutation_zero_differences(self):
        p, _ = paired_permutation_test(np.zeros(30), 2000, 0)
        self.assertEqual(p, 1.0)

    def test_monte_carlo_matches_exact(self):
        rng = np.random.default_rng(1)
        d = rng.normal(0.3, 1, 16)
        p_exact, _ = paired_permutation_test(d, 0, 0, exact_max_n=16)
        p_mc, m = paired_permutation_test(d, 200_000, 3, exact_max_n=0)
        self.assertIn("Monte Carlo", m)
        self.assertAlmostEqual(p_exact, p_mc, delta=0.005)

    def test_mc_p_never_zero_and_deterministic(self):
        d = np.ones(40)
        p1, _ = paired_permutation_test(d, 5000, 7)
        p2, _ = paired_permutation_test(d, 5000, 7)
        self.assertEqual(p1, p2)
        self.assertAlmostEqual(p1, 1 / 5001)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])
        self.assertEqual(holm([0.2]), [0.2])

    def test_bootstrap_ci(self):
        lo, hi = bootstrap_mean_ci(np.full(50, 0.4), 1000, 0)
        self.assertAlmostEqual(lo, 0.4, places=12)
        self.assertAlmostEqual(hi, 0.4, places=12)
        x = np.random.default_rng(0).binomial(1, 0.5, 400).astype(float)
        lo, hi = bootstrap_mean_ci(x, 4000, 0)
        se = x.std(ddof=1) / math.sqrt(len(x))
        self.assertAlmostEqual(hi - lo, 2 * 1.96 * se, delta=0.01)
        self.assertEqual(bootstrap_mean_ci(x, 2000, 5), bootstrap_mean_ci(x, 2000, 5))


class TestAggregation(unittest.TestCase):
    def test_paper_weighting(self):
        import pandas as pd
        from src.statistics import compare, paper_means
        # paper A: 4 questions all 1; paper B: 1 question 0. Question mean 0.8, paper mean 0.5.
        df = pd.DataFrame({"config_id": ["x"] * 5, "paper_id": list("AAAAB"), "complete": [1, 1, 1, 1, 0]})
        pm = paper_means(df, "complete")
        self.assertEqual(pm.value.mean(), 0.5)
        self.assertEqual(pm.set_index("paper_id").n_questions.to_dict(), {"A": 4, "B": 1})
        df2 = df.assign(config_id="y", complete=[0, 0, 0, 0, 0])
        r = compare(paper_means(pd.concat([df, df2]), "complete"), "x", "y", 500, 500, 0)
        self.assertEqual((r["diff"], r["n_papers"], r["papers_a_better"], r["papers_equal"]), (0.5, 2, 1, 1))

    def test_unpaired_papers_rejected(self):
        import pandas as pd
        from src.statistics import compare, paper_means
        df = pd.DataFrame({"config_id": ["x", "x", "y"], "paper_id": ["A", "B", "A"], "complete": [1, 0, 1]})
        with self.assertRaises(ValueError):
            compare(paper_means(df, "complete"), "x", "y", 100, 100, 0)


if __name__ == "__main__":
    unittest.main()

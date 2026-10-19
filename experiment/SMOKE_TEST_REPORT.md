# Smoke test report

Date: 2026-10-17.

**Environment:**

- Windows 11, CPU only.
- Python 3.14.2, torch 2.12.0+cpu, transformers 5.12.1, sentence-transformers 6.1.0, numpy 2.4.2,
  pandas 3.0.0.
- Model `nomic-ai/nomic-embed-text-v1.5` @ `e9b6763…`, remote code @ `7710840…`.

**No fresh-sample paper was embedded or retrieved.** The fresh sample was only tokenized, to count tokens
for the runtime estimate. All runs below use either the synthetic fixture or pilot papers, which are
already exploratory.

## Commands and outcomes

| # | Command (from `experiment/`) | Purpose | Result |
|---|---|---|---|
| 1 | `python -m unittest discover -s tests -t .` | 36 unit + regression tests | **36 passed** (after fixing one float-equality assertion in a test) |
| 2 | `python run_experiment.py run --sample fixture --out results/fixture_a`, then the same to `fixture_b`, plus `analyze` on both | Synthetic end-to-end run (hashing embedder, 6 papers, 18 questions) | All stages ran. Every raw, aggregated, statistics and table file is **byte-identical** between a and b. |
| 3 | `python run_experiment.py verify --run results/fixture_a --verify-rows 324` | Independent recomputation | 324/324 rows, 0 mismatches |
| 4 | `python run_experiment.py run --sample pilot --limit-papers 5 --out results/pilot5_a` | Real model, cold cache | 9 configurations, 5 papers, 12 questions, 372 s; 492 document tokens/s |
| 5 | same with `--no-cache` → `pilot5_b`; with a warm cache → `pilot5_c`; `analyze` on all three | Determinism and cache correctness | `per_question.csv`, `rankings.jsonl`, `chunks.csv`, `per_paper.csv`, `primary_comparisons.csv` and the tables are byte-identical across a, b and c. `summary.csv` differs only in `embed_seconds` and `cache_hit`. |
| 6 | `python run_experiment.py verify --run results/pilot5_a --verify-rows 216` | Independent recomputation, real model | 216/216 rows (all), 0 mismatches |
| 7 | Cross-check of `pilot5_a` against the original `../results/{budget2048,k5}/per_question.csv` | New pipeline vs the audited pilot | Hit, Complete, Completable, Random Hit, first-relevant rank, retrieved tokens, K and candidate counts: **108/108 identical in both modes**. Evidence recall: identical in 81/108 and 76/108 rows; mean absolute difference 0.008 / 0.009, maximum 0.08 / 0.10. Expected, because the original is character-based and the new one counts unique token positions. |
| 8 | `python run_experiment.py manifest` | Freeze the fresh sample | 100 papers / 287 questions from 231 eligible unused papers; lock written |
| 9 | `analyze` outputs | Figures and tables generated | `figures/complete_by_config.png`, `figures/primary_comparisons.png`, `tables/*.md` produced from raw files |

## Audit reproduction checks (scratch, original code)

These are recorded in `EXPERIMENT_AUDIT.md`:

- The data rebuild is byte-identical.
- The cache key confirms the document prefix.
- All 1,635 × 2 original rows reproduce with query prefix `search_query: `.
- `analysis.md` regenerates byte-identically.
- All aggregates and CIs match.

## What the tests cover

- **Chunking**: boundaries, the overlap rounding table, final partial chunks, short and empty documents,
  exact fits, full coverage, the stride, and the O + 1 guarantee.
- **Evidence**: token mapping, including mid-token starts and ends; whitespace-only and empty spans raise.
- **Metrics**: multi-span (different chunks allowed); a span split across two retrieved chunks
  (Hit = 1, Complete = 0); partial hits; misses; overlapping chunks (recall and precision are not
  inflated); overlapping gold spans (counted once); fewer chunks than K; exact Random Hit, checked by
  Monte Carlo; Completable independent of ranking; a question with no evidence raises.
- **Retrieval and aggregation**: K per mode; stable tie-break; paper weighting (paper mean 0.5 vs question
  mean 0.8); unpaired papers rejected.
- **Statistics**: exact permutation p on known cases; all-zero differences give p = 1; Monte Carlo agrees
  with exact; the Monte Carlo p is never 0 and is deterministic; Holm on a known example; bootstrap
  degenerate case, width vs the normal approximation, and determinism.
- **Regression against the pilot**: the copied `build_doc`/`pick_answer` are identical on all 281 dev
  papers; the pilot corpus and questions regenerate; the pilot draw replays to 42 papers; the 205,160
  token total; chunk counts and Completable equal the original on all 15 configurations × 109 questions.
- **Input limit**: over-long inputs raise. The maximum input observed is 518 tokens at 512-token chunks.

## Known limitations and issues

- **No clean-environment test.** I did not build a fresh virtual environment: it would mean downloading
  torch and the rest, and the installed environment already reproduces the pilot. `requirements.txt` pins
  the tested versions. Status: **pending**. I recommend it before publication.
- **Library warnings.** transformers warns that `get_extended_attention_mask` (used by Nomic's remote
  code) is deprecated. It works on the pinned 5.12.1; a future transformers may break the remote code,
  which is another reason to pin versions. Hugging Face also warns about unauthenticated downloads; that is
  harmless.
- **Runtime estimate.** It assumes the 492 tok/s measured on 5 papers holds for 100. Long papers (up to
  23,245 tokens) do not change per-chunk cost.
- **Integration sample size.** The pilot integration run uses 5 papers and 12 questions. Its numbers are
  a software test, not results.

## Ready for the full run?

**Yes, from a software standpoint.** No critical failure is open. Remaining: your approval of the
protocol, sample size and command, and optionally the clean-environment test.

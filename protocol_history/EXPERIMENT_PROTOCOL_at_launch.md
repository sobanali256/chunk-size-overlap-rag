# Experiment protocol: chunk size and overlap, fresh-sample evaluation

Version 1.0, written 2026-10-17, **before any fresh-sample retrieval was run**. Machine-readable settings:
`configs/clean_rerun.yaml`. Sample: `manifests/` (hashes in `manifests/sample_lock.json`).

Status: **APPROVED and FROZEN on 2026-10-17** by the author, at 100 papers, before the fresh run was
launched. Any later change is listed under "Deviations" (§10), and an analysis it affects is labelled
exploratory.

## 1. Research question and scope

Under a controlled retrieval setup, how do chunk size and overlap affect evidence retrieval and evidence
completeness for questions about individual research papers?

Conclusions are limited to the setup tested:

- QASPER v0.3 papers, from the dev file.
- One embedding model.
- Exact cosine search within the question's own paper.
- Fixed-size token windows from the model's tokenizer.
- Chunk sizes 128 / 256 / 512 and overlaps 0 / 10 / 20%.

No claim is made about an optimal chunk size in general, about 64- or 1,024-token chunks, about
multi-document indexes, or about answer generation.

## 2. Relation to the pilot

The 40-paper, 109-question pilot is **exploratory**: its held-out questions influenced the grid and the
selection rule (`EXPERIMENT_AUDIT.md` §2). It is used only for:

- Software tests and reproduction checks.
- Pilot-based precision estimates (§9).

No pilot number is pooled with fresh results. No pilot number is presented as confirmation.

## 3. Sample

| Item | Rule |
|---|---|
| Source | `data/qasper_raw/qasper-dev-v0.3.json` (SHA-256 `2ae7ee62…af93`), 281 papers |
| Excluded as pilot | All 42 papers that the pilot's draw (`prepare_qasper.py`, 40 papers, seed 0) looked at: the 40 used and the 2 skipped. Also any paper whose normalised title equals one of them (0 found). |
| Paper eligibility | Has full text and at least one eligible question (§4). 231 papers are eligible; 8 have no eligible question. |
| Draw | `sorted(eligible ids)`, then `random.Random(20261017).shuffle`, then the first **100**. The draw is nested: a larger N would include these 100. |
| Result | 100 papers, 287 eligible questions (1–5 per paper; mean 2.87). By type: 129 multi-sentence, 95 factoid, 63 multi-part; 76 questions have more than one span. |
| Untouched reserve | The 131 eligible papers not drawn stay unused for future confirmation. |

The sample was frozen before any retrieval on it (`sample_lock.json`). `run_experiment.py` refuses to run
the fresh sample if a manifest or the QASPER file changes.

## 4. Question eligibility

These are the pilot rules, unchanged, applied per question in this order. Every exclusion is recorded in
`manifests/question_manifest.csv`.

1. `unanswerable`: every annotation is marked unanswerable.
2. `no_highlighted_evidence`: no answerable annotation highlights any text.
3. `table_or_figure_only`: every answerable annotation that has highlights cites a table or figure
   (`FLOAT SELECTED`). Annotations that mix text and table evidence are skipped, as in the pilot.
4. `evidence_not_verbatim`: no remaining annotation has all of its highlights verbatim in the rebuilt
   document.
5. **New rule**, `ambiguous_evidence`: a selected highlight occurs more than once in the document, so it
   cannot be located. The pilot used the first occurrence; this applies to 0 pilot and 0 fresh questions.
6. **New rule**, `duplicate_question`: the same question text (case-insensitive) as an earlier eligible
   question about the same paper. Applies to 0 pilot and 0 fresh questions.
7. `over_cap`: beyond the first 5 eligible questions of a paper, in QASPER order.

The gold evidence is the first annotation (in QASPER order) that passes rules 1–4. Its highlights,
de-duplicated, are the spans. Each span is located by exact string match; no normalisation is applied.
Question type is derived from the evidence, as in the pilot. It is descriptive only.

Fresh-sample exclusions within the 100 papers: 32 table/figure only, 20 unanswerable, 10 not verbatim,
6 no highlight, 17 over the cap. In the whole dev pool, 23 of 864 questions (2.7%) fail the verbatim
match. That is not a material share, so evidence boundaries are never approximated.

## 5. Fixed system

| Component | Setting |
|---|---|
| Document text | `build_doc` from the pilot, unchanged: title, abstract and sections as Markdown |
| Tokenizer | `nomic-ai/nomic-embed-text-v1.5` @ `e9b6763023c676ca8431644204f50c2b100d9aab` (BertTokenizer, lower-case), no special tokens. The same tokenizer sets chunk boundaries, counts budget tokens and checks the input limit. |
| Chunking | token windows `[start, start + size)`; `overlap = round(size × pct / 100)`; `stride = size − overlap`; the last chunk may be shorter. Overlap tokens: 0/13/26 at 128, 0/26/51 at 256, 0/51/102 at 512. This is identical to the pilot. |
| Embedding | `nomic-ai/nomic-embed-text-v1.5` at the same revision; remote code `nomic-ai/nomic-bert-2048` @ `7710840340a098cfb869c4f65e87cf2b1b70caca`; 768 dimensions (no Matryoshka truncation); mean pooling; L2-normalised. Prefixes `search_document: ` and `search_query: `, which are those the model card requires and the pilot used. Any input over 8,192 tokens is an error, not a truncation; the longest expected input is about 520 tokens. |
| Search | Exact dot product (= cosine), restricted to the question's own paper. Ties are broken by chunk position. |
| Not used | LLM generation, reranking, query rewriting, hybrid search |

The pilot's model, prefixes, chunker and search were verified correct (`EXPERIMENT_AUDIT.md` §1). The only
change to the system is pinning revisions.

## 6. Configurations and retrieval modes

There are 9 configurations, `s{size}_o{pct}`: sizes 128, 256 and 512, each at overlaps 0, 10 and 20%.

- **Primary mode**: nominal 2,048-token budget, `K = 2048 / size` = 16 / 8 / 4. The budget is *nominal*:
  - Short final chunks lower the actual count.
  - A paper with fewer than K chunks returns all of them (0 papers at 512 tokens / K = 4).
  - The tokens actually retrieved are recorded per question, and their distribution is reported.
- **Secondary mode**: fixed K = 5. This is descriptive only. It retrieves more text with larger chunks, so
  it answers a different question ("what does a constant-K pipeline get?"). No hypothesis tests are run
  on it.

Both modes use the same ranking.

## 7. Metrics

All metrics are computed on token positions of the paper's tokenization.

- **Evidence tokens.** E_j is the set of token positions of gold span j: tokens whose character range
  intersects the span. E is the union of all E_j.
- **Chunks.** A chunk is a token interval. R is the top-K retrieved chunks; U(R) is the union of their
  token positions.

**Primary: Complete@K.** 1 if for every span j some chunk in R contains all of E_j, else 0. Different spans
may be covered by different chunks.

**Secondary metrics:**

- **Hit@K**: 1 if U(R) ∩ E ≠ ∅, i.e. at least one evidence token is retrieved. Being in the right paper is
  not enough.
- **Evidence Recall@K**: |U(R) ∩ E| / |E|. Unique token positions only: text repeated by overlapping
  chunks, and overlapping gold spans, count once.

**Diagnostics** (descriptive, no tests):

- **Completable**: Complete@K computed over all of the paper's chunks. It is an upper bound on Complete@K
  for that chunking, not a score retrieval is expected to reach.
- **Random Hit@K**: the exact probability that a uniform random K-subset (without replacement) of the
  paper's n chunks hits: `1 − C(n−r, k)/C(n, k)`, k = min(K, n). No sampling and no seed. It is not used
  for selection.
- **MRR@K** and the rank of the first overlapping chunk.
- **Unique evidence precision**: |U(R) ∩ E| / |U(R)|.
- **Redundancy**: 1 − |U(R)| / Σ|c|.
- **Tokens retrieved**: Σ|c|, with overlap counted as sent, plus the unique count |U(R)|.
- **Index measures**: chunk count; chunk-length mean, median, 5th percentile and minimum; duplicate-token
  fraction 1 − document tokens / indexed tokens; float32 index bytes.

**Not reported:** chunk-level precision, any thresholded "all evidence" metric, and the pilot's
character-precision metric, which counts repeated text twice.

The token-position geometry equals the pilot's character geometry for containment: the regression test
shows Completable identical on all 1,635 pilot rows. Recall moves from characters to tokens.

## 8. Statistical analysis (pre-specified)

**Unit.** The paper. For each configuration, each question's metric is averaged within its paper, then
the overall value is the mean over papers. Every configuration is evaluated on the same 100 papers and 287
questions. All comparisons are paired by paper. Every reported value states its number of papers and
questions.

**Primary family.** Complete@K, primary mode, 8 tests of mean per-paper (A − B):

| # | A | B |
|---|---|---|
| 1 | 128 / 10% | 128 / 0% |
| 2 | 128 / 20% | 128 / 0% |
| 3 | 256 / 10% | 256 / 0% |
| 4 | 256 / 20% | 256 / 0% |
| 5 | 512 / 10% | 512 / 0% |
| 6 | 512 / 20% | 512 / 0% |
| 7 | 128 / 0% | 256 / 0% |
| 8 | 256 / 0% | 512 / 0% |

**Test.** A two-sided paired sign-flip permutation test on the 100 per-paper differences d_i, with
statistic |mean(d)|. The test uses Monte Carlo with 100,000 random sign vectors (seed 1729) and
p = (1 + #{|T*| ≥ |T_obs|}) / 100,001. The minimum attainable p is about 1e-5. (Exact enumeration is used
only if n ≤ 16, which does not apply here.)

**Correction.** Holm's step-down procedure across the 8 tests, at α = 0.05. For each test the report gives:

- mean A, mean B, the difference and its 95% CI;
- the raw p and the Holm-adjusted p;
- the paired paper count;
- the counts of papers where A > B, A < B and A = B.

**Confidence intervals.** 95% percentile bootstrap that resamples papers with replacement: 10,000
resamples, seed 2718. This is applied to the per-paper differences for comparisons and to the per-paper
values for each configuration's Complete@K, Hit@K and Recall@K.

**Interpretation rules (binding):**

- p > 0.05 does not mean equivalence or a tie. No configuration is called equivalent or non-inferior,
  because no margin was pre-specified.
- No "winner" is declared from the highest mean. The report states which pre-specified differences
  survive Holm, together with their CIs.
- Wide CIs or non-significant results are reported as inconclusive.
- Secondary metrics, the fixed-K mode, per-type breakdowns and failure analyses are descriptive or
  exploratory, and are labelled so.
- No question-level McNemar test is used.

**Pre-defined exploratory diagnostics.** These do not count towards any test. In each configuration, each
question falls into one category:

- `complete`;
- `unfit`: Completable = 0;
- `ranked_out`: Completable = 1 and Hit = 0;
- `partial`: Hit = 1 and Complete = 0 but Completable = 1.

There is also a breakdown by question type, and worked examples drawn from these categories.

## 9. Sample-size assessment (pilot-based, exploratory)

The SD of per-paper Complete@K differences in the pilot (40 papers, budget mode) is 0.34–0.47 across the
8 comparisons. For n = 100 papers this gives:

- **95% CI half-width** of a difference: about ±0.07–0.09.
- **Detectable difference at 80% power**, with α = 0.05/8 two-sided (conservative for Holm): about
  0.12–0.17 absolute Complete@K.

The pilot's own point estimates were mostly ≤ 0.05 for the overlap comparisons, −0.20 for 128 vs 256, and
+0.03 for 256 vs 512. These estimates are biased by selection and are not targets. **Consequently the
study is well powered only for large effects.** Overlap effects of the pilot's size will probably be
inconclusive. Using all 231 eligible papers would narrow the detectable difference to about 0.08–0.11, at
about 2.3× the compute, and would leave no unused reserve. The sample size is fixed before any fresh
result is seen, and is not changed afterwards.

## 10. Execution, outputs, deviations

Command (from `experiment/`):

```
python run_experiment.py run --sample fresh --out results/fresh_v1 --approved-full-run
python run_experiment.py analyze --run results/fresh_v1
```

**Outputs:**

- `raw/`:
  - `per_question.csv`: every metric, per question, configuration and mode.
  - `rankings.jsonl`: the full ranking, scores, and evidence character and token spans per question and
    configuration.
  - `chunks.csv`: the token and character offsets of every chunk.
  - `engineering.csv`.
  - `questions.jsonl`.
- `aggregated/`: per-paper values, the summary, failure categories.
- `statistics/primary_comparisons.csv`.
- `tables/`, `figures/`.
- `run_metadata.json`: versions, timings, sample lock.
- `config_used.yaml`.

**Checks after the run, before interpretation:**

1. All 9 configurations × 2 modes are present.
2. 287 questions and 100 papers in each configuration, with no duplicates (enforced in code).
3. Independent recomputation of a random sample of 20 question × configuration rows from `rankings.jsonl`
   and `chunks.csv`.
4. Confirmation that the tests use 100 paired papers.

**If a serious error is found after the run starts:** stop and document it. If the error affects results,
the whole run is discarded and re-run from scratch. No partial or selective retention.

**Deviations:** none so far.

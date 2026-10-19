# Experiment audit: "Chunk Size & Overlap: What Actually Changes Retrieval?"

> This is the audit of the **pilot** that led to the clean re-run. It cites pilot scripts, outputs and article
> drafts (`chunk_experiment.py`, `analyze.py`, `results/`, `article_*.md`) that are not in this repository.

Audit date: 2026-10-17. Scope: the original project folder as found. Nothing in the original
folder was modified; all new work is in `experiment/`.

Status labels used below:

- **OK**: verified and correct.
- **FIX**: verified and needs correcting.
- **UNVERIFIED**: evidence is missing.
- **N/A**: not applicable.

"Verified" means I confirmed it from code, data or an executed check, not from the article's description.

## 0. Summary

**The pilot is reproducible.** I re-ran the pilot from the original cached chunk embeddings plus freshly
embedded queries, and all 1,635 question × configuration rows in both modes (equal budget and K = 5)
reproduce the saved `per_question.csv` exactly, on every metric. Every aggregate, confidence interval,
`results/analysis.md` and every article table cell I checked regenerates from the saved raw files. The
implementation is sound and was salvaged almost entirely.

**The validation was not clean.** The 31 "held-out" questions were scored and printed before the grid was
extended to 64 tokens, and before the selection rule was finalised. The article itself says the
both-metrics selection rule was adopted because the one-metric rule's choice failed on held-out. The
held-out set is therefore not an independent test of the chosen configuration or of the grid. All pilot
numbers are exploratory.

**The statistics need replacing.**

- Configurations are compared with question-level exact McNemar tests, which treat questions from the same
  paper as independent.
- Means weight papers by how many questions they have.
- "Not significantly different" was used as "tied", i.e. a non-significant test was treated as equivalence.
- 45 tests were run with no correction applied.

**Three metric names or definitions need tightening:**

- Complete@K with several spans.
- "Character precision", which counts repeated text twice.
- Evidence recall, which is character-based.

The values are computed as the code says. The issue is how the article describes them.

## 1. What was actually run (reconstructed and verified)

| Item | Finding | Evidence | Status |
|---|---|---|---|
| Data source | QASPER v0.3 **dev** file only (`data/qasper_raw/qasper-dev-v0.3.json`). Train was not used. | `prepare_qasper.py` usage; regenerating with `--qasper ...dev... --papers 40 --max-per-paper 5 --seed 0` reproduces `eval.jsonl` and all 40 `corpus/*.md` byte for byte. The script docstring says `--papers 30` (`prepare_qasper.py:30`), but the data was built with 40. | OK (docstring FIX) |
| Papers / questions | 40 papers, 109 questions (54 factoid, 42 multi-sentence, 13 multi-part); 1–5 questions per paper. The draw looked at 42 papers; 2 were skipped because they had no usable question. | `count_funnel.py` replay = MATCH; my own replay | OK |
| Question funnel | 139 questions in the 40 kept papers. Dropped: 22 table/figure only, 5 unanswerable, 1 no highlight, 1 evidence not verbatim, 1 over the 5-per-paper cap. | `count_funnel.py` output | OK |
| Dev / held-out split | By paper: 28 dev papers / 78 questions, 12 held-out papers / 31 questions. No paper is in both. Uses `rng.sample` on the same RNG after the shuffle (`prepare_qasper.py:138-139`). | eval.jsonl | OK as a split; see §2 for how it was used |
| Train/dev mixing | None: only the dev file was read. | code | OK |
| Evidence | First annotation (in `answers` order) that is answerable, has highlights, has no `FLOAT SELECTED` highlight, and whose highlights are all verbatim in the rebuilt text (`prepare_qasper.py:71-82`). Other annotators' evidence is ignored. Duplicate highlights are removed. | code | OK, documented as a design choice |
| Evidence → offsets | `text.find(ev)`, i.e. the first occurrence (`chunk_experiment.py:94`). A warning is printed if the string occurs more than once. In the pilot no evidence string occurs twice, no two spans of a question overlap, and none is shorter than 20 characters. | my check over eval.jsonl | OK for this data |
| Tokenizer | `nomic-ai/nomic-embed-text-v1.5`'s own BertTokenizer, `add_special_tokens=False`, offsets mapping (`chunk_experiment.py:123-126`). Revision `e9b6763…` reproduces all 15 chunkings exactly (see next rows). Total 205,160 tokens, matching the article. | hash check below | OK |
| Chunking | Fixed token windows; `stride = size − overlap`; last chunk may be short (`chunk_experiment.py:160-170`). Overlap = `round(size × pct / 100)` tokens (`:333`), e.g. 128/10% → 13, 256/20% → 51. Chunk text = characters from the first token's start to the last token's end; boundaries can fall inside a word (sub-word tokens). | code; unit tests | OK |
| Embedding model | `nomic-ai/nomic-embed-text-v1.5`, 768-d, mean pooling, L2-normalised. Model revision was **not recorded**. The current revision `e9b6763023c6…` reproduces the cached chunk embeddings (cosine 1.000000 on 80 re-embedded chunks) and every ranking. | cache filenames; re-embedding check | Revision UNVERIFIED for the original run; equivalent in effect |
| Prefixes | Document prefix `search_document: `: **verified**, because recomputing the cache key (`chunk_experiment.py:271`) with that prefix matches all 15 cache files, and with `""` matches none. Query prefix `search_query: `: **verified** by reproduction (1,635/1,635 rows match with it; 555 and 680 of 1,635 match without it). These are the prefixes the model card requires. The model ships no default prompts, so they had to be passed by hand. | executed checks | OK |
| Truncation | The model's `max_seq_length` is 8,192. The longest original input (a 1,024-token chunk plus prefix and special tokens) is 1,032 tokens. Nothing was truncated. The script only warns, and only checks chunk size, not prefix and special tokens (`:315-319`). | executed check | OK |
| Similarity / ranking | Dot product of normalised vectors = cosine, exact. Each question is ranked only against its own paper's chunks (`--scope doc`, `:345-350`). The log confirms `scope=doc`. `np.argsort` is not a stable sort, but no ties are expected with float scores. | code; run_log.txt | OK |
| Token budget | `K = budget // size` (`:336`). "Tokens / query" is the sum of the chunks' document-token counts. It is below 2,048 when final chunks are short or a paper has fewer than K chunks. The budget is nominal. | code; logs | OK (article already says so) |
| Run commands | Not recorded. The flags for model, prefixes, scope and trust-remote-code are absent from the logs. Reconstructed and verified by reproduction: `--model nomic-ai/nomic-embed-text-v1.5 --trust-remote-code --doc-prefix "search_document: " --query-prefix "search_query: " --scope doc --sizes 64 128 256 512 1024 --overlaps 0 10 20`, then `--k 5` or `--token-budget 2048`, each with `--split all/dev/heldout`. | | FIX: record commands |
| Environment | No requirements file. Library versions of the original run are unknown. The current environment reproduces the results: Python 3.14.2, torch 2.12.0+cpu, transformers 5.12.1, sentence-transformers 6.1.0. | | UNVERIFIED for the original; FIX for the new run |
| Seeds | Data draw seed 0; bootstrap seed 0 with 1,000 resamples. Embedding on CPU is deterministic in practice. | code | OK |

## 2. Held-out evaluation: how it was actually used

The timeline comes from file modification times. They are consistent with each other, but an mtime is not
proof of intent.

| Time (2026-10-11) | Event |
|---|---|
| 15:36 | `eval.jsonl` built (dev/held-out labels assigned) |
| 15:50 → 17:19 | 12 configurations (128–1,024) embedded |
| **17:25** | `run_log_v1.txt` written. It contains the **full held-out results** (`--split heldout`) for all 12 configurations, in both modes. |
| 19:19 | `chunk_experiment.py` changed (it now writes `per_question.csv`; the v1 log's "Wrote" line lacks it) |
| 19:31 → 19:58 | 64-token configurations added and embedded |
| 20:00 → 20:11 | Full re-run of 15 configurations × 3 splits × 2 modes |
| 20:13 | `analyze.py` last modified (selection rule, 45 tests), 1 minute later `results/analysis.md` |

Did the held-out results influence each decision?

| Decision | Influenced by held-out? | Basis |
|---|---|---|
| Chunk-size grid (64 added) | **Exposed.** The article justifies 64 by "the best Hit@K in the planned grid landed at its 128-token edge". That is true on dev alone too, but the held-out numbers had already been printed (17:25) and the `--split all` numbers that include held-out were also visible. Independence cannot be shown. | article line 174; logs |
| Overlap grid | Unchanged from the original article draft (0/10/20). | `article_original.md` |
| Metric definitions | Every metric in the final article was already computed by the code at the first logged run: the v1 log at 17:25 prints `rand` and `max`, i.e. Random Hit@K and Completable. The article draft `article_original.md` (saved 19:04, after v1) lists Hit, Complete, recall, MRR, char precision, redundancy and tokens, but not Random Hit@K or Completable; those reached the article only after results existed. Whether any metric was added to the code after an unlogged earlier run cannot be established. | Metric code: no evidence of influence. Article emphasis on the floor/ceiling: **UNVERIFIED** |
| Selection criterion | **Yes.** The article states that the one-metric rule picked 1,024/0%, which "then scores 0.42 Hit@K on held-out, against a random 0.42", and the rule was then changed to "tied on both metrics". `article_updates.md` says the same. The final rule was chosen in the light of held-out results. | article lines 238, 403 |
| Preferred configuration (128/10%) | **Yes**, through the changed rule above. | |
| Framing / conclusions | **Yes.** The takeaways cite the held-out failure as a lesson, and the all-109-question tables mix dev and held-out. | article lines 396-404 |

**Conclusion.** The 31 held-out questions are **not** an untouched test set for the grid, the selection
rule, the chosen configuration or the conclusions. All pilot results, held-out included, should be treated
as exploratory. Re-running the same configurations on the same 40 papers cannot fix this. The fresh
evaluation (`EXPERIMENT_PROTOCOL.md`) uses 100 papers that the pilot never drew.

## 3. Metric audit

Implementation: `chunk_experiment.py:196-246` (`score_query`). Each metric was checked against the
article's metrics table (`article_final.md:142-152`).

| Metric | Code definition | Article definition | Verdict |
|---|---|---|---|
| Hit@K | A top-K chunk's character range overlaps any evidence span by ≥ 1 character, in the same document (`:198-205`). | "Did any of the top K chunks touch the evidence?" | OK. Correct-paper-only does not count. |
| Complete@K | **Every** span lies wholly inside **some** top-K chunk; different spans may sit in different chunks (`:191-193, 209`). | "Did one retrieved chunk contain the entire evidence span?" | FIX the wording. For the 19 multi-span questions, the article reads as "one chunk holds all evidence", which the code does not require. |
| Completable | Same test over every chunk of the paper (`:211`). | ceiling for Complete@K | OK. An upper bound under this metric, not a guarantee. |
| Evidence recall@K | Characters: for each span, the union of its overlap with top-K chunks, summed over spans, divided by the summed span lengths (`:218-223`). Overlapping chunks are merged within a span, so they do not double-count. | "fraction of the evidence text" | OK for this data. If two gold spans overlapped, the denominator would double-count (0 cases in the pilot). Character-based: the new run uses unique token positions. |
| Random Hit@K | Exact `1 − C(n−r, k)/C(n, k)`, with k = min(K, n), n = chunks in the paper, r = chunks overlapping the evidence (`:214-215`). It equals the probability that a uniform random K-subset hits. | same formula | OK. It is an exact expectation, so no seeds are involved. Verified against Monte Carlo in the unit tests. |
| MRR@K | 1 / rank of the first overlapping chunk if it is within K, else 0. | same | OK |
| Character precision | Σ over top-K chunks of their evidence characters ÷ Σ of their lengths (`:227-228`). Text shared by overlapping chunks is counted in both numerator and denominator. | "fraction of retrieved text that is evidence" | FIX the name: it is occurrence-weighted precision. The new run reports a unique-position version under an explicit name. |
| Redundancy | 1 − unique retrieved characters / total retrieved characters. | same | OK |
| Chunk-level precision | relevant chunks / K (`:241`). In `results.csv` but not used by the article. | article argues against it | N/A |
| "All evidence@K" (recall ≥ 0.99) | Not present in the code or the article. | | N/A |
| Aggregation | Plain mean over questions (`:369-372`), so papers with 5 questions weigh 5× papers with 1. | not stated | FIX: the new run averages within each paper first, then across papers |

Edge cases checked:

- Spans split across two retrieved chunks: Hit = 1, Complete = 0, and recall counts both halves.
- Fewer chunks than K: `k = min(K, n)` and random hit = 1. Affects 62 of 109 questions at 1,024/K = 5, which the article notes.
- Short final chunks: they are counted at their real length.

The pilot data contains no duplicate evidence text and no overlapping gold spans. Questions without usable
text evidence were excluded during preparation. The unit tests in `experiment/tests/test_core.py`
reproduce each case on synthetic data.

## 4. Statistics audit

| Item | Finding | Status |
|---|---|---|
| 95% CIs | Cluster bootstrap over papers, 1,000 resamples, seed 0 (`:249-264`). It resamples whole papers but estimates the **question-weighted** mean (ratio of sums), consistent with the reported means. All CIs regenerate exactly. | OK as implemented; replaced by the paper-mean bootstrap |
| Paired tests | Exact McNemar (binomial on discordant questions), `analyze.py:38-45`. It treats questions as independent, but questions about the same paper are correlated. The article calls it "stricter and more appropriate" than CI overlap; it is paired, but it ignores the clustering. | FIX |
| Multiplicity | 45 tests (`analyze.py:115-125`): both modes, both metrics, plus 5 ad-hoc pairs that include 64-token comparisons added after the results were seen. A Bonferroni threshold is printed but not applied. The article correctly calls the two p < 0.05 results "suggestive" and says neither survives correction. | FIX: one pre-specified family with Holm |
| "Tied" = p > 0.05 | `analyze.py:82,92`. A non-significant test is used as equivalence. The article itself warns against this in bold, then uses a two-metric version of the same rule. | FIX: no equivalence claims without a margin |
| Selection on dev | The computation uses only dev rows. But the rule was revised after held-out was seen (§2). | FIX (see §2) |

## 5. Can the article be regenerated?

| Output | Regenerable? | Notes |
|---|---|---|
| `results/*/results.csv`, `results_table.md` | Yes | Aggregates and CIs recomputed from `per_question.csv`: 0 mismatches. dev and held-out files equal the corresponding subsets of the all-question file. |
| `results/analysis.md` | Yes | `analyze.py` re-run in a scratch copy: byte-identical |
| `per_question.csv` from scratch | Yes | 1,635/1,635 rows in both modes, from cached chunk embeddings + re-embedded queries |
| Chart | Yes | `analyze.py:156-205` |
| Big results tables in the article | Yes | All cells match `results.csv` under the original `:.2f` formatting |
| By-type table, boundary counts, failure table, "62 of 109", miss ranks | Yes | Recomputed from `per_question.csv` / `results_by_type.csv`. They were never written to a file by a script: they were copied by hand. |
| Evidence length (median 44, p90 131, single-sentence 39 tokens) | Yes by recomputation (43.5 / 130.8 / 38.5) | No script in the repo; FIX: generate it |
| Failure examples | Partly | They come from `failures.jsonl`. Rank details need per-question ranks, which are only partially saved. |

## 6. What can be salvaged

| Component | Current status | Problem | Proposed action |
|---|---|---|---|
| Dataset processing | Correct and reproducible | No recorded command; docstring says 30 papers | Reuse `build_doc` / `pick_answer` unchanged (copied, with a regression test asserting identity on all 281 dev papers). Add 2 explicit exclusion rules (ambiguous duplicate evidence, duplicate question); neither removes anything in the pilot or in the fresh sample. |
| Paper/question splitting | Paper-level split was correct | Held-out exposed before decisions were final | Retire the dev/held-out labels; treat all 40 pilot papers as exploratory; draw 100 fresh, never-drawn dev papers |
| Chunking | Correct | None | Reuse the identical windowing and overlap rounding; unit-tested |
| Embeddings | Correct model and prefixes; no truncation | Revisions and command not recorded; cache key lacks the revision | Pin model, remote code and tokenizer revisions; hard-fail on over-long input; cache key includes revisions, prefixes and library versions |
| Retrieval | Correct (exact, per paper) | Unstable sort (immaterial) | Same; stable tie-break |
| Metrics | Hit, Complete, Completable, Random Hit, recall correct for this data | Wording of Complete@K for multi-span; char precision counts repeats; char-based recall; question weighting | Token-position definitions (geometry identical: the regression test shows Completable equal on all 1,635 pilot rows); unique-position precision; paper-level aggregation |
| Statistics | Bootstrap fine; McNemar unclustered; 45 uncorrected tests; equivalence fallacy | | Paired paper-level permutation test, 8 pre-specified tests, Holm, paired paper bootstrap; no "tied" |
| Figures and tables | Regenerable but partly hand-copied | Some numbers have no script | Every table and figure produced by `reporting.py` from raw outputs |
| Article | Clear and largely honest about limitations | Mixes dev and held-out; claims validation; McNemar description; 64 and 1,024 results; Complete@K wording | Rewrite methods, results and conclusions after the fresh run; keep the background and failure-analysis sections; label the pilot as exploratory |

The article's reported numbers are unchanged in this phase.

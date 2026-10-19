# Audit guide: chunk size & overlap, fresh_v1

Model weights, embedding caches and the QASPER source file are not included.

The article itself is published separately and is not part of this repository.

## 1. Fresh-run results: `experiment/results/fresh_v1/`

- `raw/`: `per_question.csv` (every metric, per question × configuration × mode), `rankings.jsonl` (full ranking, scores, evidence character and token spans), `chunks.csv`, `questions.jsonl`, `engineering.csv`
- `aggregated/`: `per_paper.csv`, `summary.csv`, `failure_categories.csv`, and `per_paper_canonical_ids.csv`. `per_paper.csv` is exactly as the frozen pipeline wrote it, so 8 paper IDs lack trailing zeros (`1604.004` for `1604.00400`). `per_paper_canonical_ids.csv` is a post-run export by `experiment/export_canonical_ids.py`: the same rows and values with the canonical IDs. Use it to join per-paper results to the manifests
- `statistics/primary_comparisons.csv`: the 8 tests, with raw and Holm-adjusted p
- `tables/`, `figures/`: includes the original `complete_by_config.png` and `complete_by_config_upper_bound.png`, the same figure with only its footnote changed from "ceiling" to "chunk-fit upper bound". The new one was drawn by `experiment/replot_complete_by_config.py`, which reproduces the original pixel for pixel when given the old footnote.
- `config_used.yaml`, `run_metadata.json`

Logs, in `experiment/results/`:
- `fresh_v1_relaunch.log`: the run that produced fresh_v1
- `fresh_v1_verify.log`: the protocol's verify step (20 random rows, 0 mismatches)
- `fresh_v1_verify_full.log`: the same recomputation on all 5,166 rows, 0 mismatches
- `fresh_v1_verify_all_fields.log`: output of `experiment/verify_all_fields.py`, added after the audit. It checks every column of every row of `per_question.csv` (the 16 metric fields, the structural fields and, since 2026-10-19, the `sample` column), plus the aggregates, the canonical-ID export and the statistics: 137,175 checks, 0 mismatches. It warns that `per_paper.csv` writes 8 paper IDs without trailing zeros; the values are unaffected (see the protocol's §10 log). Mutation tests (planted errors in per-question fields, chunk boundaries, a p-value, the `sample` column and a canonical ID) were caught.
- `fresh_v1_interrupted/`: the first launch, which was stopped overnight before any metric was computed. It contains only a partial `rankings.jsonl` and a log, and it was not analysed. See the protocol's §10 operational log.

## 2. Protocol, manifests and pilot

| File | What it is |
|---|---|
| `protocol_history/EXPERIMENT_PROTOCOL_at_launch.md` | The protocol as frozen at launch (see below) |
| `experiment/EXPERIMENT_PROTOCOL.md` | The current protocol |
| `protocol_history/protocol_launch_vs_current.diff` | The only differences: the four appended §10 operational-log entries (overnight stop; full-field verifier; figure wording; `sample` check and canonical-ID export) |
| `experiment/manifests/frozen_at_launch.sha256` | SHA-256 of the protocol, config, manifests and all code, written at launch |
| `experiment/manifests/paper_manifest.csv` | All 281 dev papers, each with a status (`pilot_used` 40, `pilot_drawn_skipped` 2, `eligible` 231, `excluded_no_eligible_question` 8) and `selected` |
| `experiment/manifests/question_manifest.csv` | Every question in every paper, with its exclusion reason and `selected` |
| `experiment/manifests/sample_lock.json` | The seed, the procedure and the manifest hashes |
| `fresh_exclusion_counts.csv` | Question statuses within the 100 selected papers: 287 eligible, 32 table/figure only, 20 unanswerable, 17 over the cap, 10 not verbatim, 6 no highlight |
| `eval.jsonl` | The pilot's 109 questions. Its 40 paper IDs (the `doc` field) are the pilot paper IDs |
| `experiment/configs/clean_rerun.yaml` | The frozen settings, including the 8 comparisons |
| `experiment/run_experiment.py`, `experiment/src/` | The code, so the hashes in `frozen_at_launch.sha256` can be checked |
| `experiment/EXPERIMENT_AUDIT.md`, `SMOKE_TEST_REPORT.md`, `CHANGES.md`, `README.md` | Supporting documents |
| `experiment/requirements.txt`, `experiment/tests/`, `experiment/.gitignore` | The pinned environment, the 36 unit and regression tests (`python -m unittest discover -s tests -t .` from `experiment/`; 36 passed again on 2026-10-19 with the QASPER file in place; from this bundle alone, 29 run and pass and the 7 pilot-regression tests skip with "QASPER dev file not present" until it is downloaded as described in `experiment/README.md`), and the ignore rules for `cache/` and `results/` |

### How the at-launch protocol was recovered

No copy of the protocol was saved before the §10 note was appended. `EXPERIMENT_PROTOCOL_at_launch.md` was rebuilt by replaying, from the session transcript, the tool calls that wrote the protocol (2026-10-16 21:47:04 UTC) and that edited its status line to APPROVED/FROZEN (22:05:34 UTC). The SHA-256 of the result is `249c8c7b…cfeb0a`. That equals the `EXPERIMENT_PROTOCOL.md` line in `frozen_at_launch.sha256`, which was written at 22:05:36 UTC, just before launch.

The later changes are four appended operational-log entries. The first was added at 23:43 UTC (04:43 local time), after the first launch was stopped. The other three were added after the audit: one records the new verifier, one the redrawn figure, and one the verifier's `sample` check and the canonical-ID export. That write also converted the file's line endings to CRLF, which is why the current file's hash differs. Every other file in `frozen_at_launch.sha256` still matches.

The 8 comparisons are in §8 of the at-launch protocol and in `configs/clean_rerun.yaml` (unchanged since launch).

### Pilot overlap check

The 40 `pilot_used` IDs in `paper_manifest.csv` are exactly the 40 papers in the pilot's `eval.jsonl`. The 100 `selected` papers include none of them and none of the 2 `pilot_drawn_skipped` papers.

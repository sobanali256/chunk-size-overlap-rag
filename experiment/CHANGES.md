# Change record

All new work is in `experiment/`. Nothing in the original project folder was modified, moved or deleted.
The folder is not a git repository, so isolation is by directory. I did not run `git init` without
asking.

## Outside the project folder

- Downloaded into the Hugging Face cache (`~/.cache/huggingface/hub`), at pinned revisions:
  - `nomic-ai/nomic-embed-text-v1.5` @ `e9b6763…` (547 MB weights, plus tokenizer and configs);
  - `nomic-ai/nomic-bert-2048` @ `7710840…` (remote modeling code, inspected before execution).

  The model was needed both to verify the pilot's query prefix and for the new run.
- Scratch copies for the audit (regenerated corpus, re-run of `analyze.py`) are in the session scratch
  folder, not in the project.

## Files created

| File | Why |
|---|---|
| `EXPERIMENT_AUDIT.md`, `EXPERIMENT_PROTOCOL.md`, `SMOKE_TEST_REPORT.md`, `README.md`, `CHANGES.md` | Required deliverables |
| `configs/clean_rerun.yaml` | Single source of settings |
| `manifests/paper_manifest.csv`, `question_manifest.csv`, `sample_lock.json` | Frozen fresh sample and every exclusion |
| `src/data.py` | Pilot `build_doc`, `pick_answer`, `classify`, `answer_text` copied unchanged; explicit exclusion reasons; pilot-draw replay; nested seeded sampling |
| `src/chunking.py` | Pilot windowing, unchanged, as token intervals |
| `src/embeddings.py` | Pinned model, code and tokenizer revisions; hard input-limit check; complete cache key |
| `src/retrieval.py` | Exact per-paper cosine; stable tie-break; K per mode |
| `src/metrics.py` | Token-position metrics; unique-position recall and precision |
| `src/statistics.py` | Paper means, paired bootstrap, sign-flip permutation, Holm |
| `src/pipeline.py`, `src/reporting.py`, `src/verify.py`, `run_experiment.py` | Run, analysis, independent check |
| `tests/` | Unit, regression and fixture tests |
| `requirements.txt`, `.gitignore` | Environment; keep `cache/` and `results/` out of version control |

## Decisions that differ from the pilot

The reasons are given in the audit and protocol.

1. **Fresh papers.** 100 papers the pilot never drew, instead of the dev/held-out split.
2. **Grid.** 128/256/512 × 0/10/20 instead of adding 64 and 1,024 (your specification).
3. **Aggregation.** Per-paper mean, then mean over papers, instead of a question-weighted mean.
4. **Tests.** A paired paper-level sign-flip permutation test with Holm over 8 pre-specified tests,
   replacing 45 question-level McNemar tests with no correction applied.
5. **Interval estimates.** A paper bootstrap of per-paper means (10,000 resamples) instead of a cluster
   bootstrap of question means (1,000).
6. **Evidence recall.** Unique token positions instead of characters.
7. **Precision.** "Character precision" (occurrence-weighted) is replaced by unique-position evidence
   precision.
8. **Selection rule.** No "tied if p > 0.05" rule and no configuration selection; equivalence is never
   claimed.
9. **Exclusion rules.** Two new rules (ambiguous duplicate evidence, duplicate question). Each removes 0
   questions in the pilot and in the fresh sample.
10. **Revisions.** Pinned model, remote-code and tokenizer revisions. The original's are unknown, but the
    current ones reproduce its embeddings.

## Unchanged from the pilot

- Dataset file and document builder.
- Answer-selection and eligibility rules 1–4 and the 5-per-paper cap.
- Tokenizer, overlap rounding and windowing.
- Model and prefixes.
- Exact per-paper cosine search.
- The 2,048 nominal budget and K = 5.
- The Hit, Complete and Completable geometry, and the exact Random Hit formula.

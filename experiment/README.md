# Chunk size & overlap: clean re-run

This folder is a reproducible re-run of the chunk-size / overlap retrieval experiment on a fresh sample of
QASPER papers. The original pilot files in the parent folder are left untouched.

| File | What it is |
|---|---|
| `EXPERIMENT_AUDIT.md` | Audit of the original pilot: what was run, what was verified, what has to change |
| `EXPERIMENT_PROTOCOL.md` | The frozen design and statistical analysis plan |
| `SMOKE_TEST_REPORT.md` | Tests executed before the full run |
| `CHANGES.md` | Record of the files created and the decisions taken |
| `configs/clean_rerun.yaml` | The only place settings live |
| `manifests/` | Frozen fresh sample: `paper_manifest.csv` (all 281 dev papers with status), `question_manifest.csv` (every question with its eligibility reason), `sample_lock.json` (hashes) |
| `src/` | `data` (eligibility, sampling), `chunking`, `embeddings` (cache), `retrieval`, `metrics`, `statistics`, `pipeline`, `reporting`, `verify` |
| `tests/` | Unit tests, pilot regression tests, the synthetic fixture |
| `run_experiment.py` | Entry point |
| `verify_all_fields.py` | Post-run check of every per-question column, the aggregates and the statistics (added after fresh_v1; not a frozen file) |
| `export_canonical_ids.py` | Post-run export of `per_paper.csv` with canonical paper IDs, as `per_paper_canonical_ids.csv` (added after fresh_v1) |
| `replot_complete_by_config.py` | Redraws the Complete@K figure with the "chunk-fit upper bound" footnote (added after fresh_v1) |

## Setup

Requirements: Python 3.11 or later (tested on 3.14.2) and the packages in `requirements.txt`. The tested
versions are listed there.

The QASPER v0.3 train/dev release must be at `../data/qasper_raw/`:

```
curl -L -o ../data/qasper_raw/qasper.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar -xzf ../data/qasper_raw/qasper.tgz -C ../data/qasper_raw
```

The first model load downloads `nomic-ai/nomic-embed-text-v1.5` (about 547 MB) and its remote code, both
at the revisions pinned in the config. The model needs `trust_remote_code`. The remote code was inspected
and contains no network or subprocess calls.

## Commands (run from this folder)

```
python -m unittest discover -s tests -t .                                  # unit + regression tests
python run_experiment.py run --sample fixture --out results/fixture         # synthetic smoke test, seconds
python run_experiment.py run --sample pilot --limit-papers 5 --out results/pilot5   # real model, pilot papers
python run_experiment.py analyze --run results/pilot5
python run_experiment.py verify  --run results/pilot5

# full confirmatory run, ONLY after the protocol is approved:
python run_experiment.py run --sample fresh --out results/fresh_v1 --approved-full-run
python run_experiment.py analyze --run results/fresh_v1
python run_experiment.py verify  --run results/fresh_v1
```

`manifest` rebuilds the sample from scratch. It refuses to run while `manifests/sample_lock.json` exists.

## Caching

Chunk and query embeddings are cached in `cache/`. The cache key covers:

- model id and revision, remote-code revision, tokenizer revision;
- prefix, normalisation and max length;
- library versions;
- every input text.

A change to any of these, including any change to chunk boundaries, gives a new key. Delete `cache/` at
any time; it only saves time. `cache/` is a generated file, so do not commit it. `results/` is committed for the published runs (`fresh_v1` and the smoke-test runs); new runs can be deleted or kept locally.

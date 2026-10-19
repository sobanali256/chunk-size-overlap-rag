# Chunk size & overlap in RAG retrieval

Code, frozen protocol, data manifests and results for the article *Chunk Size & Overlap: What Actually
Changes Retrieval?*, which is published separately.

The study compares fixed-size chunks of 128, 256 and 512 tokens at 0%, 10% and 20% overlap. It uses 100
randomly sampled QASPER papers and 287 questions, with `nomic-embed-text-v1.5` embeddings and exact cosine
search. The sample, the 8 comparisons, the metrics and the tests were frozen before any retrieval was run.

**Main result.** Under an equal 2,048-token budget, 128-token chunks at 0% overlap delivered complete
evidence (Complete@K) less often than 256-token chunks: 0.323 vs 0.482. The difference is −0.159, with a
95% CI of −0.229 to −0.088 and a Holm-adjusted p of 0.0002. The 512-vs-256 comparison and all six overlap
comparisons were inconclusive after Holm correction. Inconclusive does not mean the configurations are
equivalent.

## Where things are

| Path | What it is |
|---|---|
| [`experiment/`](experiment/) | **The study.** Code, config, protocol, manifests, tests and results. Start with [`experiment/README.md`](experiment/README.md) |
| [`experiment/EXPERIMENT_PROTOCOL.md`](experiment/EXPERIMENT_PROTOCOL.md) | The frozen design and analysis plan, with a §10 log of every post-launch change |
| [`experiment/results/fresh_v1/`](experiment/results/fresh_v1/) | The confirmatory run: raw per-question results, rankings, aggregates, statistics, figures |
| [`README_AUDIT.md`](README_AUDIT.md) | File-by-file guide for auditors: verification logs, hashes, how the at-launch protocol was recovered |
| [`protocol_history/`](protocol_history/) | The protocol as frozen at launch, and its diff against the current version |

### Pilot files

The study began with a 40-paper pilot. An audit found that its held-out questions had shaped the
grid and the selection rule, so its numbers are exploratory and are not reported. Only the pilot inputs that
the experiment code and its regression tests load are kept: `eval.jsonl` (its papers are excluded from the
fresh sample), `corpus/`, `prepare_qasper.py`, `count_funnel.py`, and two pilot result files in `results/budget2048/` that
the regression tests compare against.

## Reproducing

See [`experiment/README.md`](experiment/README.md). In short, from `experiment/`:

```
pip install -r requirements.txt
# download QASPER v0.3 train/dev to ../data/qasper_raw/ (command in experiment/README.md)
python -m unittest discover -s tests -t .
python run_experiment.py run --sample fresh --out results/fresh_v1_repro --approved-full-run
python run_experiment.py analyze --run results/fresh_v1_repro
python verify_all_fields.py --run results/fresh_v1
```

The embedding caches and the QASPER source file are not in the repository. The caches are regenerated on
the first run.

## Data

QASPER (Dasigi et al., 2021, [ACL Anthology](https://aclanthology.org/2021.naacl-main.365/)) is released by
the Allen Institute for AI under CC BY 4.0. Question text and evidence spans in `eval.jsonl`, `corpus/` and
`experiment/results/*/raw/questions.jsonl` are derived from it.

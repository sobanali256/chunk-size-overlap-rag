#!/usr/bin/env python3
"""
run_experiment.py - clean re-run of the chunk size / overlap experiment.

Run from this experiment/ folder. Stages:

  python run_experiment.py manifest
      Build and freeze the fresh paper sample (no retrieval). Writes manifests/ and
      manifests/sample_lock.json. Refuses to overwrite an existing lock.

  python run_experiment.py run --sample fixture  --out results/fixture       # synthetic, hashing embedder
  python run_experiment.py run --sample pilot    --out results/pilot_repro   # pilot papers, real model
  python run_experiment.py run --sample fresh    --out results/fresh_v1 --approved-full-run
      Chunk, embed, retrieve and score. The fresh sample requires the lock file to match and the
      explicit --approved-full-run flag.

  python run_experiment.py analyze --run results/fresh_v1
      Paper-level aggregation, the 8 pre-specified tests, tables and figures.

  python run_experiment.py verify --run results/fresh_v1
      Independently recompute 20 random rows from the raw rankings and chunk offsets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src import data as D                       # noqa: E402
from src.pipeline import EvalQuestion, run      # noqa: E402
from src.reporting import analyze               # noqa: E402

MANIFESTS = HERE / "manifests"
LOCK = MANIFESTS / "sample_lock.json"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ manifest

def build_manifest(cfg: dict):
    if LOCK.exists():
        sys.exit(f"{LOCK} exists: the sample is frozen. Delete it only if the protocol is re-opened.")
    ds = cfg["dataset"]
    qpath = (HERE / ds["file"]).resolve()
    data = D.load_qasper(qpath)
    drawn = D.pilot_drawn_papers(data, ds["pilot"]["replay"]["n_papers"], ds["pilot"]["replay"]["seed"])
    pilot_eval = [json.loads(x) for x in (HERE / ds["pilot"]["eval_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    pilot_used = sorted({r["doc"][:-3] for r in pilot_eval})
    if not set(pilot_used) <= set(drawn):
        sys.exit("pilot replay does not cover the papers in eval.jsonl; cannot establish independence")
    pilot_titles = {D.norm_title(data[p]["title"]) for p in drawn}

    paper_rows, q_rows, eligible = [], [], []
    for pid in sorted(data):
        p = data[pid]
        text, paras = D.build_doc(p)
        qs = D.paper_questions(pid, p, text, paras, ds["max_questions_per_paper"])
        n_ok = sum(q.status == "eligible" for q in qs)
        if pid in pilot_used:
            status = "pilot_used"
        elif pid in drawn:
            status = "pilot_drawn_skipped"
        elif D.norm_title(p["title"]) in pilot_titles:
            status = "excluded_same_title_as_pilot"
        elif not p["full_text"]:
            status = "excluded_no_full_text"
        elif n_ok == 0:
            status = "excluded_no_eligible_question"
        else:
            status = "eligible"
            eligible.append(pid)
        paper_rows.append({"paper_id": pid, "title": p["title"].strip(), "status": status,
                           "n_qas": len(p["qas"]), "n_eligible_questions": n_ok,
                           "doc_chars": len(text), "doc_sha256": hashlib.sha256(text.encode()).hexdigest()})
        for q in qs:
            q_rows.append({"paper_id": pid, "question_id": q.qasper_question_id, "index_in_paper": q.index_in_paper,
                           "status": q.status, "type": q.type, "n_spans": len(q.spans),
                           "evidence_mapping": "verbatim_unique" if q.status in ("eligible", "over_cap") else "",
                           "question": q.question})

    sample = set(D.draw_sample(eligible, ds["sample"]["n_papers"], ds["sample"]["seed"]))
    for r in paper_rows:
        r["selected"] = r["paper_id"] in sample
    for r in q_rows:
        r["selected"] = r["paper_id"] in sample and r["status"] == "eligible"
    MANIFESTS.mkdir(exist_ok=True)
    for name, rows in (("paper_manifest.csv", paper_rows), ("question_manifest.csv", q_rows)):
        with open(MANIFESTS / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    lock = {"qasper_file": str(Path(ds["file"])), "qasper_sha256": D.sha256_file(qpath),
            "n_papers_requested": ds["sample"]["n_papers"], "seed": ds["sample"]["seed"],
            "procedure": ds["sample"]["procedure"], "n_eligible_unused_papers": len(eligible),
            "n_selected_papers": len(sample),
            "n_selected_questions": sum(r["selected"] for r in q_rows),
            "paper_manifest_sha256": D.sha256_file(MANIFESTS / "paper_manifest.csv"),
            "question_manifest_sha256": D.sha256_file(MANIFESTS / "question_manifest.csv")}
    LOCK.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    print(json.dumps(lock, indent=1))


# ------------------------------------------------------------------- samples

def load_fresh(cfg):
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for key, name in (("paper_manifest_sha256", "paper_manifest.csv"), ("question_manifest_sha256", "question_manifest.csv")):
        if D.sha256_file(MANIFESTS / name) != lock[key]:
            sys.exit(f"{name} does not match {LOCK}: the frozen sample was modified")
    qpath = (HERE / cfg["dataset"]["file"]).resolve()
    if D.sha256_file(qpath) != lock["qasper_sha256"]:
        sys.exit("QASPER file differs from the one the sample was drawn from")
    data = D.load_qasper(qpath)
    sel = [r for r in csv.DictReader(open(MANIFESTS / "question_manifest.csv", encoding="utf-8")) if r["selected"] == "True"]
    papers, questions = {}, []
    for pid in sorted({r["paper_id"] for r in sel}):
        text, paras = D.build_doc(data[pid])
        papers[pid] = text
        qs = {q.index_in_paper: q for q in D.paper_questions(pid, data[pid], text, paras,
                                                            cfg["dataset"]["max_questions_per_paper"])}
        for r in (x for x in sel if x["paper_id"] == pid):
            q = qs[int(r["index_in_paper"])]
            assert q.status == "eligible" and q.qasper_question_id == r["question_id"]
            questions.append(EvalQuestion(pid, f"{pid}:{q.qasper_question_id}", q.question, q.spans, q.type))
    return papers, questions


def load_pilot(cfg):
    corpus = HERE.parent / "corpus"
    rows = [json.loads(x) for x in (HERE / cfg["dataset"]["pilot"]["eval_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    papers = {r["doc"][:-3]: (corpus / r["doc"]).read_text(encoding="utf-8") for r in rows}
    qs = []
    for r in rows:
        t = papers[r["doc"][:-3]]
        qs.append(EvalQuestion(r["doc"][:-3], r["id"], r["question"],
                               [(t.find(e), t.find(e) + len(e)) for e in r["evidence"]], r["type"]))
    return papers, qs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["manifest", "run", "analyze", "verify"])
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "clean_rerun.yaml")
    ap.add_argument("--sample", choices=["fixture", "pilot", "fresh"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--run", type=Path, help="run directory for analyze")
    ap.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--approved-full-run", action="store_true")
    ap.add_argument("--verify-rows", type=int, default=20)
    ap.add_argument("--limit-papers", type=int, default=None, help="pilot sample only: first N papers")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    if args.stage == "manifest":
        return build_manifest(cfg)
    if args.stage == "verify":
        from src.verify import verify
        problems = verify(args.run, cfg, n_rows=args.verify_rows)
        print("\n".join(problems) if problems else
              f"verify: {args.verify_rows} sampled rows recomputed independently, 0 mismatches")
        sys.exit(1 if problems else 0)
    if args.stage == "analyze":
        used = Path(args.run) / "config_used.yaml"   # the effective config of that run
        print(analyze(args.run, load_cfg(used) if used.exists() else cfg))
        return

    if not args.sample or not args.out:
        sys.exit("run needs --sample and --out")
    cache = None if args.no_cache else args.cache_dir
    if args.sample == "fixture":
        from tests.fixture import fixture_sample, FixtureTokenizer
        from src.embeddings import HashingEmbedder
        papers, questions = fixture_sample()
        tok = FixtureTokenizer()
        emb = HashingEmbedder(tok)
        fcfg = json.loads(json.dumps(cfg))
        fcfg["chunking"]["sizes"] = [8, 16, 32]
        fcfg["modes"]["primary"]["budget"] = 64
        rename = {"s128": "s8", "s256": "s16", "s512": "s32"}
        fcfg["statistics"]["comparisons"] = [[rename[a.split("_")[0]] + "_" + a.split("_")[1],
                                              rename[b.split("_")[0]] + "_" + b.split("_")[1]]
                                             for a, b in cfg["statistics"]["comparisons"]]
        cfg = fcfg
        meta = run(fcfg, papers, questions, emb, tok, args.out, None, "fixture")
    else:
        if args.sample == "fresh":
            if not args.approved_full_run:
                sys.exit("The fresh sample is the confirmatory evaluation. Pass --approved-full-run "
                         "only after the protocol has been approved.")
            papers, questions = load_fresh(cfg)
        else:
            papers, questions = load_pilot(cfg)
            if args.limit_papers:
                keep = sorted(papers)[: args.limit_papers]
                papers = {p: papers[p] for p in keep}
                questions = [q for q in questions if q.paper_id in keep]
        from src.embeddings import Embedder
        emb = Embedder(cfg["embedding"])
        meta = run(cfg, papers, questions, emb, emb.tokenizer, args.out, cache, args.sample,
                   {"lock": json.loads(LOCK.read_text()) if args.sample == "fresh" else None})
    print(f"{meta['n_papers']} papers, {meta['n_questions']} questions -> {args.out} "
          f"({meta['wall_seconds']}s)")
    (Path(args.out) / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()

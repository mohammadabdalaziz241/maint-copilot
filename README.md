# Maintenance Copilot

An agentic retrieval assistant over industrial maintenance documents, built to be **evaluated, not demoed**.

> Status: Phase 1 (data check and retrieval baseline). Nothing below is measured yet; every number
> that appears in this README will link to a committed result file.

## What it will do

A maintenance technician asks a question about USAF ground-support and servicing procedures: fuel
servicing, corrosion control, coatings, torque tools, pressure systems, shop practice. The system
retrieves the relevant pages, and a bounded tool-calling agent decides whether it needs more than
retrieval: querying tables extracted from the manuals (SQL), doing unit and torque calculations, or
pulling the warnings and steps of a procedure. Every answer is a typed object with page-level
citations. Actions with side effects (drafting a work order) wait for human approval.

## Results so far

| System | Queries | nDCG@10 | Recall@100 | Source |
|---|---|---|---|---|
| BM25 (bm25s defaults), page markdown | 283 English | 0.456 (95% CI 0.413–0.501) | 0.735 | [`bm25_baseline.json`](evals/results/bm25_baseline.json) |
| Same index, all six query languages (replication) | 1,698 | 0.157 | — | same file; paper reports 0.156 |

The second row reproduces the published BM25 figure for this corpus (ViDoRe V3, arXiv 2601.08620,
Table 1) to within 0.001, which checks the evaluation harness. The published figure averages six
query languages against English documents; English-only is roughly three times higher. All later
tuning uses the 113-query dev split; the 170-query sealed split is scored once per frozen system
([protocol](evals/PROTOCOL.md)).

## Data

Corpus: ViDoRe V3 Industrial: 27 USAF Technical Orders (English), 5,244 pages with page images and
markdown text, 283 human-annotated queries with reference answers and graded relevance. Licence and content checks live in [`DATA.md`](DATA.md) and
`docs/corpus_check.md`. Data is downloaded locally and never committed.

### Download (run in your own terminal, from this folder)

```bash
python3 -m pip install -U "huggingface_hub>=0.34"
# small files first: dataset card, queries, qrels
hf download vidore/vidore_v3_industrial --repo-type dataset \
  --local-dir data/vidore_v3_industrial --include "README.md" "queries/*" "qrels/*"
# then the corpus (page images + metadata)
hf download vidore/vidore_v3_industrial --repo-type dataset \
  --local-dir data/vidore_v3_industrial --include "corpus/*"
```

If `hf` is not found, use `huggingface-cli download` with the same arguments. If the dataset is gated,
run `hf auth login` first.

### Check the corpus

```bash
python3 -m pip install pandas pyarrow
python3 scripts/check_corpus.py data/vidore_v3_industrial   # writes docs/corpus_check.md
```

### Reproduce the baseline

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 scripts/run_bm25.py        # writes evals/results/bm25_baseline.json
```

## Layout

```
src/maint_copilot/   package code (ingest, retrieval, agent, tools)
scripts/             one-off and reproducible entry points
evals/               golden set manifests, evaluation harness, reports
docs/                design notes and data checks
data/                local downloads (git-ignored)
```

## Licence

Code: [Apache-2.0](LICENSE). Data: ViDoRe V3 Industrial, CC BY 4.0, downloaded locally and not
redistributed; attribution and the terms for derived files are in [`DATA.md`](DATA.md).

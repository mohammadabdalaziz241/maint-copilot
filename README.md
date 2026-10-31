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

**Reference baseline (untuned, full English set).**

| System | Queries | nDCG@10 | Recall@100 | Source |
|---|---|---|---|---|
| BM25 (bm25s defaults), page markdown | 283 English | 0.456 (95% CI 0.413–0.501) | 0.735 | [`bm25_baseline.json`](evals/results/bm25_baseline.json) |
| Same index, all six query languages (replication) | 1,698 | 0.157 | — | same file; paper reports 0.156 |

The second row reproduces the published BM25 figure for this corpus (ViDoRe V3, arXiv 2601.08620,
Table 1) to within 0.001, which checks the evaluation harness.

**Retrieval design choices (113-query dev split only).** Embedding model: `BAAI/bge-small-en-v1.5`
(33M parameters, CPU). Δ is the paired difference against BM25 on the same queries.

| System | nDCG@10 (95% CI) | Recall@100 | Δ nDCG@10 vs BM25 (95% CI) |
|---|---|---|---|
| BM25 | 0.464 (0.399–0.529) | 0.714 | — |
| Dense, whole page (truncated at 512 tokens) | 0.320 (0.267–0.374) | 0.634 | −0.143 (−0.194 to −0.095) |
| Dense, structure-based chunks ≤384 tokens, max-pooled | 0.380 (0.323–0.437) | 0.684 | −0.084 (−0.133 to −0.036) |
| Hybrid RRF (BM25 + dense pages) | 0.406 (0.348–0.466) | 0.747 | −0.057 (−0.096 to −0.019) |
| Hybrid RRF (BM25 + dense chunks) | 0.460 (0.397–0.524) | 0.760 | −0.004 (−0.036 to 0.028) |
| **Hybrid + cross-encoder rerank of top 20** (`bge-reranker-base`, page = max over chunks) | **0.487 (0.424–0.550)** | 0.760 | +0.024 |

Source: [`dense_hybrid_dev.json`](evals/results/dense_hybrid_dev.json). Findings: chunking helps the
dense retriever by 0.06 nDCG@10; a small dense model alone loses to BM25 on these manuals; the
chunk-level hybrid matches BM25 on ranking and finds more relevant pages in its top 100, which is
what a reranker needs. Per-query inspection: BM25 wins on rare exact terms ("Mechanical Beam
Analytical Balance", "reamer", "HSV"); dense wins on paraphrase ("minimum brake lining thickness").
Reranking the top 20 adds +0.027 nDCG@10 over the hybrid (95% CI −0.006 to +0.060,
[`rerank_dev_top20.json`](evals/results/rerank_dev_top20.json)): likely but not yet demonstrated on
113 queries. Depth is a cost decision: on this 4-core CPU the reranker takes ~0.29 s per chunk pair,
so top 20 costs ~14 s per query and top 50 (recall 0.678 vs 0.566) ~33 s; the deeper setting is
deferred to a GPU run. The 170-query sealed split has not been scored.

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
hf download BAAI/bge-small-en-v1.5 --local-dir models/bge-small-en-v1.5
python3 scripts/embed_corpus.py --model models/bge-small-en-v1.5 --variant chunk
python3 scripts/embed_corpus.py --model models/bge-small-en-v1.5 --variant page
python3 scripts/run_dense_hybrid.py   # dev split only; writes evals/results/dense_hybrid_dev.json
hf download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
python3 scripts/run_rerank.py --top-n 20   # resumable; writes evals/results/rerank_dev_top20.json
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

# Evaluation protocol (pre-registered 31 Oct 2026)

**Benchmark.** ViDoRe V3 Industrial, English queries only: 283 queries over 5,244 pages, graded
relevance (1 or 2) from the dataset's qrels; human-written reference answers in `answer`.

**Split.** `evals/splits/split_v1.json`, created once by `scripts/make_split.py` (seed 20261031,
stratified by query type): **113 dev / 170 sealed**.
Sealed id list SHA-256: `2a6fb737f93a33dbc17ee72de5b0bc437574feee58348ae3b2f9f20cc9725198`.

**Rules.**
1. Every design choice (chunking, embedding model, fusion weights, reranker, prompts, agent budgets)
   is made on the dev split only.
2. Untuned reference baselines (BM25 with library defaults) may also be reported on the full 283
   queries, solely for comparison with published numbers.
3. The sealed split is scored once per frozen system, in a single final comparison run; results are
   committed as JSON with per-query scores. No changes are made after seeing sealed results.
4. Reported metrics: nDCG@10 (primary, linear gains), Recall@10, MRR, with 95% bootstrap confidence
   intervals over queries. Differences between systems are tested with a paired bootstrap on the
   same queries.
5. The judge-based answer metrics (faithfulness, correctness against the reference answer, citation
   precision) are added in Phase 2 with their own calibration record.

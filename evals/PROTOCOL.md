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
5. Answer correctness is judged by a different, stronger model than the answerer (answerer
   `claude-haiku-5-5`, judge `claude-sonnet-5-5`), binary PASS/FAIL against the human-written
   reference answer, reasoning before verdict.
6. **Judge calibration (added 31 Oct 2026, before any labels were collected).** 40 dev answers
   (20 judged PASS, 20 FAIL; seed 20261031) are labelled blind by a human
   (`evals/calibration/to_label_v1.csv`). Judge pass rates are reported as trustworthy only if the
   judge agrees with the human on at least 80% of human-PASS items and at least 80% of human-FAIL
   items. Otherwise the judge prompt is revised on these items and re-checked on a fresh sample.
7. Citation precision and citation hit rate are scored without a judge, against the qrels.

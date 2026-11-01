# Final sealed test, pre-registered 1 Nov 2026

Written and committed before any sealed query has been answered, judged or retrieved by a tuned
system. Nothing below changes after the first sealed result exists.

## What is being tested

The sealed split (`evals/splits/split_v1.json`, 170 English queries, SHA-256
`2a6fb737f93a33dbc17ee72de5b0bc437574feee58348ae3b2f9f20cc9725198`, checked in code by
`maint_copilot.data.load_split`) is scored once, with systems frozen at the commit that adds this
file. The dev split (113 queries) produced the findings below; the sealed test checks whether
they hold on questions that played no part in any design choice.

| Claim from dev | Dev result |
|---|---|
| Hybrid retrieval is no better than BM25 | nDCG@10 −0.004 (95% CI −0.036 to +0.028) |
| Reranking the hybrid top 20 helps a little | nDCG@10 +0.027 (95% CI −0.006 to +0.060) |
| Agent v1 answers more questions correctly than the retrieve-once baseline | pass rate +0.115 (95% CI +0.035 to +0.195), cost 2.1× |
| Agent v1 is less faithful to its sources than the baseline | supported share per answer −0.044 (95% CI −0.077 to −0.010) |

BM25 with library defaults was already scored on all 283 queries as an untuned reference
(protocol rule 2); it involves no design choice, so it is not a leak.

## Frozen systems

| Role | System |
|---|---|
| Retrieval | BM25 (bm25s defaults, English stemmer); dense `bge-small-en-v1.5`, chunks of at most 384 tokens, page score = max over chunks; hybrid RRF of the two (k = 60, depth 100); hybrid top 20 reranked by `bge-reranker-base` (page score = max over chunks) |
| Baseline answerer | hybrid top 10 pages, answer prompt v3, `claude-haiku-5-5`, thinking off |
| Agent | v1 (`a1.1`), `claude-haiku-5-5`, tools and budgets as in `docs/agent_design.md` (8 tool calls, 60k input tokens, USD 0.05 per question) |
| Judge | `claude-sonnet-5-5`, judge prompt as calibrated on dev (`evals/calibration/`) |
| Faithfulness verifier | `claude-sonnet-5-5`, claim-level, quotes checked in code |

Not included, to keep the cost down: agent v2 (it failed two of its three dev rules), the
open-weight `gpt-oss-20b` (comparable to Haiku on dev), and answering from reranked pages (the
agent cannot use the reranker, so the paired comparison uses the hybrid baseline, as on dev). The
agent runs once per question; pass^3 remains a dev-only figure.

## Measures and decision rules

All confidence intervals are 95% bootstrap intervals over queries (10,000 resamples, seed 0);
differences are paired on the same queries.

**Retrieval (descriptive).** nDCG@10 (primary), Recall@10, Recall@100 and MRR for the four
retrievers; paired differences vs BM25, and reranked vs hybrid. A dev claim is "confirmed" if the
sealed interval leads to the same conclusion (excludes zero in the same direction, or includes zero
where dev did), otherwise "not confirmed".

**Correctness (primary).** Paired difference in judge pass rate, agent v1 minus baseline. The dev
result **replicates** if the interval lies above zero and the agent's answer cost per question is
at most 3× the baseline's (the rule from `docs/agent_design.md`). Otherwise the README reports that
the gain did not replicate.

**Faithfulness.** Paired difference in supported share per answer (answers with at least one
factual claim in both arms), same verifier for both arms. The dev finding (agent less faithful)
**replicates** if the interval lies entirely below zero; if it includes zero the README says the
difference was not confirmed. Also reported: unsupported claims per answer (paired), answers with
no unsupported claim, and claim-level rates with Wilson intervals.

**Secondary, descriptive only.** Pass rates on the two groups the agent targets (questions where
the baseline's retrieval missed every relevant page; questions written from document summaries);
citation precision and hit rate (scored against the qrels, no judge); answer cost, latency and tool
calls per question; the judge pass rate corrected with the dev calibration labels (no new labels).

Two primary comparisons are made, with no correction for multiple comparisons; this is stated
alongside the results.

## Procedure

1. Commit this file and the split-aware scripts; the commit hash is recorded with the results.
2. Retrieval on the Mac, where the dev retrieval ran (free; the reranker only runs there):
   `run_dense_hybrid.py --split sealed`, then `run_rerank.py --split sealed --top-n 20`. The hybrid
   candidates file is copied to the cloud workspace for the answering steps.
3. API steps in the cloud workspace, in this order, so that the primary comparison finishes first:
   1. `run_answers.py --split sealed --context hybrid --k 10` (baseline answers and judge)
   2. `run_agent.py --split sealed` (agent answers and judge)
   3. `compare_agent.py --split sealed --agent-version a1.1 --repeats 1`
   4. `run_faithfulness.py evals/results/answers_sealed_hybrid_k10_v3.json`
   5. `run_faithfulness.py evals/results/agent_sealed_a1.1_r0.json`
   6. `compare_faithfulness.py --split sealed --agent-version a1.1`
4. Interruptions (credit, network) are resumed from the per-query cache; nothing completed is run
   again. As on dev, a question with no answer after retries is scored FAIL, and a verifier failure
   is reported as skipped.
5. Every sealed result is committed and reported whatever it shows. No code, prompt or setting
   changes after the first sealed answer. If a bug is found afterwards, the as-run result stays the
   reported one and the fix is shown beside it as a labelled deviation.

## Budget

Estimated from the dev runs' recorded token usage, scaled by 170/113. The dev estimate matched
the billed spend within 1.3%.

| Step | Dev (113) | Sealed estimate (170) |
|---|---|---|
| Baseline answers + judge | $0.75 | $1.13 |
| Agent answers + judge | $0.98 | $1.48 |
| Faithfulness, baseline | $4.15 | $6.24 |
| Faithfulness, agent | $2.95 | $4.44 |
| **Total** | | **≈ $13.30** |

The run stops and is reported as incomplete if the estimated spend reaches $15.

## Outcome (1 Nov 2026)

Run at commit `6987267`. Correctness: agent v1 minus baseline +0.077 (95% CI +0.018 to +0.141),
cost 2.13×, so the dev result **replicates**. Faithfulness: supported share per answer −0.029
(95% CI −0.057 to −0.002), so the dev finding that the agent is less faithful **replicates**.
Retrieval: hybrid vs BM25 and dense vs BM25 confirmed; reranking vs hybrid not confirmed in the
sense of this plan, because the sealed gain (+0.080, CI +0.049 to +0.114) is clear where the dev
gain was not. Full figures: `evals/results/sealed_summary.json` and the README.

Deviations: (1) the paid steps started while the free sealed reranking was still running; no
paid step uses the reranker. (2) `run_faithfulness.py` crashed on one agent answer (query 107)
submitted without reading any page; a guard added after the crash records it as a verifier
failure, reported as skipped as this plan specifies. (3) The faithfulness runs were restarted in
parallel; cached checks were kept and a few in-flight calls were repeated. Estimated cost $12.58.

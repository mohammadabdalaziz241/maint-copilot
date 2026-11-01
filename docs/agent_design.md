# Agent design (Phase 3), fixed 31 Oct 2026 before any agent code was written

## Why an agent, measured

Dev split, retrieved top 10, prompt v3 (`evals/results/answers_dev_rerank_k10_v3.json`):

| Group | Judge pass rate |
|---|---|
| Retrieval missed every relevant page (22 of 113) | 0.136 (vs 0.615 when a relevant page was retrieved) |
| Questions written from document summaries (72) | 0.417 (vs 0.707 for single-image questions) |
| Evidence includes tables (42) | 0.476 (vs 0.549 without) |
| Evidence includes figures (33) | 0.424 (needs visual retrieval, out of scope here) |
| Questions needing arithmetic | about 1 of 113 |

The agent targets the first two: **search again with reworded queries** when results miss, and
**read further** (adjacent pages) to cover multi-page answers. A calculator or SQL table tool is
not justified by these numbers and is added only if the agent's own failure analysis calls for it.

## Version 1

- Plain Anthropic Messages API tool-use loop (no agent framework). Agent model `claude-haiku-5-5`.
- Tools:
  - `search_pages(query, k<=10)`: shared hybrid retriever (BM25 + dense chunks, RRF); returns page
    ids, sources and short snippets.
  - `read_page(page_id)`: full page text.
  - `read_adjacent(page_id, before<=2, after<=2)`: neighbouring pages of the same manual.
  - `submit_answer(answer, citations, insufficient_evidence)`: ends the run; citations must be
    pages the agent has read in full, checked in code.
- Budgets per question: at most 8 tool calls, 60k input tokens, USD 0.05 (at the recorded price
  table). On reaching a budget the agent is told to answer from what it has read.
- Every step is traced: tool, input, pages returned, tokens, latency.
- `draft_work_order` is a separate demonstration path with human approval enforced in code and a
  test; it is not part of the benchmark score.

## Constraint

The cross-encoder reranker (1.1 GB weights) cannot be copied into the cloud workspace (400 MB
per-file limit), so the agent searches with the hybrid retriever **without** reranking. The
non-agentic baseline is re-run under the same condition (hybrid top 10, prompt v3).

## Experiment (pre-registered)

- Arms: baseline (retrieve once with the hybrid retriever, answer from top 10, prompt v3) vs agent;
  both `claude-haiku-5-5`, same frozen judge (`claude-sonnet-5-5`), all 113 dev queries.
- Primary measure: judge pass rate, paired difference with 95% bootstrap CI.
- Secondary: pass rate on the two target groups; cost, latency and tool calls per question;
  relative faithfulness with the same verifier; pass^3 (agent run three times; a query counts
  only if all three runs pass).
- Success rule: the agent beats the baseline with a CI excluding zero, at no more than 3x the cost
  per question. Otherwise the README reports that the agent did not help.
- The sealed split is not used in Phase 3 design work.

## Amendments before the first full run (31 Oct 2026, results not yet seen)

1. `read_page(page_id)` became `read_pages(page_ids)` with at most 5 ids per call, to save turns;
   the agent sees the same pages either way. Requests for more than 5 are truncated and the agent
   is told so.
2. Version a1.1: in a smoke test on 3 dev questions, one final `submit_answer` arrived empty
   because the output hit the 2,000-token cap. The cap is now 4,000 tokens, and an empty or cut-off
   submission is returned to the agent as an error instead of being accepted.
3. The baseline (hybrid top 10, no reranking, prompt v3) was run before the agent:
   judge pass rate 0.478 (95% CI 0.389–0.575).

## Version 2 (a2), fixed 1 Nov 2026 before any a2 code was run

**Why.** v1 (a1.1) beat the baseline on correctness (+0.115, 95% CI +0.035 to +0.195) but was
measurably less faithful by the same verifier: supported share per answer −0.044 (95% CI −0.077
to −0.010), unsupported claims per answer +0.164 (+0.036 to +0.300).

**Change.** After `submit_answer`, a separate self-check call (`claude-haiku-5-5`, the agent's own
model, not the evaluation verifier) splits the draft into claims, checks each against the pages
the agent read with a quoted piece of evidence, and rewrites the answer keeping only supported
content (no new facts; citations restricted to pages read). Quotes are checked against the page
text in code and the rate is logged. Everything else (tools, budgets, prompts) is unchanged.

**Experiment.** One run of a2 on the 113 dev questions, judged by the same frozen judge and
checked by the same verifier (`claude-sonnet-5-5`), compared pairwise with the same hybrid
baseline as v1.

**Success rule (all three):**
1. Faithfulness no worse than the baseline: the paired difference in supported share per answer
   has a 95% CI that includes zero or lies above it.
2. Correctness still above the baseline: paired pass-rate difference with 95% CI above zero.
3. Answer cost per question at most 3× the baseline (self-check call included).

If a2 fails any rule, v1 is kept and reported with its trade-off. Repeats (pass^3) are run only
if a2 passes.

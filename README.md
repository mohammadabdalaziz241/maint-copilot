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

**Grounded answers (113 dev queries).** `claude-haiku-5-5` answers from the top-k pages with page
citations; structured output restricts citations to the page ids it was given. `claude-sonnet-5-5`
judges each answer against the human reference answer. The judge was checked against 40 blind
human labels before its numbers were used (below).

| Context | Prompt | Judge pass rate (95% CI) | Declined | Citation precision | Source |
|---|---|---|---|---|---|
| Retrieved top 5 | v2 | 0.443 (0.354–0.531) | 31.9% | 0.592 | [`answers_dev_rerank_k5_v2.json`](evals/results/answers_dev_rerank_k5_v2.json) |
| Retrieved top 10 | v2 | 0.504 (0.416–0.593) | 27.4% | 0.541 | [`answers_dev_rerank_k10_v2.json`](evals/results/answers_dev_rerank_k10_v2.json) |
| **Retrieved top 10** | **v3** | **0.522 (0.434–0.611)** | **0.9%** | 0.541 | [`answers_dev_rerank_k10_v3.json`](evals/results/answers_dev_rerank_k10_v3.json) |
| Benchmark's relevant pages (oracle) top 10 | v3 | 0.584 (0.496–0.673) | 0.9% | 1.000 | [`answers_dev_oracle_k10_v3.json`](evals/results/answers_dev_oracle_k10_v3.json) |

What the runs showed:
- **Ten retrieved pages answer about as well as the benchmark's own relevant pages** (v2: 0.504 vs
  0.504), so retrieval stopped being the bottleneck at k = 10.
- **The v2 prompt refused too often**: even with the correct pages it declined on 31–40% of
  questions, rejecting near-miss wording and refusing to synthesise "explain/summarise" answers.
  v3 allows partial answers that say what is not covered: on oracle pages, +0.080 pass rate (paired
  95% CI +0.018 to +0.142; 11 answers FAIL→PASS, 2 PASS→FAIL); on retrieved pages +0.018 (within
  noise).
- **Questions whose evidence spans many pages are harder, but less than first measured.** With the
  original prompt and 5 oracle pages, questions with more than 5 relevant pages passed 0.364 vs
  0.565 for the rest; with 10 pages and prompt v3 the gap narrows to 0.545 vs 0.609 (oracle) and
  0.455 vs 0.565 (retrieved; 44 vs 69 queries). Whether iterative (agentic) retrieval closes the
  remaining gap is an experiment for Phase 3, not an assumption.
- A citation bug (the model citing printed page numbers instead of page ids, 16–24 invalid
  citations per run) was removed by constraining citations in the output schema.
- Known failure: for "Should paint be thinned excessively…?" the answerer returned a refusal
  (`stop_reason=refusal`); it is scored as FAIL.

**Judge calibration** ([`agreement_v1.json`](evals/calibration/agreement_v1.json)). On 40 dev
answers (20 the judge passed, 20 it failed) labelled blind by a human: agreement 87.5%, Cohen's
κ = 0.75. The judge failed every answer the human failed (15/15) and passed 20 of the 25 the human
passed (0.80, Wilson 95% CI 0.61–0.91), meeting the pre-registered threshold of 0.8 on both, at the
boundary. All five disagreements are open-ended "explain/summarise" questions where the judge
required every detail of a long reference answer and the human accepted the core facts. The judge
is therefore kept frozen and read as a **conservative lower bound**. Adjusting for its measured
strictness (stratified by verdict), the retrieved-top-10 v3 pass rate of 0.522 corresponds to an
estimated **0.64 by the human standard (range 0.49–0.75)**.

**Faithfulness** ([`faithfulness_dev_rerank_k10_v3.json`](evals/results/faithfulness_dev_rerank_k10_v3.json)).
`claude-sonnet-5-5` splits each answer into atomic claims and labels every claim against the 10
pages the answerer saw, naming the supporting page and quoting its evidence; each quote is then
checked against the page text in code.

| Answers (retrieved top 10) | Answers checked | Claims supported per answer (95% CI) | Answers with no unsupported claim | Fact claims: supported / partial / unsupported | Verifier quote found on page |
|---|---|---|---|---|---|
| Prompt v2 | 82 (31 declined) | 0.947 (0.929–0.964) | 0.939 | 668 / 35 / 5 of 708 | 0.989 |
| Prompt v3 | 112 (1 declined) | 0.933 (0.910–0.954) | 0.920 | 901 / 49 / 9 of 959 | 0.980 |

- **Cutting refusals did not cost faithfulness on the questions both prompts answered**: paired
  difference −0.003 (95% CI −0.019 to +0.013, 82 queries). The 30 questions v3 answers but v2
  declined are slightly less grounded (0.902 supported; 2 contain an unsupported claim).
- **Judge FAILs are mostly incomplete, not invented**: claims are supported at 0.945 in answers the
  judge passed and 0.919 in answers it failed.
- The 9 unsupported v3 claims include outside knowledge leaking in (aircraft manufacturers),
  a rule applied to the wrong procedure, a misread strength value and one contradicted yes/no
  answer; they are listed per query in the results file.
- The verifier's evidence quotes were found on the named page for 98% of supported or partially
  supported claims.
- **Human spot check of the verifier** ([`faithfulness_agreement_v1.json`](evals/calibration/faithfulness_agreement_v1.json)):
  20 claims labelled blind (all 9 the verifier called unsupported, 11 it called supported), 85%
  agreement. The human confirmed 7 of the 9 unsupported claims; one of the other two was the
  verifier listing its own critique of an answer as a claim (an extraction error). Of the 11
  "supported" claims the human confirmed 10; the miss was an answer that confused the emergency
  shutdown step with the final system-shutdown step. **The verifier's absolute rate (1.0% of
  claims unsupported) is therefore an underestimate:** by the human standard the estimate is about
  10%, with a 95% range of 2–38% from so small a sample, so no absolute faithfulness figure is
  claimed here. Comparisons made with the same verifier (v2 vs v3 above) remain valid as relative
  measures.

Dev cost per 113-question run: about $0.07 for answers and $0.56 for judging, from token counts
and third-party price lists ([`llm.py`](src/maint_copilot/llm.py)). Check against billing: the
Anthropic console showed $21.29 spent by 1 Nov 2026, against $21.02 estimated from the logged
tokens (1.3% lower; uncached smoke tests and failed calls are not in the logs). Answer latency p50 2.0–2.6 s, p95 3.9–4.4 s.

## Phase 3: agent (dev split)

Design and experiment were fixed before any agent code ([`docs/agent_design.md`](docs/agent_design.md)).
The agent is a plain Anthropic tool-use loop (no framework): `claude-haiku-5-5` with
`search_pages`, `read_pages`, `read_adjacent` and `submit_answer`, at most 8 tool calls per
question, and citations restricted in code to pages it has read in full
([`agent.py`](src/maint_copilot/agent.py)). The baseline retrieves once with the same hybrid
retriever (no reranker; see the design note on why) and answers from the top 10 pages with the
same answer rules.

| | Baseline | Agent | Source |
|---|---|---|---|
| Judge pass rate (113 dev questions) | 0.478 | **0.593** (repeats: 0.628, 0.655) | [`agent_vs_baseline_dev_a1.1.json`](evals/results/agent_vs_baseline_dev_a1.1.json) |
| Paired difference, first run | | **+0.115** (95% CI +0.035 to +0.195); 18 FAIL→PASS, 5 PASS→FAIL | |
| Passes in all 3 agent runs (pass^3) | | 0.549 | |
| Answer cost per question | $0.0014 | $0.0029 (2.1×) | estimated from tokens |
| Median latency | 2.5 s | 6.5 s | |
| Tool calls per question | | 3.9 mean; 1–2 of 113 runs hit the budget | |

**The pre-registered success rule is met on correctness** (CI excludes zero, cost ≤ 3×). What the
analysis shows about *why*:
- **Not better retrieval.** The agent's pages contain a relevant page less often than the
  baseline's 10 (0.72 vs 0.82). In 15 of the 18 questions it turned from FAIL to PASS, the
  baseline already had a relevant page but denied the pages said it or picked the wrong item. The
  agent reads 3.7 chosen pages on average and answers from them decisively.
- **Target groups improved less than hoped**: questions where the baseline retrieved no relevant
  page 0.10 → 0.20 (n = 20); summary questions 0.40 → 0.47 (n = 72).
- **Answer length is not the explanation**: agent answers are longer (213 vs 127 words), but in
  both arms the shortest answers pass most often, and on the 22 questions where both answers have
  similar length the agent still passes more (0.727 vs 0.591).
- **Faithfulness is worse** ([`faithfulness_agent_vs_baseline_dev_a1.1.json`](evals/results/faithfulness_agent_vs_baseline_dev_a1.1.json)).
  Same verifier, paired on the 110 dev questions where both answers contain factual claims:

  | | Agent | Baseline | Paired difference (95% CI) |
  |---|---|---|---|
  | Claims supported per answer | 0.889 | 0.933 | −0.044 (−0.077 to −0.010) |
  | Answers with no unsupported claim | 0.809 | 0.909 | −0.100 (−0.182 to −0.018) |
  | Unsupported claims per answer | 0.282 | 0.118 | +0.164 (+0.036 to +0.300) |
  | Unsupported claim rate (all checked answers) | 2.6% (34/1,306) | 1.4% (13/948) | |

  The agent's answers carry about 40% more claims (11.9 vs 8.6 per answer) and a larger share of
  them are unsupported. Absolute rates are underestimates (the verifier misses some unsupported
  claims; see the spot check), but both arms are measured by the same verifier.

**Agent v2 (a2): a self-check before answering.** After the agent submits, a separate Haiku
call checks each claim of the draft against the pages read (with quotes verified in code) and
rewrites the answer without unsupported content. The design and its success rule were committed
before the run ([`docs/agent_design.md`](docs/agent_design.md)): faithfulness no worse than the
baseline, correctness still above it (CI excluding zero), and cost at most 3×.

| Dev, 113 questions, one run | Baseline | Agent v1 | Agent v2 |
|---|---|---|---|
| Judge pass rate | 0.478 | 0.593 | 0.549 |
| Paired vs baseline (95% CI) | | +0.115 (+0.035 to +0.195) | +0.071 (−0.009 to +0.159) |
| Claims supported per answer | 0.933 | 0.889 | **0.951** |
| Answers with no unsupported claim | 0.909 | 0.809 | **0.973** |
| Unsupported claims per answer (paired diff vs baseline) | 0.118 | 0.282 (+0.164) | **0.027 (−0.091; −0.173 to −0.018)** |
| Unsupported claim rate | 1.4% (13/948) | 2.6% (34/1,306) | **0.2% (3/1,350)** |
| Answer cost per question (× baseline) | $0.0014 | $0.0029 (2.1×) | $0.0044 (3.1×) |
| Median latency | 2.5 s | 6.5 s | 11.4 s |

Sources: [`agent_vs_baseline_dev_a2.json`](evals/results/agent_vs_baseline_dev_a2.json),
[`faithfulness_agent_vs_baseline_dev_a2.json`](evals/results/faithfulness_agent_vs_baseline_dev_a2.json).
The self-check removed 121 claims across 78 of 111 answers; 90% of its evidence quotes were found
on the named page.

**Verdict, as pre-registered: v2 fails** two of the three conditions, both narrowly: its
correctness gain over the baseline is no longer significant (CI −0.009 to +0.159) and its cost is
3.1×, just over the 3× limit. It does fix faithfulness: unsupported claims per answer fall below
the baseline's, with a CI excluding zero. **v1 therefore remains the reported agent, with its
trade-off stated**: more often correct, less faithful. The two versions mark the two ends of a
precision/recall trade-off; for a deployment where an unsupported maintenance instruction is
unacceptable, v2's profile is the safer one (a judgement, not part of the pre-registered test).
v1's three runs ranged 0.593–0.655, so a single run of either version carries run-to-run noise
of a similar size to the v1–v2 gap.

A human-approval path for work orders is included: the agent can only propose a draft, a named
person must approve it, decisions are final, and drafts may only cite pages the agent read
([`work_orders.py`](src/maint_copilot/work_orders.py), tests in `tests/test_work_orders.py`).

## Phase 4: open-weight model on a university GPU (dev split)

Pre-registered in [`docs/open_weight_design.md`](docs/open_weight_design.md). The selection rule
picked the first model in a fixed order that loads and passes a smoke test on one **RTX 4000 Ada
(20 GB)**: `openai/gpt-oss-20b` (Apache-2.0, native MXFP4), served with **vLLM 0.32.0**
(PyTorch 2.13, CUDA 13), 17.8 of 20.5 GB GPU memory in use. Only the answering model changes: the
113 dev prompts are byte-identical to the Haiku baseline's (verified by checksum), citations are
constrained by the same JSON schema, and the same frozen judge scores both. Scripts in
[`cluster/`](cluster/); answers in `evals/cluster/`.

| Dev, hybrid top 10, prompt v3 | `claude-haiku-5-5` (API) | `gpt-oss-20b` (self-hosted) |
|---|---|---|
| Judge pass rate | 0.478 | 0.522 (95% CI 0.434–0.611) |
| Paired difference (gpt-oss − Haiku) | | +0.044 (95% CI −0.035 to +0.124); 14 FAIL→PASS, 9 PASS→FAIL |
| **Pre-registered verdict (margin −0.10)** | | **comparable** |
| Schema-valid answers | 113/113 | 113/113 |
| Citation precision / at least one relevant page cited | 0.524 / 0.788 | 0.510 / 0.628 |
| Latency p50 / p95, one request at a time | 2.5 s / 4.4 s | 11.6 s / 26.9 s |
| Throughput, 8 concurrent requests | | 12.9 questions/min |
| Cost per 1,000 questions | $1.41 (tokens) | $0.18–0.96 batched, $0.51–2.67 one at a time (GPU-hours × $0.14–0.74/h) |

Source: [`open_weight_vs_haiku_dev_gpt-oss-20b.json`](evals/results/open_weight_vs_haiku_dev_gpt-oss-20b.json).
What it shows:
- **Quality is comparable**: the lower confidence bound (−0.035) clears the pre-registered −0.10
  margin; the point estimate favours gpt-oss but the CI includes zero, so it is not "better".
- **It is 4.6× slower per request** (median 11.6 s vs 2.5 s), mostly from its built-in reasoning
  (about 890 output tokens per answer at the default setting), and it cites a relevant page less
  often (0.63 vs 0.79 of answers).
- **Self-hosting only wins on cost when the GPU is kept busy**: batched, it is cheaper than Haiku at
  any price in the range; one request at a time it ranges from cheaper to nearly twice the cost.
  The GPU price range is from a third-party summary (17 Oct 2026) and is not verified.
- Faithfulness was not checked for this arm (budget), as stated in the pre-registration.

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
# answers + judge (needs ANTHROPIC_API_KEY in .env; answers are cached per query)
python3 scripts/run_answers.py --context rerank --k 10
python3 scripts/run_answers.py --context oracle --k 10
python3 scripts/run_answers.py --context hybrid --k 10    # Phase 3 baseline
python3 scripts/run_agent.py --repeats 3
python3 scripts/compare_agent.py --agent-version a1.1 --repeats 3
python3 scripts/run_faithfulness.py evals/results/agent_dev_a1.1_r0.json
python3 scripts/compare_faithfulness.py --agent-version a1.1
python3 scripts/run_agent.py --self-check --repeats 1    # agent v2
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

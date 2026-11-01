# Open-weight comparison (Phase 4), fixed 1 Nov 2026 before any cluster run

## Question

Can a self-hosted open-weight model replace `claude-haiku-5-5` as the answering model in the
non-agentic pipeline (hybrid top 10, prompt v3, citations restricted to offered page ids), and at
what cost and latency?

## Model selection rule (decided before trying any model)

Use the first model in this order that (a) is Apache-2.0 licensed, (b) fits on the RTX 4000 Ada
(20 GB) with a context of at least 24k tokens, and (c) loads in vLLM and returns schema-valid
answers on a 3-question smoke test:

1. `openai/gpt-oss-20b` (native MXFP4). vLLM's recipe does not yet confirm Ada support.
2. Gemma 4 12B instruction-tuned, FP8 weights (Ada supports FP8 natively).
3. Gemma 4 26B-A4B instruction-tuned, 4-bit.
4. Qwen3.6-35B-A3B, 4-bit.

Exact repository ids are taken from the Hugging Face API on the cluster, not from memory. Every
model tried and why it was rejected is recorded.

## Fixed settings

- Same inputs as the Haiku baseline: the dev split's hybrid top-10 pages, exported from this
  repository so prompts are byte-identical (`scripts/export_cluster_inputs.py`).
- Same system prompt and JSON schema (answer, citations restricted to offered ids,
  insufficient_evidence). Sampling: the server's defaults (temperature 1.0), matching the Haiku
  runs, which used the API default.
- Served with vLLM's OpenAI-compatible server on one RTX 4000 Ada; requests sent one at a time
  for latency, then 8 at a time for throughput.
- Judged by the same frozen judge (`claude-sonnet-5-5`) against the same reference answers.

## Measures

- Primary: judge pass rate, paired with the Haiku baseline on the same 113 dev questions
  (baseline 0.478, `answers_dev_hybrid_k10_v3.json`).
- **Non-inferiority margin: −0.10.** The open-weight model is called "comparable" if the lower
  bound of the 95% CI for (open-weight − Haiku) is above −0.10; "worse" if the upper bound is below
  0; otherwise "inconclusive".
- Secondary: schema-valid answer rate, abstention rate, citation precision (against qrels),
  latency p50/p95 (single stream), throughput (questions per minute at concurrency 8), GPU-hours
  per 1,000 questions, and cost per 1,000 questions at a stated range of cloud GPU hourly prices,
  against Haiku's token cost.
- Faithfulness is not checked in this phase (budget); this is stated with the results.

## Outcome (1 Nov 2026)

- Model selected by the rule: `openai/gpt-oss-20b` (first in order; loaded on the RTX 4000 Ada with
  vLLM 0.32.0 and returned 3/3 schema-valid answers in the smoke test). No other model was tried.
- One implementation detail differs from the Haiku runs: a single line describing the JSON format
  is appended to the user message, because vLLM constrains decoding to the schema but, unlike the
  Anthropic API, does not tell the model what the schema is (vLLM's documentation recommends this).
- Judge pass rate 0.522 vs Haiku 0.478; paired difference +0.044 (95% CI −0.035 to +0.124).
  Lower bound above −0.10: **comparable**.
- The vLLM version was not captured by the answer script (the interpreter could not import vllm);
  it is taken from the setup log (0.32.0).

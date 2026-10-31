# Design decisions

## 31 Oct 2026: tools retargeted to the corpus

The original plan used a bearing-fault classifier (from the author's MSc dissertation) as an agent
tool, cross-checked against maintenance manuals. The corpus check (`docs/corpus_check.md`) showed
that this does not fit ViDoRe V3 Industrial: **0 of 283** English benchmark queries mention bearings,
vibration or rotating machinery, and the 27 manuals cover fuel servicing, corrosion control,
coatings, hand and torque tools, pressure systems, electrical shop practice and vehicle maintenance.
Forcing a vibration tool onto this corpus would mean inventing the questions that justify it.

Tools are therefore chosen from what the corpus and its queries actually need:

| Tool | Why the corpus needs it | Ground truth |
|---|---|---|
| `search_pages` (hybrid retrieval) | Every query | qrels (graded relevance) |
| `query_tables` (read-only SQL over tables extracted from the manuals into Postgres) | 88 of 283 English queries have tables among their evidence; e.g. O-ring dash-number dimensions, coating thicknesses | reference answers |
| `calculate` (units, torque-wrench extension correction, tolerance checks) | 29 numerical queries; TO 32B14-3-1-101 covers torque-indicating devices | deterministic, unit-tested |
| `get_procedure` (ordered steps + WARNING/CAUTION blocks for a task) | 28 multi-hop and other procedural queries (fuel-leak shutdown, painting precautions) | reference answers |
| `draft_work_order` (side effect, human approval required) | demonstrates approval gating | integration test |

The agent must beat a non-agentic retrieve-then-answer baseline on the dev split to justify itself;
where it does not, the README says so.

## Baseline sanity check

BM25 over the shipped page markdown reproduces the published multilingual BM25 figure (0.1565 vs
0.156). English-only nDCG@10 is 0.456. 212 of 5,244 pages have under 50 characters of markdown
(schematics and figures), which is why a visual-retrieval arm stays in scope.

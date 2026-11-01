# Example agent run: held-out question 218

Rendered by `scripts/show_trace.py` from committed result files. Page text comes from ViDoRe V3 Industrial (CC BY 4.0; USAF Technical Orders, Distribution Statement A).

**Question.** Why should hands not be used to check the output of airless spray guns?

**Reference answer (human-written).** Airless spray guns use very high pressure, and to prevent injury, hands should not be used to check the output.

## Baseline: retrieve once, answer from the top 10 pages

Relevant page among the 10 retrieved: **no**. Judge verdict: **FAIL**.

> The pages do not address checking the output of airless spray guns with hands. They do state that spray gun nozzles are sources of very high pressure, and that during operation and cleaning the nozzle shall never be pointed toward any person in close proximity; failure to comply could result in injury to, or death of, personnel or long term health hazards. Nozzles increase hydraulic pressure by a factor of approximately 100 in airless spraying.

## Agent v1

| Step | Tool | Input | Result |
|---|---|---|---|
| 1 | `search_pages` | `airless spray gun hands check output injection hazard` | 8 pages with snippets, e.g. TO 1-1-8 p.92; TO 1-1-3 p.127; read later: **TO 36-1-191 p.71** |
| 2 | `search_pages` | `never use hand to check spray gun tip leaks` | 8 pages with snippets, e.g. TO 1-1-8 p.104; TO 1-1-8 p.103 |
| 3 | `submit_answer` | answer citing TO 36-1-191 p.71 | **rejected by code**: cited a page not yet read |
| 4 | `read_pages` | TO 36-1-191 p.71 | full text of TO 36-1-191 p.71 |
| 5 | `submit_answer` | answer citing TO 36-1-191 p.71 | submitted |

4 tool calls, 9.6 s, about $0.0020. Steps 1 and 2 were issued together in one model turn.

**Final answer** (cites TO 36-1-191 p.71). Judge verdict: **PASS**.

> Airless spray guns use very high pressure, so hands must not be used to check the output. The WARNING in TO 36-1-191 states: "Airless spray guns use very high pressure. To prevent injury, hands will not be used to check the output." The page does not spell out the specific injury mechanism (high-pressure fluid injection into the skin), so that reasoning is implied by the high-pressure warning rather than stated.

**Faithfulness check** (separate verifier model; each evidence quote is then looked up on the page in code):

| Claim | Verdict | Evidence quote found on the page |
|---|---|---|
| Airless spray guns use very high pressure. | SUPPORTED | yes |
| Hands must not be used to check the output of airless spray guns, to prevent injury. | SUPPORTED | yes |
| The WARNING in TO 36-1-191 states: "Airless spray guns use very high pressure. To prevent injury, hands will not be used to check the output." | SUPPORTED | yes |
| The page does not spell out the specific injury mechanism (high-pressure fluid injection into the skin). (scope note) | SUPPORTED | yes |

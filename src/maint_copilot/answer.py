"""Grounded answering over retrieved pages, with page-level citations and abstention."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, create_model

from maint_copilot.llm import Usage

ANSWER_SYSTEM = """You answer questions about U.S. Air Force technical orders for maintenance \
technicians. Use only the numbered pages provided. Do not use outside knowledge.

Rules:
- Answer with everything the pages support, and list the ids of the pages you used in \
`citations`. Every factual statement must come from a cited page.
- Copy numbers, units, part numbers and specification codes exactly as written on the page.
- Include WARNING and CAUTION text when the page attaches it to the procedure asked about.
- For "explain", "summarize", "compare" or "list" questions, combine the relevant facts from \
all pages into one answer.
- If the question's wording does not match the pages exactly (a different name, a broader or \
narrower scope), answer from the closest matching content and state the difference in one short \
sentence.
- If only part of the question is covered, answer that part and say which part the pages do not \
cover. This is still an answer: set `insufficient_evidence` to false.
- Set `insufficient_evidence` to true only when none of the pages contain information relevant \
to the question; then cite nothing.
- Be concise: the technician needs the answer, not a description of the pages."""

MAX_PAGE_CHARS = 6000
PROMPT_VERSION = "v3"  # v3: partial answers allowed; abstain only when nothing relevant


class GroundedAnswer(BaseModel):
    answer: str = Field(description="The answer, using only the provided pages.")
    citations: list[int] = Field(description="Ids of the pages the answer relies on.")
    insufficient_evidence: bool = Field(
        description="True if the provided pages do not contain the answer."
    )


def format_context(pages: Sequence[Mapping]) -> str:
    """pages: dicts with corpus_id, doc_id, page_number_in_doc, markdown (in rank order)."""
    blocks = []
    for p in pages:
        text = p["markdown"][:MAX_PAGE_CHARS] or "[no extractable text on this page]"
        header = (
            f'<page id="P{p["corpus_id"]}" source="{p["doc_id"]}, page {p["page_number_in_doc"]}">'
        )
        blocks.append(f"{header}\n{text}\n</page>")
    return "\n\n".join(blocks)


def _answer_schema(page_ids: Sequence[int]) -> type[BaseModel]:
    """Schema whose citations can only be the page ids actually offered (enforced by decoding)."""
    allowed = Literal[tuple(f"P{i}" for i in page_ids)]  # type: ignore[valid-type]
    return create_model(
        "GroundedAnswer",
        answer=(str, Field(description="The answer, using only the provided pages.")),
        citations=(list[allowed], Field(description="Ids of the pages the answer relies on.")),
        insufficient_evidence=(
            bool,
            Field(description="True if the provided pages do not contain the answer."),
        ),
    )


def answer_question(
    client,
    model: str,
    question: str,
    pages: Sequence[Mapping],
    max_tokens: int = 2000,
    thinking: bool = False,
) -> dict:
    """Answer from the given pages. Thinking is off by default; it is evaluated as an option."""
    context = format_context(pages)
    schema = _answer_schema([int(p["corpus_id"]) for p in pages])
    t0 = time.perf_counter()
    resp, out, error = None, None, None
    for attempt_tokens in (max_tokens, max_tokens * 2):
        try:
            resp = client.messages.parse(
                model=model,
                max_tokens=attempt_tokens,
                system=ANSWER_SYSTEM,
                output_format=schema,
                thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
                messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
            )
        except ValidationError as err:  # output cut off mid-JSON: retry once with more room
            error = f"invalid structured output: {str(err).splitlines()[0][:160]}"
            continue
        out = resp.parsed_output
        if out is not None:
            error = None
            break
        error = f"no parsed output (stop_reason={resp.stop_reason})"
        if resp.stop_reason != "max_tokens":
            break
    latency = time.perf_counter() - t0
    if out is None:
        return {
            "answer": "",
            "citations": [],
            "invalid_citations": [],
            "prompt_version": PROMPT_VERSION,
            "insufficient_evidence": False,
            "error": error,
            "context_page_ids": [int(p["corpus_id"]) for p in pages],
            "model": model,
            "thinking": thinking,
            "usage": Usage.from_response(resp.usage).__dict__ if resp else Usage().__dict__,
            "latency_s": round(latency, 3),
            "stop_reason": resp.stop_reason if resp else None,
        }
    offered = {int(p["corpus_id"]) for p in pages}
    cited = [int(str(c).lstrip("P")) for c in dict.fromkeys(out.citations)]
    return {
        "answer": out.answer,
        "citations": [c for c in cited if c in offered],
        "invalid_citations": [c for c in cited if c not in offered],
        "prompt_version": PROMPT_VERSION,
        "insufficient_evidence": out.insufficient_evidence,
        "error": None,
        "context_page_ids": [int(p["corpus_id"]) for p in pages],
        "model": model,
        "thinking": thinking,
        "usage": Usage.from_response(resp.usage).__dict__,
        "latency_s": round(latency, 3),
        "stop_reason": resp.stop_reason,
    }


def citation_scores(citations: Sequence[int], rels: Mapping[int, int]) -> dict[str, float | None]:
    """Judge-free citation quality against the benchmark's relevance labels.

    precision: share of cited pages that are labelled relevant (None if nothing cited)
    hit: 1 if at least one cited page is relevant
    """
    cited = list(dict.fromkeys(citations))
    if not cited:
        return {"citation_precision": None, "citation_hit": 0.0, "n_cited": 0}
    good = sum(1 for c in cited if rels.get(c, 0) > 0)
    return {
        "citation_precision": good / len(cited),
        "citation_hit": float(good > 0),
        "n_cited": len(cited),
    }
